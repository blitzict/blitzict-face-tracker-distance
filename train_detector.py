"""
train_face_detector.py  —  Train the custom FaceDetectorCNN with Hard Negative Mining
======================================================================================

TRAINING OVERVIEW (step by step)
──────────────────────────────────
 Step 1  BUILD DATASET
         Positive examples (face patches):
           • photos_all_faces/  — 1,364 pre-cropped faces from DnHFaces
           • Haar-annotated crops from full scene photos (photos_all/)
           • datasets/lfw/      — 13k diverse LFW faces (run download_datasets.py)
         Negative examples (background patches):
           • Random crops from full-scene images that don't overlap any face

 Step 2  PHASE 1 — Bootstrap training (EPOCHS_PHASE1 epochs)
         Train FaceDetectorCNN from scratch on the initial dataset.

 Step 3  HARD NEGATIVE MINING
         Run the Phase-1 model on full-scene images with a sliding window.
         Collect all patches where the model fires (false positives) and add
         them as hard negatives.  Forces the model to learn fine distinctions
         between real faces and face-like textures/backgrounds.

 Step 4  PHASE 2 — Retrain on augmented dataset (EPOCHS_PHASE2 epochs)
         Train a fresh FaceDetectorCNN from scratch on Phase-1 data +
         hard negatives.  Typically reaches P > 0.95, R > 0.98.

 Step 5  Save best checkpoint to  checkpoints/face_detector.pth

USAGE
─────
    # Download extra face data first (recommended):
    python download_datasets.py

    # Then train:
    python train_face_detector.py
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from facetrack.detector import FaceDetectorCNN, DETECTOR_TRAIN_TF, DETECTOR_INFER_TF, DETECTOR_PATCH_SIZE

# ── Paths ──────────────────────────────────────────────────────────────────────
PHOTOS_ALL_DIR  = Path('DnHFaces/open_data_set/photos_all')
FACES_DIR       = Path('DnHFaces/open_data_set/photos_all_faces')
LFW_DIR              = Path('datasets/lfw')              # populated by download_datasets.py
WIDER_FACE_DIR       = Path('datasets/wider_face')       # populated by download_datasets.py --wider
WIDER_FACE_CROPS_DIR = Path('datasets/wider_face_crops') # flat face crops extracted from wider_face
CELEBA_DIR           = Path('datasets/tmp_celeba/img_align_celeba/img_align_celeba')
UTKFACE_DIR          = Path('datasets/utkface')          # populated by download_datasets.py --utk
FAIRFACE_DIR         = Path('datasets/fairface')         # manual (see download_datasets.py --fairface)
HARD_NEG_CACHE_DIR   = Path('checkpoints/hard_neg_cache') # mined hard negatives, persisted to disk
# Dataset caps. With the lazy-loading FacePatchDataset, these can be None ( = take all )
# without RAM impact — samples are stored as (path, label, bbox) tuples.
CELEBA_MAX           = None
UTKFACE_MAX          = None
FAIRFACE_MAX         = None
ANNO_FILE       = Path('checkpoints/face_detector_annotations.json')
CKPT_PATH       = Path('checkpoints/face_detector.pth')
CKPT_DIR        = Path('checkpoints')

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS_PHASE1  = 40
EPOCHS_PHASE2  = 30     # reduced from 40 — val typically saturates by epoch 20
                        # on this dataset, so the extra 10 epochs buy little
BATCH_SIZE     = 128
LR             = 3e-3
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4       # lazy-loading samples are just (path, label, bbox)
                         # tuples, so fork-copy is cheap — 4 workers is fine.
NEG_PER_IMAGE  = 10     # random negative crops per full-scene image
IOU_THRESH     = 0.25   # crop must have IoU < this with ALL face boxes to be a negative

# Hard negative mining config
MINE_SCALES    = [0.35, 0.22, 0.14, 0.09]
MINE_STRIDE    = 0.4
MINE_THRESH    = 0.5    # detector score threshold during mining (lower = more candidates)
MAX_HARD_NEGS  = 6000   # cap on total hard negatives collected

# Scale augmentation: simulates faces at various distances within the sliding window.
# e.g. scale=0.25 → face occupies 16×16 of the 64×64 patch ≈ face 4× further away.
SCALE_AUG_RATIOS  = [0.20, 0.33, 0.50, 0.70]  # face occupies this fraction of patch
SCALE_AUG_PER_IMG = 2                           # synthetic distance samples per positive


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def iou(box_a, box_b):
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    aA = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    aB = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (aA + aB - inter)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1a: Auto-annotate full-scene images with Haar cascade
# ─────────────────────────────────────────────────────────────────────────────

def auto_annotate(force: bool = False) -> dict:
    """
    Run OpenCV's Haar cascade on every image in photos_all/ to get bounding boxes.
    These boxes are used to:
      (a) generate positive face crops from full-scene images
      (b) ensure random negative crops don't accidentally contain a face
    Result is cached to JSON so this only runs once.
    """
    if ANNO_FILE.exists() and not force:
        print(f'Loading cached annotations from {ANNO_FILE}')
        with open(ANNO_FILE) as f:
            return json.load(f)

    print('Auto-annotating with Haar cascade (runs once, result cached) ...')
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    annotations = {}
    imgs = sorted(PHOTOS_ALL_DIR.glob('*.JPG'))
    detected = missed = 0

    for p in tqdm(imgs, desc='Annotating'):
        img  = cv2.imread(str(p))
        h, w = img.shape[:2]
        scale = 0.5
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(15, 15))
        boxes = []
        if len(faces):
            for (x, y, fw, fh) in faces:
                x1 = max(int(x/scale) - int(fw*0.2/scale), 0)
                y1 = max(int(y/scale) - int(fh*0.2/scale), 0)
                x2 = min(int((x+fw)/scale) + int(fw*0.2/scale), w-1)
                y2 = min(int((y+fh)/scale) + int(fh*0.2/scale), h-1)
                boxes.append([x1, y1, x2, y2])
            detected += 1
        else:
            missed += 1
        annotations[p.name] = boxes

    CKPT_DIR.mkdir(exist_ok=True)
    with open(ANNO_FILE, 'w') as f:
        json.dump(annotations, f)
    print(f'Annotated: {detected} with faces, {missed} without / {len(imgs)} total')
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# WIDER FACE annotation parser
# ─────────────────────────────────────────────────────────────────────────────

def load_wider_face_crops(max_crops: int = 30_000) -> list:
    """
    Parse WIDER FACE train annotations and return lazy (path, bbox) tuples.

    Annotation format (wider_face_train_bbx_gt.txt):
        <relative_image_path>
        <num_faces>
        x1 y1 w h blur expr illum invalid occl pose
        ...

    Only loads non-invalid, non-heavily-occluded faces to keep quality high.
    Image dimensions are read once per source image (to clamp padded bboxes);
    the pixels themselves are read lazily at training time.
    """
    anno_file = WIDER_FACE_DIR / 'wider_face_train_bbx_gt.txt'
    img_root  = WIDER_FACE_DIR / 'images'

    if not anno_file.exists() or not img_root.exists():
        return []

    out = []
    with open(anno_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines) and len(out) < max_crops:
        rel_path  = lines[i]; i += 1
        if i >= len(lines):
            break
        num_faces = int(lines[i]); i += 1

        img_path = img_root / rel_path
        if not img_path.exists():
            # Still need to advance past the face lines for this image
            i += num_faces
            continue

        # Read dimensions only (fast; PIL loads header without pixels)
        try:
            iw, ih = Image.open(img_path).size
        except Exception:
            i += num_faces
            continue

        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = list(map(int, lines[i].split())); i += 1
            x, y, w, h = parts[:4]
            invalid    = parts[7] if len(parts) > 7 else 0
            occlusion  = parts[8] if len(parts) > 8 else 0

            if invalid == 1 or occlusion == 2:
                continue
            if w < 20 or h < 20:   # skip tiny annotations (noise)
                continue

            # Add 20 % margin so crop includes forehead and chin
            pad_x, pad_y = int(w * 0.20), int(h * 0.20)
            x1 = max(x - pad_x, 0);       y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, iw);  y2 = min(y + h + pad_y, ih)
            if x2 <= x1 or y2 <= y1:
                continue

            out.append((str(img_path), (x1, y1, x2, y2)))

            if len(out) >= max_crops:
                break

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scale augmentation helper
# ─────────────────────────────────────────────────────────────────────────────

def make_scaled_samples(face_imgs: list, bg_patches: list) -> list:
    """
    Synthesise training examples where the face appears SMALL inside the patch.

    This is the critical augmentation for distance robustness:
    without it the model only ever sees faces that fill the entire 64×64 patch
    (close-up), and fails to detect faces that are far away (small).

    For each scale in SCALE_AUG_RATIOS, the face is resized to
    scale × 64 pixels and composited onto a random background patch.
    """
    out = []
    ps  = DETECTOR_PATCH_SIZE

    for face_img in face_imgs:
        chosen_scales = random.sample(SCALE_AUG_RATIOS,
                                      min(SCALE_AUG_PER_IMG, len(SCALE_AUG_RATIOS)))
        for scale in chosen_scales:
            face_sz = max(int(ps * scale), 8)

            # Background: real scene crop if available, else grey noise
            if bg_patches:
                bg = random.choice(bg_patches).copy()
            else:
                arr = np.random.randint(40, 200, (ps, ps, 3), dtype=np.uint8)
                bg  = Image.fromarray(arr)

            face_sm = face_img.resize((face_sz, face_sz), Image.BILINEAR)

            max_off = ps - face_sz
            if max_off <= 0:
                continue
            px = random.randint(0, max_off)
            py = random.randint(0, max_off)
            bg.paste(face_sm, (px, py))
            out.append((bg, 1))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b: Build the (image, label) sample list
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(annotations: dict, hard_neg_paths: list = None):
    """
    Assemble all (path, label, bbox) tuples for training. Label 1 = face,
    label 0 = background. `bbox` is either None (use the whole image) or
    (x1, y1, x2, y2) pixel coords to crop inside the source image.

    Images are never decoded here — FacePatchDataset.__getitem__ reads and
    crops them on demand. This lets RAM scale with *sample count*, not
    *pixel data*, so caps (CELEBA_MAX, FAIRFACE_MAX, etc.) are optional.

    Positive sources:
      • DnHFaces `photos_all_faces/` — pre-cropped thumbnails (whole file).
      • DnHFaces `photos_all/` + Haar bboxes — crop inside full scene image.
      • LFW, CelebA, UTKFace, FairFace — each file is already a tight crop.
      • WIDER FACE full set (if present) — bbox inside full source image.
      • `wider_face_crops/` — standalone flat crop files.
    Negative sources:
      • Random non-overlapping crops from DnHFaces scenes.
      • Hard negatives mined in Phase 2, persisted to
        `checkpoints/hard_neg_cache/` and supplied as a path list.
    """
    random.seed(42)
    samples = []

    def _take(paths, max_n):
        """Shuffle and optionally cap a list of paths. Skip cap if max_n is None."""
        lst = list(paths)
        random.shuffle(lst)
        return lst if max_n is None else lst[:max_n]

    # ── Positives: DnHFaces pre-cropped faces ────────────────────────────────
    dnh_n = 0
    for p in FACES_DIR.glob('*.jpg'):
        samples.append((str(p), 1, None))
        dnh_n += 1
    if dnh_n:
        print(f'  + {dnh_n:,} DnHFaces pre-cropped positives')

    # ── Scene positives + scene-derived random negatives ─────────────────────
    scene_pos = scene_neg = 0
    for fname, boxes in annotations.items():
        full_path = PHOTOS_ALL_DIR / fname
        if not full_path.exists():
            continue
        # Image dims via header read (no pixel decode)
        try:
            w, h = Image.open(full_path).size
        except Exception:
            continue
        full_path_s = str(full_path)

        for (x1, y1, x2, y2) in boxes:
            if x2 > x1 and y2 > y1:
                samples.append((full_path_s, 1, (x1, y1, x2, y2)))
                scene_pos += 1

        if not boxes:
            continue
        min_sz = 60; max_sz = min(h, w) // 2
        neg_added = attempts = 0
        while neg_added < NEG_PER_IMAGE and attempts < 300:
            attempts += 1
            sz  = random.randint(min_sz, max_sz)
            rx1 = random.randint(0, max(0, w-sz-1))
            ry1 = random.randint(0, max(0, h-sz-1))
            rx2 = rx1 + sz; ry2 = ry1 + sz
            if not any(iou([rx1, ry1, rx2, ry2], b) > IOU_THRESH for b in boxes):
                samples.append((full_path_s, 0, (rx1, ry1, rx2, ry2)))
                neg_added += 1
                scene_neg += 1
    if scene_pos or scene_neg:
        print(f'  + {scene_pos:,} scene positives, {scene_neg:,} scene random negatives')

    # ── Positives: LFW ───────────────────────────────────────────────────────
    if LFW_DIR.exists():
        lfw_paths = list(LFW_DIR.glob('**/*.jpg'))
        for p in lfw_paths:
            samples.append((str(p), 1, None))
        print(f'  + {len(lfw_paths):,} LFW positives')
    else:
        print('  (LFW not found — run  python download_datasets.py)')

    # ── Positives: CelebA aligned faces ──────────────────────────────────────
    if CELEBA_DIR.exists():
        celeba_paths = _take(CELEBA_DIR.glob('*.jpg'), CELEBA_MAX)
        for p in celeba_paths:
            samples.append((str(p), 1, None))
        print(f'  + {len(celeba_paths):,} CelebA positives')
    else:
        print('  (CelebA not found — run: '
              'kaggle datasets download -d jessicali9530/celeba-dataset '
              '-p datasets/tmp_celeba --unzip)')

    # ── Positives: UTKFace (broad demographic spread) ────────────────────────
    if UTKFACE_DIR.exists():
        utk_paths = _take(UTKFACE_DIR.glob('*.jpg'), UTKFACE_MAX)
        for p in utk_paths:
            samples.append((str(p), 1, None))
        print(f'  + {len(utk_paths):,} UTKFace positives')
    else:
        print('  (UTKFace not found — run  python download_datasets.py --utk)')

    # ── Positives: FairFace (demographically balanced) ───────────────────────
    if FAIRFACE_DIR.exists():
        fair_paths = _take(FAIRFACE_DIR.rglob('*.jpg'), FAIRFACE_MAX)
        for p in fair_paths:
            samples.append((str(p), 1, None))
        print(f'  + {len(fair_paths):,} FairFace positives')
    else:
        print('  (FairFace not found — run  python download_datasets.py --fairface '
              'for manual instructions)')

    # ── Positives: WIDER FACE crops (diverse scales) ─────────────────────────
    wider_tuples = load_wider_face_crops(max_crops=30_000)   # (path, bbox)
    wider_n = 0
    for path, bbox in wider_tuples:
        samples.append((path, 1, bbox))
        wider_n += 1
    if WIDER_FACE_CROPS_DIR.exists():
        for p in WIDER_FACE_CROPS_DIR.glob('*.jpg'):
            samples.append((str(p), 1, None))
            wider_n += 1
    if wider_n:
        print(f'  + {wider_n:,} WIDER FACE positives')
    else:
        print('  (WIDER FACE not found — run  python download_datasets.py --wider)')

    # ── Hard negatives (passed in as paths, already persisted to disk) ───────
    if hard_neg_paths:
        for p in hard_neg_paths:
            samples.append((str(p), 0, None))

    pos = sum(1 for _, l, _ in samples if l == 1)
    neg = sum(1 for _, l, _ in samples if l == 0)
    print(f'Dataset: {pos:,} positives, {neg:,} negatives  ({len(samples):,} total)')
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Hard negative mining
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mine_hard_negatives(model, annotations: dict, device) -> list:
    """
    Slide the current model over full-scene images. Collect patches where the
    model fires but there is no real face — face-like false positives.

    Each mined patch is persisted to HARD_NEG_CACHE_DIR as a JPG so the
    lazy-loading Dataset can reference it by path. The previous cache (if
    any) is cleared first to avoid stale negatives from earlier runs.

    Returns a list of string paths.
    """
    print('Mining hard negatives ...')
    model.eval()

    # Wipe any previous cache — old hard negs might not match this detector
    if HARD_NEG_CACHE_DIR.exists():
        for old in HARD_NEG_CACHE_DIR.glob('*.jpg'):
            try: old.unlink()
            except OSError: pass
    HARD_NEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out_paths: list = []
    imgs = sorted(PHOTOS_ALL_DIR.glob('*.JPG'))
    random.shuffle(imgs)

    for p in tqdm(imgs, desc='Mining'):
        if len(out_paths) >= MAX_HARD_NEGS:
            break
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        MINE_W, MINE_H = 640, 480
        img_small  = cv2.resize(img_bgr, (MINE_W, MINE_H))
        img_rgb    = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        h, w       = img_small.shape[:2]

        orig_h, orig_w = img_bgr.shape[:2]
        sx, sy = MINE_W / orig_w, MINE_H / orig_h
        scaled_boxes = [[int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]
                        for b in annotations.get(p.name, [])]

        patches, metas = [], []
        for scale in MINE_SCALES:
            win    = int(min(h, w) * scale)
            if win < 32:
                continue
            stride = max(int(win * MINE_STRIDE), 8)
            for y in range(0, h-win+1, stride):
                for x in range(0, w-win+1, stride):
                    crop = img_rgb[y:y+win, x:x+win]
                    patches.append(DETECTOR_INFER_TF(Image.fromarray(crop)))
                    metas.append((x, y, x+win, y+win))

        if not patches:
            continue

        scores = []
        for i in range(0, len(patches), 256):
            batch = torch.stack(patches[i:i+256]).to(device)
            scores.extend(torch.sigmoid(model(batch)).cpu().tolist())

        for (x1, y1, x2, y2), score in zip(metas, scores):
            if score >= MINE_THRESH:
                if not any(iou([x1, y1, x2, y2], b) > IOU_THRESH for b in scaled_boxes):
                    crop_np = img_rgb[y1:y2, x1:x2]
                    if crop_np.size == 0:
                        continue
                    out_file = HARD_NEG_CACHE_DIR / f'hn_{len(out_paths):06d}.jpg'
                    # Save as BGR via cv2 to skip a PIL detour
                    cv2.imwrite(str(out_file),
                                cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR))
                    out_paths.append(str(out_file))
                    if len(out_paths) >= MAX_HARD_NEGS:
                        break

    print(f'Mined {len(out_paths):,} hard negatives (persisted to {HARD_NEG_CACHE_DIR})')
    return out_paths


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FacePatchDataset(Dataset):
    """
    Lazy-loading dataset. Each sample is a (path, label, bbox) tuple stored
    as-is in the `samples` list. Images are decoded per __getitem__ call, so
    RAM usage scales with sample *count* (megabytes) rather than decoded
    pixels (gigabytes). DataLoader workers fork-copy only metadata.
    """

    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, bbox = self.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Return a grey fallback so training can't crash on a single bad file
            zero = torch.zeros(3, DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE,
                               dtype=torch.float32)
            return zero, torch.tensor(label, dtype=torch.float32)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = img_bgr.shape[:2]
            x1 = max(x1, 0); y1 = max(y1, 0)
            x2 = min(x2, w); y2 = min(y2, h)
            if x2 <= x1 or y2 <= y1:
                zero = torch.zeros(3, DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE,
                                   dtype=torch.float32)
                return zero, torch.tensor(label, dtype=torch.float32)
            img_bgr = img_bgr[y1:y2, x1:x2]

        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return self.transform(pil), torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model, tr_samples, va_samples, device, epochs, lr, label):
    """Run one full training phase and save the best checkpoint."""
    tr_ds = FacePatchDataset(tr_samples, DETECTOR_TRAIN_TF)
    va_ds = FacePatchDataset(va_samples, DETECTOR_INFER_TF)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           persistent_workers=NUM_WORKERS > 0)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           persistent_workers=NUM_WORKERS > 0)

    # pos_weight=2 compensates for class imbalance (more negatives than positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler('cuda')
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                           total_steps=epochs * len(tr_loader),
                           pct_start=0.1, anneal_strategy='cos')

    best_val = float('inf')
    print(f'\n── {label} ──')
    print(f"{'Epoch':>6} | {'TrLoss':>8} {'TrAcc':>7} | {'VaLoss':>8} {'VaAcc':>7} {'P':>5} {'R':>5} | {'Time':>5}")
    print('─' * 65)

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        tr_loss = tr_correct = tr_n = 0
        for imgs, labels in tr_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            preds       = (logits.detach() > 0).float()
            tr_correct += (preds == labels).sum().item()
            tr_loss    += loss.item() * imgs.size(0)
            tr_n       += imgs.size(0)

        model.eval()
        va_loss = va_correct = va_n = va_tp = va_fp = va_fn = 0
        with torch.no_grad():
            for imgs, labels in va_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                preds  = (logits > 0).float()
                va_correct += (preds == labels).sum().item()
                va_loss    += loss.item() * imgs.size(0)
                va_n       += imgs.size(0)
                va_tp += ((preds==1) & (labels==1)).sum().item()
                va_fp += ((preds==1) & (labels==0)).sum().item()
                va_fn += ((preds==0) & (labels==1)).sum().item()

        P = va_tp / (va_tp + va_fp + 1e-8)
        R = va_tp / (va_tp + va_fn + 1e-8)
        print(f"{epoch+1:>6} | {tr_loss/tr_n:>8.4f} {tr_correct/tr_n:>6.1%} "
              f"| {va_loss/va_n:>8.4f} {va_correct/va_n:>6.1%} {P:>4.2f} {R:>4.2f}"
              f" | {time.time()-t0:>4.1f}s")

        if va_loss / va_n < best_val:
            best_val = va_loss / va_n
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_loss': best_val}, CKPT_PATH)
            print(f"         ↑ saved (val_loss={best_val:.4f}, P={P:.2f}, R={R:.2f})")

    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(resume_from_phase1: bool = False):
    CKPT_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    annotations = auto_annotate()

    model = FaceDetectorCNN().to(device)

    if resume_from_phase1:
        # Skip Phase 1 training — load the existing best checkpoint and go
        # straight to mining. Useful after a Phase 2 OOM or crash.
        print(f'Resuming from existing Phase 1 checkpoint: {CKPT_PATH}')
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f'  Loaded (epoch {ckpt.get("epoch", "?")}, val_loss {ckpt.get("val_loss", "?"):.4f})')
    else:
        # Phase 1: bootstrap
        print('\nBuilding Phase-1 dataset ...')
        samples = build_samples(annotations, hard_neg_paths=None)
        labels  = [l for _, l, _ in samples]
        tr_idx, va_idx = train_test_split(range(len(samples)), test_size=0.15,
                                          random_state=42, stratify=labels)
        run_training(model,
                     [samples[i] for i in tr_idx],
                     [samples[i] for i in va_idx],
                     device, EPOCHS_PHASE1, LR, 'Phase 1 — Bootstrap')

        # Free Phase-1 sample objects before Phase-2 starts loading —
        # otherwise both datasets (plus fork-copies in DataLoader workers)
        # sit in RAM simultaneously and trigger OOM on 32 GB machines.
        del samples, labels, tr_idx, va_idx
        gc.collect()

        # Reload best Phase-1 weights (run_training saves best, not last)
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])

    # Phase 2: hard negative mining + retrain (from-scratch weights)
    hard_neg_paths = mine_hard_negatives(model, annotations, device)

    # Free the Phase-1 model before loading Phase-2 data
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('\nBuilding Phase-2 dataset ...')
    samples2 = build_samples(annotations, hard_neg_paths=hard_neg_paths)
    labels2  = [l for _, l, _ in samples2]
    tr_idx2, va_idx2 = train_test_split(range(len(samples2)), test_size=0.15,
                                        random_state=42, stratify=labels2)

    model2 = FaceDetectorCNN().to(device)   # fresh weights — train from scratch on augmented data
    run_training(model2,
                 [samples2[i] for i in tr_idx2],
                 [samples2[i] for i in va_idx2],
                 device, EPOCHS_PHASE2, LR, 'Phase 2 — Hard Negative Retrain')

    print(f'\nTraining complete.  Checkpoint: {CKPT_PATH}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resume-from-phase1', action='store_true',
                   help='Skip Phase-1 training; use existing checkpoint as mining seed.')
    args = p.parse_args()
    train(resume_from_phase1=args.resume_from_phase1)
