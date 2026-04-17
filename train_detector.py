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
CELEBA_MAX           = 30_000                             # cap CelebA positives
ANNO_FILE       = Path('checkpoints/face_detector_annotations.json')
CKPT_PATH       = Path('checkpoints/face_detector.pth')
CKPT_DIR        = Path('checkpoints')

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS_PHASE1  = 40
EPOCHS_PHASE2  = 40
BATCH_SIZE     = 128
LR             = 3e-3
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
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
    Parse WIDER FACE train annotations and return PIL face crops.

    Annotation format (wider_face_train_bbx_gt.txt):
        <relative_image_path>
        <num_faces>
        x1 y1 w h blur expr illum invalid occl pose
        ...

    Only loads non-invalid, non-heavily-occluded faces to keep quality high.
    Caps at max_crops to stay memory-friendly.
    """
    anno_file = WIDER_FACE_DIR / 'wider_face_train_bbx_gt.txt'
    img_root  = WIDER_FACE_DIR / 'images'

    if not anno_file.exists() or not img_root.exists():
        return []

    crops  = []
    with open(anno_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines) and len(crops) < max_crops:
        rel_path  = lines[i]; i += 1
        if i >= len(lines):
            break
        num_faces = int(lines[i]); i += 1

        img_path = img_root / rel_path
        img_bgr  = cv2.imread(str(img_path)) if img_path.exists() else None

        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = list(map(int, lines[i].split())); i += 1
            x, y, w, h = parts[:4]
            invalid    = parts[7] if len(parts) > 7 else 0
            occlusion  = parts[8] if len(parts) > 8 else 0

            if img_bgr is None or invalid == 1 or occlusion == 2:
                continue
            if w < 20 or h < 20:   # skip tiny annotations (noise)
                continue

            ih, iw = img_bgr.shape[:2]
            # Add 20 % margin so crop includes forehead and chin
            pad_x, pad_y = int(w * 0.20), int(h * 0.20)
            x1 = max(x - pad_x, 0);       y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, iw);  y2 = min(y + h + pad_y, ih)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_bgr = img_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))

            if len(crops) >= max_crops:
                break

    return crops


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

def build_samples(annotations: dict, hard_negs: list = None):
    """
    Assemble all (PIL_image, label) pairs for training.
    Label 1 = face,  Label 0 = background.

    Positives come from:
      • photos_all_faces/ — pre-cropped DnHFaces images
      • Haar-annotated crops from full scenes
      • LFW/ — 13k diverse faces (run download_datasets.py)
      • WIDER FACE — 390k faces at varying scales/distances (run download_datasets.py --wider)
      • Scale-augmented synthetics — each face resized to 20/33/50/70 % of the patch
        to simulate how it looks at 1×, 2×, 3×, 5× the detection distance.

    Negatives come from:
      • Random crops from full scenes not overlapping any face
      • Hard negatives from Phase-2 mining (passed in as hard_negs)
    """
    random.seed(42)
    samples  = []
    bg_patches = []   # collected below, used for scale augmentation compositing

    # ── Positives: DnHFaces pre-cropped faces ─────────────────────────────
    dnhfaces = []
    for p in FACES_DIR.glob('*.jpg'):
        img = Image.open(p).convert('RGB')
        samples.append((img, 1))
        dnhfaces.append(img)

    # ── Positives + easy negatives from full-scene images ──────────────────
    scene_faces = []
    for fname, boxes in annotations.items():
        full_path = PHOTOS_ALL_DIR / fname
        if not full_path.exists():
            continue
        img_bgr = cv2.imread(str(full_path))
        if img_bgr is None:
            continue
        h, w    = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for (x1, y1, x2, y2) in boxes:
            if x2 > x1 and y2 > y1:
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    pil = Image.fromarray(crop)
                    samples.append((pil, 1))
                    scene_faces.append(pil)

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
                crop = img_rgb[ry1:ry2, rx1:rx2]
                if crop.size > 0:
                    pil = Image.fromarray(crop).resize(
                            (DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE))
                    samples.append((pil, 0))
                    bg_patches.append(pil)   # reuse as background for scale aug
                    neg_added += 1

    # ── Positives: LFW diverse faces ──────────────────────────────────────
    lfw_faces = []
    if LFW_DIR.exists():
        lfw_imgs = list(LFW_DIR.glob('**/*.jpg'))
        for p in lfw_imgs:
            try:
                img = Image.open(p).convert('RGB')
                samples.append((img, 1))
                lfw_faces.append(img)
            except Exception:
                pass
        print(f'  + {len(lfw_faces):,} LFW positives')
    else:
        print('  (LFW not found — run  python download_datasets.py)')

    # ── Positives: CelebA aligned faces (200k, capped) ────────────────────
    celeba_faces = []
    if CELEBA_DIR.exists():
        celeba_imgs = list(CELEBA_DIR.glob('*.jpg'))
        random.shuffle(celeba_imgs)
        for p in celeba_imgs[:CELEBA_MAX]:
            try:
                celeba_faces.append(Image.open(p).convert('RGB'))
            except Exception:
                pass
        for img in celeba_faces:
            samples.append((img, 1))
        print(f'  + {len(celeba_faces):,} CelebA positives')
    else:
        print('  (CelebA not found — run: '
              'kaggle datasets download -d jessicali9530/celeba-dataset '
              '-p datasets/tmp_celeba --unzip)')

    # ── Positives: WIDER FACE crops (diverse scales / real-world distances) ──
    wider_faces = load_wider_face_crops(max_crops=30_000)
    # Also load flat pre-extracted crops if present
    if WIDER_FACE_CROPS_DIR.exists():
        for p in WIDER_FACE_CROPS_DIR.glob('*.jpg'):
            try:
                wider_faces.append(Image.open(p).convert('RGB'))
            except Exception:
                pass
    if wider_faces:
        for img in wider_faces:
            samples.append((img, 1))
        print(f'  + {len(wider_faces):,} WIDER FACE positives')
    else:
        print('  (WIDER FACE not found — run  python download_datasets.py --wider)')

    # ── Scale augmentation — critical for distance robustness ─────────────
    # Teach the model to recognise faces that appear SMALL in the window
    # (i.e. the person is far from the camera).
    all_face_imgs = dnhfaces + scene_faces + lfw_faces[:3000] + wider_faces[:5000]
    # Scale augmentation disabled — it was teaching the detector to fire on
    # plain backgrounds (walls, empty space) because the composited mini-faces
    # were often indistinguishable from background texture after downsampling.
    # _ = make_scaled_samples(all_face_imgs, bg_patches)
    # samples.extend(_)
    # print(f'  + {len(_):,} scale-augmented positives (disabled)')

    # ── Hard negatives from mining ─────────────────────────────────────────
    if hard_negs:
        for img in hard_negs:
            samples.append((img, 0))

    pos = sum(1 for _, l in samples if l == 1)
    neg = sum(1 for _, l in samples if l == 0)
    print(f'Dataset: {pos:,} positives, {neg:,} negatives  ({len(samples):,} total)')
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Hard negative mining
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mine_hard_negatives(model, annotations: dict, device) -> list:
    """
    Slide the current model over full-scene images.
    Collect patches where the model fires but there is no real face —
    these are false positives that look face-like and make hard training examples.
    """
    print('Mining hard negatives ...')
    model.eval()
    hard_negs = []
    imgs = sorted(PHOTOS_ALL_DIR.glob('*.JPG'))
    random.shuffle(imgs)

    for p in tqdm(imgs, desc='Mining'):
        if len(hard_negs) >= MAX_HARD_NEGS:
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
                    if crop_np.size > 0:
                        hard_negs.append(Image.fromarray(crop_np))
                        if len(hard_negs) >= MAX_HARD_NEGS:
                            break

    print(f'Mined {len(hard_negs):,} hard negatives')
    return hard_negs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FacePatchDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


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

def train():
    CKPT_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    annotations = auto_annotate()

    # Phase 1: bootstrap
    print('\nBuilding Phase-1 dataset ...')
    samples = build_samples(annotations, hard_negs=None)
    labels  = [l for _, l in samples]
    tr_idx, va_idx = train_test_split(range(len(samples)), test_size=0.15,
                                      random_state=42, stratify=labels)
    model = FaceDetectorCNN().to(device)
    run_training(model,
                 [samples[i] for i in tr_idx],
                 [samples[i] for i in va_idx],
                 device, EPOCHS_PHASE1, LR, 'Phase 1 — Bootstrap')

    # Phase 2: hard negative mining + retrain
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    hard_negs = mine_hard_negatives(model, annotations, device)

    print('\nBuilding Phase-2 dataset ...')
    samples2 = build_samples(annotations, hard_negs=hard_negs)
    labels2  = [l for _, l in samples2]
    tr_idx2, va_idx2 = train_test_split(range(len(samples2)), test_size=0.15,
                                        random_state=42, stratify=labels2)

    model2 = FaceDetectorCNN().to(device)   # fresh weights — train from scratch on augmented data
    run_training(model2,
                 [samples2[i] for i in tr_idx2],
                 [samples2[i] for i in va_idx2],
                 device, EPOCHS_PHASE2, LR, 'Phase 2 — Hard Negative Retrain')

    print(f'\nTraining complete.  Checkpoint: {CKPT_PATH}')


if __name__ == '__main__':
    train()
