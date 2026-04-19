"""
train_landmarks.py  —  Train the eye-center landmark regressor
===============================================================

TRAINING DATA
─────────────
 1. WIDER-selected  (datasets/tmp_wider — downloaded earlier)
    4,275 images, each with 68 facial landmarks per face.
    We take the mean of landmarks 36-41 as the right-eye centre (viewer's perspective),
    and the mean of 42-47 as the left-eye centre — the iBUG 68-point convention.

 2. CelebA  (datasets/tmp_celeba)
    ~200,000 aligned face images with 5 landmarks each:
        lefteye, righteye, nose, leftmouth, rightmouth
    Perfect — we already only need the two eye landmarks.

All annotations are converted into a unified format:
    (PIL_face_crop, left_eye_xy_normalised, right_eye_xy_normalised)

The labels are normalised to [0, 1] relative to the crop size so the model
output can be used directly regardless of crop resolution.

USAGE
─────
    python train_landmarks.py
"""

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
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from facetrack.landmarks import (
    LandmarkCNN, LANDMARK_TRAIN_TF, LANDMARK_INFER_TF,
    flip_landmarks_horizontal,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
WIDER_DIR        = Path('datasets/tmp_wider')
WIDER_ANNO       = WIDER_DIR / 'annotations.txt'
WIDER_IMG_ROOT   = WIDER_DIR / 'train'

CELEBA_DIR       = Path('datasets/tmp_celeba')
# The Kaggle "jessicali9530/celeba-dataset" layout:
#   img_align_celeba/img_align_celeba/*.jpg
#   list_landmarks_align_celeba.csv   (or .txt in some dumps)
CELEBA_IMG_ROOT  = CELEBA_DIR / 'img_align_celeba' / 'img_align_celeba'

CKPT_PATH        = Path('checkpoints/landmark_net.pth')
CKPT_DIR         = Path('checkpoints')

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS       = 30
BATCH_SIZE   = 256
LR           = 2e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
VAL_FRAC     = 0.05           # CelebA is huge, 5 % is plenty (~10k images)
MAX_CELEBA   = 100_000        # cap to keep epoch time reasonable

# iBUG 68-point indices used to derive the 5 CelebA-style keypoints.
# (0-indexed)
RIGHT_EYE_IDX    = list(range(36, 42))   # 6 points → avg = right eye centre
LEFT_EYE_IDX     = list(range(42, 48))   # 6 points → avg = left  eye centre
NOSE_IDX         = 30                    # nose tip
RIGHT_MOUTH_IDX  = 48                    # right mouth corner
LEFT_MOUTH_IDX   = 54                    # left  mouth corner


# ─────────────────────────────────────────────────────────────────────────────
# WIDER-selected loader (landmarks encoded inside annotations.txt)
# ─────────────────────────────────────────────────────────────────────────────

def load_wider_samples() -> list:
    """
    Parse WIDER-selected annotations and return list of
    (source_type, image_path, bbox, normalised_eye_xy_xy_tuple).

    The crop is computed at dataset-iteration time — we don't hold any pixels
    in memory here, just metadata.
    """
    if not WIDER_ANNO.exists():
        print(f'  (WIDER-selected not found at {WIDER_ANNO} — skipping)')
        return []

    out = []
    with open(WIDER_ANNO) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        rel_path = lines[i]; i += 1
        if i >= len(lines):
            break
        n_faces = int(lines[i]); i += 1

        img_path = WIDER_IMG_ROOT / rel_path

        for _ in range(n_faces):
            if i >= len(lines):
                break
            parts = list(map(int, lines[i].split())); i += 1
            if len(parts) < 4 + 2 * 68 or not img_path.exists():
                continue

            x, y, w, h = parts[:4]
            landmarks  = parts[4:4 + 2 * 68]

            pad_x, pad_y = int(w * 0.20), int(h * 0.20)
            x1 = max(x - pad_x, 0);   y1 = max(y - pad_y, 0)
            x2 = x + w + pad_x;        y2 = y + h + pad_y

            # Derive the 5 CelebA-style landmarks from the 68 iBUG points
            def avg_abs(indices):
                xs = [landmarks[2*k]   for k in indices]
                ys = [landmarks[2*k+1] for k in indices]
                return sum(xs) / len(xs), sum(ys) / len(ys)

            rex, rey = avg_abs(RIGHT_EYE_IDX)
            lex, ley = avg_abs(LEFT_EYE_IDX)
            nox, noy = landmarks[2*NOSE_IDX],        landmarks[2*NOSE_IDX+1]
            rmx, rmy = landmarks[2*RIGHT_MOUTH_IDX], landmarks[2*RIGHT_MOUTH_IDX+1]
            lmx, lmy = landmarks[2*LEFT_MOUTH_IDX],  landmarks[2*LEFT_MOUTH_IDX+1]

            out.append(('wider', str(img_path),
                        (x1, y1, x2, y2),
                        (lex, ley, rex, rey, nox, noy, lmx, lmy, rmx, rmy)))

    print(f'  + {len(out):,} WIDER-selected samples')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CelebA loader
# ─────────────────────────────────────────────────────────────────────────────

def _find_celeba_landmark_file() -> Path:
    for name in ('list_landmarks_align_celeba.csv',
                 'list_landmarks_align_celeba.txt'):
        for parent in (CELEBA_DIR, CELEBA_DIR / 'list_landmarks_align_celeba'):
            p = parent / name
            if p.exists():
                return p
    # Recursive fallback
    hits = list(CELEBA_DIR.rglob('list_landmarks_align_celeba*'))
    return hits[0] if hits else None


def load_celeba_samples(max_samples: int = MAX_CELEBA) -> list:
    """
    Build metadata list for CelebA.  Returns list of
        ('celeba', image_path, None (aligned, no bbox), (lex, ley, rex, rey)_absolute_pixels)
    Actual images are loaded lazily in __getitem__.
    """
    lm_file = _find_celeba_landmark_file()
    if not lm_file or not lm_file.exists():
        print('  (CelebA landmarks file not found — skipping)')
        return []

    for candidate in (CELEBA_IMG_ROOT, CELEBA_DIR / 'img_align_celeba'):
        if candidate.exists() and any(candidate.glob('*.jpg')):
            img_root = candidate
            break
    else:
        print(f'  (CelebA images not found at {CELEBA_IMG_ROOT} — skipping)')
        return []

    out = []
    with open(lm_file) as f:
        lines = [l.rstrip() for l in f]

    is_csv = ',' in lines[0] if lines else False
    data_start = 0
    for idx, line in enumerate(lines):
        tok = line.split(',')[0] if is_csv else line.split()[0]
        if tok.endswith('.jpg') or tok.endswith('.png'):
            data_start = idx
            break

    data_lines = lines[data_start:]
    random.shuffle(data_lines)

    for line in data_lines:
        if len(out) >= max_samples:
            break
        parts = [p.strip() for p in (line.split(',') if is_csv else line.split()) if p.strip()]
        if len(parts) < 11:
            continue
        fname = parts[0]
        try:
            nums = list(map(float, parts[1:11]))
        except ValueError:
            continue

        img_path = img_root / fname
        if not img_path.exists():
            continue

        # CelebA CSV columns (1..10):
        #   lefteye_x, lefteye_y, righteye_x, righteye_y,
        #   nose_x,    nose_y,
        #   leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y
        # All 5 landmarks in aligned-image pixel coords.
        out.append(('celeba', str(img_path), None, tuple(nums)))

    print(f'  + {len(out):,} CelebA samples')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset + training
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    """
    Lazy-loading dataset.  Each entry is metadata only
    (source, image_path, bbox_or_None, absolute_eye_xy_xy).
    Images are read from disk per __getitem__ call — prevents memory bloat.

    Horizontal-flip augmentation (train only) is applied here so the 5-point
    label vector can be mirrored + swapped in sync with the image.
    """

    def __init__(self, samples, transform, flip_prob: float = 0.0):
        self.samples   = samples
        self.transform = transform
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, img_path, bbox, landmarks = self.samples[idx]
        # landmarks = (lex, ley, rex, rey, nox, noy, lmx, lmy, rmx, rmy)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            zero = torch.zeros(3, 64, 64, dtype=torch.float32)
            return zero, torch.full((10,), 0.5, dtype=torch.float32)

        if source == 'wider':
            ih, iw = img_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(x1, 0); y1 = max(y1, 0)
            x2 = min(x2, iw); y2 = min(y2, ih)
            if x2 <= x1 or y2 <= y1:
                zero = torch.zeros(3, 64, 64, dtype=torch.float32)
                return zero, torch.full((10,), 0.5, dtype=torch.float32)
            crop = img_bgr[y1:y2, x1:x2]
            cw, ch = x2 - x1, y2 - y1
            normed = []
            for k in range(5):
                lx = (landmarks[2*k]   - x1) / cw
                ly = (landmarks[2*k+1] - y1) / ch
                normed.extend([lx, ly])
        else:   # celeba — aligned image, use whole frame
            crop = img_bgr
            H, W = crop.shape[:2]
            normed = []
            for k in range(5):
                normed.extend([landmarks[2*k] / W, landmarks[2*k+1] / H])

        # Clamp occasional landmarks slightly outside the crop
        normed = [min(max(v, 0.0), 1.0) for v in normed]

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Horizontal flip augmentation (train only). Has to happen here so the
        # label coordinates stay consistent with the pixels.
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            normed = flip_landmarks_horizontal(normed)

        return self.transform(pil), torch.tensor(normed, dtype=torch.float32)


def train():
    CKPT_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('Loading datasets ...')
    samples = []
    samples.extend(load_wider_samples())
    samples.extend(load_celeba_samples())

    if not samples:
        print('\nNo training samples found.  Download:')
        print('  kaggle datasets download -d alirezakay/wider-selected -p datasets/tmp_wider --unzip')
        print('  kaggle datasets download -d jessicali9530/celeba-dataset -p datasets/tmp_celeba --unzip')
        return

    print(f'\nTotal: {len(samples):,} training samples')

    random.seed(42)
    random.shuffle(samples)
    n_val = max(500, int(len(samples) * VAL_FRAC))
    tr_raw = samples[n_val:]
    va_raw = samples[:n_val]

    tr_ds = LandmarkDataset(tr_raw, LANDMARK_TRAIN_TF, flip_prob=0.5)
    va_ds = LandmarkDataset(va_raw, LANDMARK_INFER_TF, flip_prob=0.0)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           persistent_workers=NUM_WORKERS > 0)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True,
                           persistent_workers=NUM_WORKERS > 0)

    print(f'Train: {len(tr_ds):,}  |  Val: {len(va_ds):,}\n')

    model     = LandmarkCNN().to(device)
    criterion = nn.SmoothL1Loss()         # robust to landmark-annotation noise
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler('cuda')
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                           total_steps=EPOCHS * len(tr_loader),
                           pct_start=0.1, anneal_strategy='cos')

    best_err = float('inf')
    print(f"{'Epoch':>5} | {'TrLoss':>8} {'TrErr':>6} | {'VaLoss':>8} {'VaErr':>6} | {'Time':>5}")
    print('─' * 58)

    for epoch in range(EPOCHS):
        t0 = time.time()

        model.train()
        tr_loss = tr_err = tr_n = 0
        for imgs, targets in tr_loader:
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                preds = model(imgs)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_err  += (preds.detach() - targets).abs().mean(dim=1).sum().item()
            tr_n    += imgs.size(0)

        model.eval()
        va_loss = va_err = va_n = 0
        with torch.no_grad():
            for imgs, targets in va_loader:
                imgs    = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds   = model(imgs)
                loss    = criterion(preds, targets)
                va_loss += loss.item() * imgs.size(0)
                va_err  += (preds - targets).abs().mean(dim=1).sum().item()
                va_n    += imgs.size(0)

        va_err_norm = va_err / va_n
        print(f"{epoch+1:>5} | {tr_loss/tr_n:>8.5f} {tr_err/tr_n:>6.4f} "
              f"| {va_loss/va_n:>8.5f} {va_err_norm:>6.4f} | {time.time()-t0:>4.1f}s")

        if va_err_norm < best_err:
            best_err = va_err_norm
            torch.save({'model': model.state_dict(),
                        'val_err': best_err,
                        'epoch': epoch}, CKPT_PATH)
            print(f"      ↑ saved  (normalised landmark error = {best_err:.4f})")

    print(f'\nTraining complete.  Best val error = {best_err:.4f} (normalised)')
    print(f'  → ≈ {best_err * 64:.2f} px error on a 64×64 crop')
    print(f'Checkpoint: {CKPT_PATH}')


if __name__ == '__main__':
    train()
