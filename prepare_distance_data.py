"""
prepare_distance_data.py  —  MPIIGaze → training features for DistanceNet
==========================================================================

MPIIGaze's per-frame annotation.txt already contains 12 detected eye
landmarks per image (dims 1-24). We use those to synthesise a face bounding
box (around the eye midpoint, expanded downward to include mouth) and run
ONLY our landmark net on the cropped region. The full sliding-window
detector is skipped entirely — ~50-100x faster than running it on every
frame, and the training-time features still come from the same landmark
network the deployment pipeline uses.

Ground truth distance: translation-z from the same annotation line (head
pose dims 30-35 = rotation vector + translation in mm).

Output: datasets/distance_features.npz with subject IDs for subject-level
holdout at training time.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from facetrack.landmarks    import (
    LANDMARK_INFER_TF, build_landmark_net, flip_landmarks_horizontal,
)
from facetrack.filters      import plausible_face_geometry
from facetrack.distance_net import extract_distance_features
from facetrack.config       import IPD_METRES

ROOT           = Path('datasets/tmp_mpiigaze/MPIIGaze/Data/Original')
OUT_FILE       = Path('datasets/distance_features.npz')
LMK_CKPT       = Path('checkpoints/landmark_net.pth')
FRAMES_PER_DAY = 60
MAX_SUBJECTS   = 15


def load_landmark(device):
    lck  = torch.load(LMK_CKPT, map_location=device, weights_only=False)
    arch = lck.get('arch', 'direct')
    lmk  = build_landmark_net(arch).to(device)
    lmk.load_state_dict(lck['model']); lmk.eval()
    print(f'Landmark : {LMK_CKPT}  arch={arch}', flush=True)
    return lmk


def load_camera_focal(subject_dir: Path) -> Optional[float]:
    cal = subject_dir / 'Calibration' / 'Camera.mat'
    if not cal.exists():
        return None
    K = sio.loadmat(cal)['cameraMatrix']
    return float(K[0, 0])


def parse_annotation_line(line: str):
    """Return (eye_landmarks [12,2], tz_mm) or None."""
    parts = line.split()
    if len(parts) < 35:
        return None
    try:
        # 1-24: 12 eye landmark points (x,y)
        eye_pts = np.array(parts[:24], dtype=np.float32).reshape(12, 2)
        # 30-35: head pose rotation (3) + translation mm (3). Index 34 = tz.
        tz = float(parts[34])
    except ValueError:
        return None
    return eye_pts, tz


def synth_face_box(eye_pts: np.ndarray, img_w: int, img_h: int):
    """Build a face bbox from the 12 eye landmarks, expanded to include mouth
    region. Returns (x1, y1, x2, y2) clamped to image bounds."""
    x_min, y_min = eye_pts.min(axis=0)
    x_max, y_max = eye_pts.max(axis=0)
    eye_w = max(x_max - x_min, 1.0)
    eye_h = max(y_max - y_min, 1.0)
    # Horizontal: 2.0× eye span (covers the whole face width).
    # Vertical:   up 0.8× eye_w for forehead, down 2.2× eye_w for mouth/chin.
    cx = (x_min + x_max) * 0.5
    # Use the eye midpoint Y as the vertical anchor, not the bbox center —
    # eyes sit near the vertical centre of the upper face half.
    cy_eye = (y_min + y_max) * 0.5
    face_w = eye_w * 2.0
    up     = eye_w * 0.8
    down   = eye_w * 2.2
    x1 = int(max(cx - face_w * 0.5, 0))
    x2 = int(min(cx + face_w * 0.5, img_w - 1))
    y1 = int(max(cy_eye - up, 0))
    y2 = int(min(cy_eye + down, img_h - 1))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return (x1, y1, x2, y2)


@torch.no_grad()
def run_landmarks(frame_bgr, box, landmark_net, device):
    """Return 5 absolute-coord landmarks on the original frame, or None."""
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    pil    = Image.fromarray(crop[:, :, ::-1])
    cw, ch = pil.size
    t_orig = LANDMARK_INFER_TF(pil).unsqueeze(0)
    t_flip = torch.flip(t_orig, dims=[3])
    batch  = torch.cat([t_orig, t_flip], dim=0).to(device)
    out2   = landmark_net(batch).cpu().numpy()
    lm_a   = out2[0].tolist()
    lm_b   = flip_landmarks_horizontal(out2[1].tolist())
    lm     = [(a + b) * 0.5 for a, b in zip(lm_a, lm_b)]
    lx, ly, rx, ry, nx_, ny_, lmx, lmy, rmx, rmy = lm

    if not plausible_face_geometry(lx, ly, rx, ry, nx_, ny_,
                                    lmx, lmy, rmx, rmy):
        return None
    return (
        x1 + lx  * cw, y1 + ly  * ch,
        x1 + rx  * cw, y1 + ry  * ch,
        x1 + nx_ * cw, y1 + ny_ * ch,
        x1 + lmx * cw, y1 + lmy * ch,
        x1 + rmx * cw, y1 + rmy * ch,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_per_day', type=int, default=FRAMES_PER_DAY)
    parser.add_argument('--out',            type=str, default=str(OUT_FILE))
    parser.add_argument('--max_subjects',   type=int, default=MAX_SUBJECTS)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    landmark_net = load_landmark(device)

    all_feats, all_labels, all_subj, all_geom = [], [], [], []
    random.seed(42)
    t_start = time.time()

    subjects = sorted(d.name for d in ROOT.iterdir()
                      if d.is_dir())[:args.max_subjects]
    print(f'Subjects: {subjects}\n', flush=True)

    for s_idx, subj in enumerate(subjects):
        subj_dir = ROOT / subj
        focal_px = load_camera_focal(subj_dir)
        if focal_px is None:
            print(f'  {subj}: no camera calibration, skipping', flush=True)
            continue

        days = sorted(d for d in subj_dir.iterdir()
                      if d.is_dir() and d.name.startswith('day'))
        subj_attempt = 0; subj_hits = 0; subj_t0 = time.time()

        for day in days:
            anno = day / 'annotation.txt'
            if not anno.exists():
                continue
            lines = anno.read_text().splitlines()
            if not lines:
                continue
            pairs = [(i + 1, line) for i, line in enumerate(lines)]
            random.shuffle(pairs)
            pairs = pairs[:args.frames_per_day]

            for idx, line in pairs:
                img_path = day / f'{idx:04d}.jpg'
                if not img_path.exists():
                    continue
                subj_attempt += 1

                parsed = parse_annotation_line(line)
                if parsed is None:
                    continue
                eye_pts, tz = parsed
                if tz < 200.0 or tz > 2500.0:
                    continue

                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    continue
                H_orig, W_orig = frame_bgr.shape[:2]

                box = synth_face_box(eye_pts, W_orig, H_orig)
                if box is None:
                    continue

                lmks_abs = run_landmarks(frame_bgr, box, landmark_net, device)
                if lmks_abs is None:
                    continue

                feats = extract_distance_features(lmks_abs, box, focal_px,
                                                   W_orig, H_orig)
                if feats is None:
                    continue

                ipd_px = float(np.hypot(lmks_abs[0] - lmks_abs[2],
                                         lmks_abs[1] - lmks_abs[3]))
                geom_dist_m = (IPD_METRES * focal_px) / max(ipd_px, 1.0)

                all_feats.append(feats)
                all_labels.append(tz / 1000.0)
                all_subj.append(s_idx)
                all_geom.append(geom_dist_m)
                subj_hits += 1
        print(f'  {subj}  focal={focal_px:.0f}px  attempted={subj_attempt:,}'
              f'  usable={subj_hits:,}  ({time.time() - subj_t0:.1f}s)',
              flush=True)

    feats_np = np.asarray(all_feats, dtype=np.float32)
    lbl_np   = np.asarray(all_labels, dtype=np.float32)
    subj_np  = np.asarray(all_subj, dtype=np.int32)
    geom_np  = np.asarray(all_geom, dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=feats_np, targets=lbl_np,
             subjects=subj_np, geom_dist=geom_np)

    print(f'\nTotal usable samples : {len(feats_np):,}  '
          f'({time.time() - t_start:.0f}s total)', flush=True)
    if len(feats_np) > 0:
        err_mm = np.abs(geom_np - lbl_np) * 1000.0
        print(f'Classical IPD formula MAE : {err_mm.mean():.1f} mm   '
              f'(median {np.median(err_mm):.1f} mm)', flush=True)
        print(f'GT distance range         : {lbl_np.min():.2f} – {lbl_np.max():.2f} m',
              flush=True)
    print(f'Saved : {out_path}', flush=True)


if __name__ == '__main__':
    main()
