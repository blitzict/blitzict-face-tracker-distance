"""
facetrack.distance_net  —  learned distance regressor
=====================================================

Replaces the hand-derived geometric formula
    dist = (IPD_metres × focal_px) / IPD_px
with a small MLP trained on real (features → ground-truth-distance) pairs.

The formula is close to correct but silently wrong along several axes:
    • assumes the population-average IPD (±3 mm per-person residual error)
    • only corrects yaw, ignores pitch / roll foreshortening
    • ignores per-camera focal-length variation beyond a linear scale
    • accumulates landmark noise linearly into the output

A learned head on top of the same inputs + full 6DoF PnP pose + the
raw geometric estimate folds all of those corrections implicitly from
training data (MPIIGaze: 15 laptops × ~14k images × Kinect-free
intrinsic-based ground-truth distance).

Architecture is intentionally tiny (~6 k params) — this is a regressor,
not a backbone. It runs per-detection in microseconds.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from facetrack.head_pose import solve_head_pose


# Number of scalar features fed to the MLP. Keep in sync with
# extract_distance_features() below — order matters for checkpoint compat.
NUM_DISTANCE_FEATURES = 20


class DistanceNet(nn.Module):
    """20-D feature → scalar distance (metres) MLP."""

    def __init__(self, in_features: int = NUM_DISTANCE_FEATURES,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),  nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),       nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),  nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softplus + small offset keeps output strictly positive without
        # saturating. Training data is 0.3–2.0 m, inference is the same band.
        raw = self.net(x).squeeze(-1)
        return torch.nn.functional.softplus(raw) + 0.1


def extract_distance_features(lmks_abs: Tuple[float, ...],
                              box: Tuple[int, int, int, int],
                              focal_px: float,
                              img_w: int, img_h: int) -> Optional[np.ndarray]:
    """
    Build the 20-D feature vector used for both training and inference.

    Feature order (MUST match between training + deployment):
        0..9   : 5 landmarks normalised into [0, 1] inside the box
        10     : box aspect ratio (w / h)
        11     : box size / image_width              (scale proxy)
        12     : ipd_px / image_width                (classic IPD cue)
        13     : eye_to_mouth_px / image_width       (vertical cue)
        14     : mouth_width_px / image_width        (horizontal cue)
        15     : nose_offset_x / ipd_px              (yaw proxy from geom)
        16     : PnP yaw degrees / 90
        17     : PnP pitch degrees / 90
        18     : PnP roll degrees / 90
        19     : PnP tvec_z / image_width            (an independent dist est.)

    Returns None if the face is degenerate (IPD < 4 px or PnP fails).
    """
    lx, ly, rx, ry, nx, ny, lmx, lmy, rmx, rmy = lmks_abs
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1);  bh = max(y2 - y1, 1)

    ipd_px = float(np.hypot(lx - rx, ly - ry))
    if ipd_px < 4.0:
        return None

    pose = solve_head_pose(lmks_abs, focal_px, img_w, img_h)
    if pose is None:
        return None

    eye_mid_x   = (lx + rx) * 0.5
    eye_mid_y   = (ly + ry) * 0.5
    mouth_mid_y = (lmy + rmy) * 0.5

    e2m_px = max(mouth_mid_y - eye_mid_y, 1.0)
    mw_px  = float(np.hypot(lmx - rmx, lmy - rmy))

    # Normalised landmark coords inside the box
    norm_lmks = []
    for (px, py) in [(lx, ly), (rx, ry), (nx, ny), (lmx, lmy), (rmx, rmy)]:
        norm_lmks.extend([(px - x1) / bw, (py - y1) / bh])

    feats = np.array([
        *norm_lmks,                                     # 0..9
        bw / max(bh, 1),                                # 10
        max(bw, bh) / img_w,                            # 11
        ipd_px   / img_w,                               # 12
        e2m_px   / img_w,                               # 13
        mw_px    / img_w,                               # 14
        (nx - eye_mid_x) / ipd_px,                      # 15
        pose['yaw_deg']   / 90.0,                       # 16
        pose['pitch_deg'] / 90.0,                       # 17
        pose['roll_deg']  / 90.0,                       # 18
        pose['tvec_mm'][2] / 1000.0 / max(img_w, 1)     # 19  (m/px scale)
    ], dtype=np.float32)

    return feats


@torch.no_grad()
def predict_distance(model: DistanceNet,
                     feats: np.ndarray,
                     device: torch.device = torch.device('cpu')) -> float:
    """Single-sample inference helper."""
    x = torch.from_numpy(feats).to(device).float().unsqueeze(0)
    y = model(x)
    return float(y.item())
