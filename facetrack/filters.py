"""
facetrack.filters  —  cheap false-positive filters
===================================================

Three signals we use AFTER the landmark CNN fires, to decide whether a
candidate detection really is a face:

    1. Geometric plausibility — are the 5 landmark positions physically
       consistent with a human face?  (eyes horizontal, nose between them,
       mouth below, etc.)
    2. Bilateral symmetry     — a flipped face still resembles the original;
       random junk doesn't.
    3. Skin-tone ratio         — faces contain a lot of skin-coloured pixels.

Each function is small and fast (≤0.5 ms) so they add no meaningful latency.
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometric plausibility of the 5 landmarks
# ─────────────────────────────────────────────────────────────────────────────

def plausible_face_geometry(lx: float, ly: float, rx: float, ry: float,
                             nx: float, ny: float,
                             lmx: float, lmy: float,
                             rmx: float, rmy: float) -> bool:
    """
    Check anatomic constraints on the 5 landmarks (all in [0, 1] crop coords).

    Returns True if the layout is plausible for a face; False otherwise.
    Loose bounds so real faces at odd angles / with occlusions still pass.
    """
    eye_left_x  = min(lx, rx)
    eye_right_x = max(lx, rx)
    eye_avg_y   = (ly + ry) * 0.5
    mouth_avg_y = (lmy + rmy) * 0.5
    ipd_norm    = abs(lx - rx)

    return (
        # Bounds: all landmarks inside the crop with a small margin
        0.02 < lx  < 0.98 and 0.02 < rx  < 0.98 and
        0.02 < ly  < 0.85 and 0.02 < ry  < 0.85 and
        0.02 < nx  < 0.98 and 0.15 < ny  < 0.98 and
        0.02 < lmx < 0.98 and 0.02 < rmx < 0.98 and
        0.25 < lmy < 0.99 and 0.25 < rmy < 0.99
        # Eyes roughly horizontal
        and abs(ly - ry) < 0.18
        # Sensible eye spacing (not tiny, not the whole crop)
        and 0.12 < ipd_norm < 0.80
        # Nose below eye line, above mouth line
        and ny > eye_avg_y
        and ny < mouth_avg_y
        # Mouth below nose
        and mouth_avg_y > ny
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bilateral symmetry score
# ─────────────────────────────────────────────────────────────────────────────

def symmetry_score(crop_bgr: np.ndarray) -> float:
    """
    Normalised cross-correlation (NCC) of the crop with its horizontal flip.

    Real faces:        0.7 – 0.9
    Walls / furniture: 0.1 – 0.4
    Textureless crop:  returned as 0.0 (undefined NCC)

    Uses a 48×48 grayscale downsample for speed (~0.1 ms).
    """
    if crop_bgr.size == 0:
        return 0.0
    small   = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flipped = gray[:, ::-1]
    g = gray    - gray.mean()
    f = flipped - flipped.mean()
    denom = float(np.sqrt((g * g).sum() * (f * f).sum()))
    if denom < 1e-6:
        return 0.0
    return float((g * f).sum() / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Skin-tone ratio in HSV space
# ─────────────────────────────────────────────────────────────────────────────

_SKIN_LO1 = np.array([  0, 20, 50], dtype=np.uint8)
_SKIN_HI1 = np.array([ 25, 255, 255], dtype=np.uint8)
_SKIN_LO2 = np.array([160, 20, 50], dtype=np.uint8)   # wraps past red
_SKIN_HI2 = np.array([180, 255, 255], dtype=np.uint8)


def skin_ratio(crop_bgr: np.ndarray) -> float:
    """
    Fraction of pixels in typical skin-tone HSV range.

    Covers pink, tan, brown — works across all ethnicities because the
    H band is intentionally wide.  Real faces: 0.3 – 0.7.
    """
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, _SKIN_LO1, _SKIN_HI1)
    m2  = cv2.inRange(hsv, _SKIN_LO2, _SKIN_HI2)
    return float(np.count_nonzero(cv2.bitwise_or(m1, m2))) / m1.size
