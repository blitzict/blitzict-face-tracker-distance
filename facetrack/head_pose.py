"""
facetrack.head_pose  —  full 6DoF head-pose estimation via PnP
===============================================================

Current distance-estimation path only corrects for yaw (inferred from nose
offset vs. eye midline). Real webcam use has head motion in all three axes:
a user tilts (roll) while laughing, looks down at a keyboard (pitch), and
glances sideways (yaw). Pitch in particular foreshortens the vertical
eye-to-mouth cue and silently inflates the distance reading.

`cv2.solvePnP` with a canonical metric 5-point face model recovers the full
rotation + translation in one call. It also produces an alternative distance
estimate — `tvec[2]` — that uses ALL five landmarks, not just the two eyes,
so it is noise-averaged compared to raw IPD.

Outputs are exposed both as an absolute distance (metres, mm / 1000) and as
three Euler angles (degrees) so the learned DistanceNet has 6DoF pose as
input features.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

# ── 5-point metric face model (head-centred, millimetres) ────────────────────
# Coordinate convention: +x right-of-head, +y down (image convention),
# +z into-head. Origin at nose tip, which matches the landmark at index 2.
# Values follow adult-population anthropometric averages:
#   IPD 63 mm (eyes at ±31.5 mm on X)
#   Eyes sit ~30 mm above nose tip and ~30 mm deeper into the head
#   Mouth corners ±25 mm from midline, ~50 mm below nose, ~20 mm deeper
# Keep as float64 — solvePnP wants this dtype.
CANONICAL_5POINT_MM = np.array([
    (-31.5, -30.0, -30.0),   # left eye  (viewer's left)
    ( 31.5, -30.0, -30.0),   # right eye
    (  0.0,   0.0,   0.0),   # nose tip
    (-25.0,  50.0, -20.0),   # left mouth corner
    ( 25.0,  50.0, -20.0),   # right mouth corner
], dtype=np.float64)


def _build_camera_matrix(focal_px: float, img_w: int, img_h: int) -> np.ndarray:
    """Simple pinhole K with principal point at image centre."""
    return np.array([
        [focal_px, 0.0,       img_w * 0.5],
        [0.0,      focal_px,  img_h * 0.5],
        [0.0,      0.0,       1.0        ],
    ], dtype=np.float64)


def solve_head_pose(lmks_abs: Tuple[float, ...],
                    focal_px: float, img_w: int, img_h: int
                    ) -> Optional[dict]:
    """
    Run PnP on 5 absolute-coord landmarks; return pose + distance.

    Args
        lmks_abs : 10-tuple (lex, ley, rex, rey, nox, noy, lmx, lmy, rmx, rmy)
                   in original image pixel coordinates.
        focal_px : camera focal length in pixels (calibrated or estimated).
        img_w/h  : image dimensions in pixels.

    Returns
        dict with keys:
            'dist_m' : distance in metres (tvec[2] / 1000)
            'yaw_deg', 'pitch_deg', 'roll_deg'  : Euler angles
            'tvec_mm' : (x, y, z) translation in mm
            'rvec'    : raw rodrigues rotation vector
        or None if solvePnP fails.
    """
    import cv2  # local import — keeps module cheap to import

    image_points = np.array([
        (lmks_abs[0], lmks_abs[1]),    # left eye
        (lmks_abs[2], lmks_abs[3]),    # right eye
        (lmks_abs[4], lmks_abs[5]),    # nose tip
        (lmks_abs[6], lmks_abs[7]),    # left mouth
        (lmks_abs[8], lmks_abs[9]),    # right mouth
    ], dtype=np.float64)

    K    = _build_camera_matrix(focal_px, img_w, img_h)
    dist = np.zeros(5, dtype=np.float64)                   # already undistorted upstream

    # SOLVEPNP_ITERATIVE wants an initial guess for 4+-point cases; seed it
    # with the IPD-derived depth so the optimiser starts in the right basin.
    ipd_px   = float(np.hypot(lmks_abs[0] - lmks_abs[2],
                              lmks_abs[1] - lmks_abs[3]))
    init_z   = 63.0 * focal_px / max(ipd_px, 1.0)          # mm
    init_tvec = np.array([[0.0], [0.0], [init_z]], dtype=np.float64)
    init_rvec = np.zeros((3, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        CANONICAL_5POINT_MM, image_points, K, dist,
        rvec=init_rvec, tvec=init_tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    # Rotation matrix → Euler angles (yaw, pitch, roll) in degrees.
    # Using the standard 'sxyz' (intrinsic XYZ) decomposition:
    #   pitch = rot around X (nodding)
    #   yaw   = rot around Y (shaking head)
    #   roll  = rot around Z (tilting head sideways)
    R, _ = cv2.Rodrigues(rvec)
    # Clamp asin argument to avoid NaN from numerical drift
    sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np.degrees(np.arcsin(sy)))
    yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))

    tvec_mm = tvec.flatten()
    dist_m  = float(tvec_mm[2]) / 1000.0

    return {
        'dist_m':    dist_m,
        'yaw_deg':   yaw,
        'pitch_deg': pitch,
        'roll_deg':  roll,
        'tvec_mm':   (float(tvec_mm[0]), float(tvec_mm[1]), float(tvec_mm[2])),
        'rvec':      rvec.flatten(),
    }
