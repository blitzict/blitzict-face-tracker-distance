"""
All tunable constants for the pipeline — kept in one place so you can sweep
them without hunting through module files.

Grouped by subsystem:
    - detection (sliding window)
    - distance  (geometric formula)
    - smoothing (tracker EMA + median)
    - filters   (symmetry, skin, temporal stability)
"""

# ── Detection sliding-window ──────────────────────────────────────────────────
DET_W, DET_H   = 320, 240             # detection resolution (downscaled)
DET_SCALES     = [0.50, 0.33, 0.20, 0.13]  # close → far  (fraction of min(H,W))
DET_STRIDE     = 0.28                 # stride = window_size × this value
DET_THRESH     = 0.40                 # sigmoid score ≥ this counts as "face patch"
DET_MAX_FACES  = 3                    # top-K peaks to extract per frame

# ── Distance estimation (IPD-based geometry) ─────────────────────────────────
#   dist = (IPD_M × focal_px) / ipd_pixels
#   focal_px = image_width × FOCAL_RATIO  (unless calibrated)
IPD_METRES         = 0.063            # adult inter-pupillary distance average
NOSE_DEPTH_M       = 0.020            # nose-tip forward offset from eye plane
FOCAL_RATIO        = 0.72             # ≈ 75° FOV — typical phone camera
FALLBACK_FACE_W_M  = 0.145            # used if landmark model is missing
FALLBACK_FACE_FILL = 0.65             # face fraction of detection window

# ── Tracker smoothing ─────────────────────────────────────────────────────────
TRACK_BOX_ALPHA  = 0.45               # box EMA — higher = faster reaction
TRACK_DIST_ALPHA = 0.30               # kept for backwards compat; main dist uses median

# ── False-positive filters ────────────────────────────────────────────────────
SYMMETRY_THRESH   = 0.25              # min horizontal-flip NCC of the crop
SKIN_RATIO_THRESH = 0.06              # min fraction of pixels in skin HSV range
LMK_TEMPORAL_MAX  = 0.30              # max per-landmark shift / box-size between frames

# ── Visual style ──────────────────────────────────────────────────────────────
PALETTE = {
    'box':  (0, 230, 118),
    'warn': (0,  80, 255),
    'fps':  (200, 200, 200),
}
