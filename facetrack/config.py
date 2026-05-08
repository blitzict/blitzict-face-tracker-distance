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
# Scales as fraction of min(H,W)=240. At 0.90 the window is 216 px in the
# detection frame (~864 px in a 1280×720 original) — enough to contain a face
# that fills most of the frame, i.e. the user very close to the camera.
# Smallest scale 0.16 → ~38 px window → faces ~150 px wide in a 1280 frame
# (≈ 2 m on a typical webcam). The 0.11 scale was dropped: it generated >50%
# of all patches but only added detection range past 3 m, which a desk
# webcam rarely needs. Add it back if you need across-the-room detection.
DET_SCALES     = [0.70, 0.50, 0.35, 0.25, 0.16]   # very close → far
DET_STRIDE     = 0.28                 # stride = window_size × this value
DET_THRESH     = 0.40                 # sigmoid score ≥ this counts as "face patch"
DET_MAX_FACES  = 3                    # top-K peaks to extract per frame

# ── Distance estimation (IPD-based geometry) ─────────────────────────────────
#   dist = (anatomical_metres × focal_px) / anatomical_pixels
#   focal_px = image_width × FOCAL_RATIO  (unless calibrated)
IPD_METRES         = 0.063            # adult inter-pupillary distance average
NOSE_DEPTH_M       = 0.020            # nose-tip forward offset from eye plane
FOCAL_RATIO        = 0.72             # ≈ 75° FOV — typical phone camera
FALLBACK_FACE_W_M  = 0.145            # used if landmark model is missing
FALLBACK_FACE_FILL = 0.65             # face fraction of detection window

# Multi-cue anthropometric constants. Each cue gives an independent distance
# estimate; we weighted-average them to reduce landmark-noise-driven jitter.
# Weights are ~1/σ² of the anatomical variability, normalised: IPD is the
# most stable (~3 mm σ), mouth width the least (~5 mm σ).
EYE_TO_MOUTH_M   = 0.065              # vertical eye-midline → mouth-midline
MOUTH_WIDTH_M    = 0.050              # mouth corner to corner
DIST_CUE_WEIGHTS = {                  # must sum to 1
    'ipd':          0.54,
    'eye_to_mouth': 0.34,
    'mouth_width':  0.12,
}

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
