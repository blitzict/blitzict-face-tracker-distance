"""
inference.py  —  Real-time Face Tracking + Distance Estimation
==============================================================

PIPELINE
────────
 Step 1  Camera frame  (e.g. 1280×720)

 Step 2  DETECTION     (background thread — never blocks display)
         FaceDetectorCNN sliding window on 320×240 downscale → face boxes.

 Step 3  LANDMARK + DISTANCE (per face)
         a. Crop face region from the FULL-RES frame.
         b. LandmarkCNN → (left_eye_xy, right_eye_xy) in the crop.
         c. Convert to original-frame coords, measure IPD in pixels.
         d. distance = (0.063 m × focal_px) / ipd_px
              focal_px: auto-estimated (image_w × FOCAL_RATIO) or user-set via C / --focal.

 Step 4  TRACKING
         CentroidTracker keeps a stable ID on each face across frames.

 Step 5  Draw boxes + eye dots + distance → display.

WHY IPD INSTEAD OF FACE WIDTH?
──────────────────────────────
 Inter-pupillary distance is ±3 mm across adults; face width is ±10 mm.
 Using IPD makes distance cross-user accurate (~±15%) without calibration,
 or ~±5% with one calibration press at 1.0 m.

USAGE
─────
    python inference.py --camera 1
    python inference.py --camera 1 --focal 850       # use measured focal
    python inference.py --det_thresh 0.3             # more sensitive detector

CONTROLS:  Q / ESC  quit      S  screenshot      C  calibrate focal at 1.0 m
"""

import argparse
import queue
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from face_detector_model import FaceDetectorCNN, DETECTOR_PATCH_SIZE
from landmark_model import LandmarkCNN, LANDMARK_INFER_TF

# ── Detection sliding-window config ───────────────────────────────────────────
# Run the detector on a downscaled copy for speed; boxes are scaled back up.
DET_W, DET_H   = 320, 240           # detection resolution (proven working)
DET_SCALES     = [0.50, 0.33, 0.20, 0.13]  # close → far face coverage
DET_STRIDE     = 0.28               # stride = window_size × this value
DET_THRESH     = 0.40               # FaceDetectorCNN sigmoid score ≥ this = face patch
DET_MAX_FACES  = 3                  # top-K peaks to extract per frame

# ── Recogniser config ─────────────────────────────────────────────────────────
REC_SIZE   = 112                    # FaceDistanceNet input size (pixels)
WARN_CONF  = 0.65                   # show warning colour below this identity confidence

# ── Distance estimation — landmark-based (IPD method) ────────────────────────
# Inter-pupillary distance (IPD) averages 63 mm ± 3 mm across adults.
# More consistent than face width (±10 mm) → better cross-user accuracy.
#
#     distance  =  (IPD_METRES × focal_px)  /  ipd_pixels_in_frame
#
# focal_px is estimated from image width × a typical webcam FOV constant.
# Users can refine it with one-shot 'C' calibration at 1.0 m.
IPD_METRES         = 0.063          # average adult inter-pupillary distance
NOSE_DEPTH_M       = 0.020          # average nose-tip forward offset from eye plane
FOCAL_RATIO        = 0.72           # focal_px ≈ image_width × this  (≈75° FOV, typical phone cam)
FALLBACK_FACE_W_M  = 0.145          # used if landmark model isn't loaded
FALLBACK_FACE_FILL = 0.65           # fraction of det window the face occupies

# Temporal smoothing — kills flicker by EMA-blending new detections with the
# last active track state.  Higher alpha = faster reaction, lower = smoother.
TRACK_BOX_ALPHA    = 0.45
TRACK_DIST_ALPHA   = 0.30

# False-positive filters (run AFTER the landmark plausibility check).
# Thresholds are deliberately loose — most false positives also fail the
# geometric constraints, so these filters only need to catch the pathological
# ones that slip through geometry.
SYMMETRY_THRESH   = 0.25   # min horizontal-flip NCC (headphones/hair break strict symmetry)
SKIN_RATIO_THRESH = 0.06   # min fraction of pixels in skin HSV range
LMK_TEMPORAL_MAX  = 0.30   # max frame-to-frame landmark shift (fraction of box size)

# ── Visual style ──────────────────────────────────────────────────────────────
PALETTE = {'box': (0, 230, 118), 'warn': (0, 80, 255), 'fps': (200, 200, 200)}
FONT    = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_detector(ckpt_path: str, device: torch.device) -> FaceDetectorCNN:
    """Load the trained FaceDetectorCNN binary classifier from checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FaceDetectorCNN().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def load_landmark_net(ckpt_path: str, device: torch.device):
    """Load the trained LandmarkCNN (eye-centre regressor). None if missing."""
    from pathlib import Path
    if not Path(ckpt_path).exists():
        return None
    try:
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = LandmarkCNN().to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        err = ckpt.get('val_err', float('nan'))
        print(f"Landmark net  : {ckpt_path}  (val err = {err:.4f} ≈ {err*64:.1f}px on 64×64)")
        return model
    except Exception as e:
        print(f"Landmark net  : failed to load ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fast tensor-based patch extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_patches(frame_t: torch.Tensor,
                    scales: list,
                    stride_ratio: float,
                    patch_size: int):
    """
    Extract all sliding-window crops from a frame tensor using GPU ops only.

    Key optimisation: instead of a PIL resize per patch, we slice the tensor
    directly and call F.interpolate on the WHOLE BATCH for each scale in a
    single GPU operation.  This is 10-20× faster than the PIL approach.

    Args:
        frame_t      : [3, H, W] float32 tensor on device, normalised to [-1, 1]
        scales       : window sizes as fraction of min(H, W)
        stride_ratio : stride = window_size × stride_ratio
        patch_size   : output size in pixels (64 for detector, 112 for recogniser)

    Returns:
        patches : [N, 3, patch_size, patch_size] tensor ready for the model
        meta    : list of (x1, y1, x2, y2) pixel coords for each patch
    """
    _, H, W    = frame_t.shape
    all_p, all_m = [], []

    for scale in scales:
        wsz    = int(min(H, W) * scale)
        if wsz < 16:
            continue
        stride = max(int(wsz * stride_ratio), 4)

        crops, metas = [], []
        for y in range(0, H - wsz + 1, stride):
            for x in range(0, W - wsz + 1, stride):
                crops.append(frame_t[:, y:y+wsz, x:x+wsz])
                metas.append((x, y, x+wsz, y+wsz))

        if not crops:
            continue

        # Resize entire scale-batch in one GPU call
        batch = torch.stack(crops)                                   # [N, 3, wsz, wsz]
        batch = F.interpolate(batch, size=(patch_size, patch_size),
                              mode='bilinear', align_corners=False)  # [N, 3, ps, ps]
        all_p.append(batch)
        all_m.extend(metas)

    if not all_p:
        return None, []

    return torch.cat(all_p, dim=0), all_m


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap voting → bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def scores_to_boxes(meta, scores, H, W):
    """
    Convert sliding-window scores to bounding boxes via single-peak detection.

    Algorithm:
      1. Paint per-pixel max score into a heatmap; track the window size of the
         BEST-scoring patch at each pixel (not the last-written size).
      2. Light Gaussian blur to smooth noise without spreading the blob.
      3. Find the global argmax — this is the face centre.
      4. Size the box from the window size at that peak pixel.

    One box per call.  Extend to multi-peak if needed later.
    """
    score_map = np.zeros((H, W), dtype=np.float32)
    size_map  = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2), s in zip(meta, scores):
        if s < DET_THRESH:
            continue
        wsz = x2 - x1
        old = score_map[y1:y2, x1:x2]
        improved = s > old
        if not improved.any():
            continue
        # Safe assignment through slice — no out= on non-contiguous views
        score_map[y1:y2, x1:x2] = np.where(improved, s,   old)
        size_map[y1:y2, x1:x2]  = np.where(improved, wsz, size_map[y1:y2, x1:x2])

    if score_map.max() < 1e-6:
        return []

    # Light blur — smooth noise only, don't spread blob across unrelated regions
    sigma   = max(int(min(H, W) * 0.04), 3)
    blurred = cv2.GaussianBlur(score_map, (0, 0), sigma)

    # Iteratively extract top-K peaks so multiple faces (or a face + false
    # positive) are all returned.  After each peak we zero a circular region
    # around it so the next argmax finds a different location.
    boxes  = []
    working = blurred.copy()
    for _ in range(DET_MAX_FACES):
        peak_idx   = int(np.argmax(working))
        cy, cx     = divmod(peak_idx, W)
        peak_score = float(working[cy, cx])
        if peak_score < DET_THRESH:
            break

        wsz = float(size_map[cy, cx])
        if wsz < 8:
            wsz = float(min(H, W) * 0.25)

        # Display box ≈ the detection window itself (face fills the window
        # when it fires).  Slightly taller than wide for face aspect.
        hw = int(wsz * 0.48)
        hh = int(wsz * 0.55)
        x1 = max(cx - hw, 0);    y1 = max(cy - hh, 0)
        x2 = min(cx + hw, W-1);  y2 = min(cy + hh, H-1)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, peak_score, wsz))

        # Suppress an area ≈ one face-width around the peak
        r = max(int(wsz * 0.7), 10)
        y1_z = max(cy - r, 0);   y2_z = min(cy + r, H)
        x1_z = max(cx - r, 0);   x2_z = min(cx + r, W)
        working[y1_z:y2_z, x1_z:x2_z] = 0

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Centroid tracker
# ─────────────────────────────────────────────────────────────────────────────

class SingleFaceTracker:
    """
    Sticky single-face tracker.  Assumes one user at a time — so instead of
    spawning new tracks whenever detections appear far from the active one
    (which causes the classic "box jumps to a wall" behaviour), this tracker:

      • Accepts only the detection CLOSEST to the current track, if that
        detection is within MAX_DISTANCE.
      • Ignores all other candidates that frame.
      • Drops the track only after MAX_DISAPPEARED frames with no nearby match.
      • Median-filters distance over the last DIST_WINDOW frames so a single
        noisy IPD measurement can't swing the number.
      • EMA-blends box coordinates so the box doesn't jitter frame-to-frame.
    """

    MAX_DISAPPEARED = 45     # ~1.5 s at 30 FPS
    MAX_DISTANCE    = 200    # pixel radius for "same face"
    DIST_WINDOW     = 7      # frames in the distance median window
    LOCK_CONFIRM    = 2      # consecutive confirming frames required to lock on

    def __init__(self):
        self.track     = None       # the one active track (dict) or None
        self.age       = 0          # consecutive frames with no matching detection
        self.dist_q    = []         # rolling window of recent distances
        self.candidate = None       # pending lock candidate (dict)
        self.cand_hits = 0          # how many frames this candidate has been seen near

    def _smooth_dist(self, new_dist):
        self.dist_q.append(new_dist)
        if len(self.dist_q) > self.DIST_WINDOW:
            self.dist_q.pop(0)
        # Median → robust to outliers (far better than EMA for noisy IPD)
        s = sorted(self.dist_q)
        return round(s[len(s) // 2], 2)

    def update(self, detections):
        # No detections this frame → age the existing track
        if not detections:
            if self.track is not None:
                self.age += 1
                if self.age > self.MAX_DISAPPEARED:
                    self.track, self.dist_q = None, []
                    self.candidate, self.cand_hits = None, 0
                    return []
                return [self.track]
            # Reset pending lock candidate when detection dries up
            self.candidate, self.cand_hits = None, 0
            return []

        cents = [((d['box'][0]+d['box'][2])//2,
                  (d['box'][1]+d['box'][3])//2) for d in detections]

        # First-time lock-on — require LOCK_CONFIRM consecutive confirmations
        # on a nearby candidate before committing.  Stops a single-frame wall
        # detection from stealing the initial lock.
        if self.track is None:
            # Pick the best candidate this frame (highest-scoring detection)
            best_i = max(range(len(detections)), key=lambda i: detections[i].get('conf', 1.0))
            cx, cy = cents[best_i]

            if self.candidate is None:
                self.candidate = detections[best_i]
                self.candidate['cx'], self.candidate['cy'] = cx, cy
                self.cand_hits = 1
                return []

            # Is this detection near the pending candidate?
            dx = cx - self.candidate['cx']; dy = cy - self.candidate['cy']
            if (dx*dx + dy*dy) ** 0.5 < self.MAX_DISTANCE:
                self.cand_hits += 1
                # Update candidate to latest observation
                self.candidate = {**detections[best_i], 'cx': cx, 'cy': cy}
                if self.cand_hits >= self.LOCK_CONFIRM:
                    self.track = self.candidate
                    self.dist_q = [self.track['dist']]
                    self.age = 0
                    self.candidate, self.cand_hits = None, 0
                    return [self.track]
            else:
                # Drift / different place — restart candidate counting
                self.candidate = detections[best_i]
                self.candidate['cx'], self.candidate['cy'] = cx, cy
                self.cand_hits = 1
            return []

        # Pick the detection closest to the existing track
        tx, ty = self.track['cx'], self.track['cy']
        dists  = [np.hypot(cx-tx, cy-ty) for cx, cy in cents]
        best_i = int(np.argmin(dists))

        if dists[best_i] > self.MAX_DISTANCE:
            # Nothing near the current track — keep the old box, age it
            self.age += 1
            if self.age > self.MAX_DISAPPEARED:
                # Lost too long — drop the track and restart candidate matching
                self.track, self.dist_q = None, []
                self.candidate, self.cand_hits = None, 0
            return [self.track] if self.track else []

        # Matched: EMA-blend box, median-filter distance
        new = detections[best_i]
        old = self.track

        # ── Temporal landmark stability gate ──────────────────────────────
        # If landmarks jumped > LMK_TEMPORAL_MAX × box_size, reject this match.
        # Genuine faces move smoothly; a false-positive on a different object
        # will have landmarks in completely different places.
        old_lmks = old.get('lmks_abs')
        new_lmks = new.get('lmks_abs')
        if old_lmks is not None and new_lmks is not None:
            ox1, oy1, ox2, oy2 = old['box']
            box_size = max(ox2 - ox1, oy2 - oy1, 1)
            max_shift = 0.0
            for i in range(5):
                dx = old_lmks[2*i]   - new_lmks[2*i]
                dy = old_lmks[2*i+1] - new_lmks[2*i+1]
                s  = (dx*dx + dy*dy) ** 0.5
                if s > max_shift:
                    max_shift = s
            if max_shift > LMK_TEMPORAL_MAX * box_size:
                # Landmark layout jumped — reject, age the existing track
                self.age += 1
                if self.age > self.MAX_DISAPPEARED:
                    self.track, self.dist_q = None, []
                    self.candidate, self.cand_hits = None, 0
                    return []
                return [self.track]

        a_b = TRACK_BOX_ALPHA
        ox1, oy1, ox2, oy2 = old['box']
        nx1, ny1, nx2, ny2 = new['box']
        smooth_box = (int((1-a_b)*ox1 + a_b*nx1), int((1-a_b)*oy1 + a_b*ny1),
                      int((1-a_b)*ox2 + a_b*nx2), int((1-a_b)*oy2 + a_b*ny2))

        self.track = {**new,
                      'box':  smooth_box,
                      'dist': self._smooth_dist(new['dist']),
                      'cx':   cents[best_i][0],
                      'cy':   cents[best_i][1]}
        self.age = 0
        return [self.track]


# Alias so the rest of the code doesn't care which tracker is used
CentroidTracker = SingleFaceTracker


# ─────────────────────────────────────────────────────────────────────────────
# Cheap false-positive filters — geometry alone can't fully separate faces from
# textured junk, so we add two visual priors that real faces satisfy.
# ─────────────────────────────────────────────────────────────────────────────

def symmetry_score(crop_bgr: np.ndarray) -> float:
    """
    Normalised cross-correlation between the crop and its horizontal flip.
    Real faces: 0.7-0.9.  Asymmetric junk (walls, furniture): 0.1-0.4.
    Uses a 48×48 grayscale downsample — runs in ~0.1 ms.
    """
    if crop_bgr.size == 0:
        return 0.0
    small = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flipped = gray[:, ::-1]
    g = gray - gray.mean()
    f = flipped - flipped.mean()
    denom = np.sqrt((g * g).sum() * (f * f).sum())
    if denom < 1e-6:
        return 0.0
    return float((g * f).sum() / denom)


# Skin hue bands cover typical pink/tan/brown skin across all ethnicities.
_SKIN_LO1 = np.array([0,   20, 50],  dtype=np.uint8)
_SKIN_HI1 = np.array([25, 255, 255], dtype=np.uint8)
_SKIN_LO2 = np.array([160, 20, 50],  dtype=np.uint8)  # wraps around on the other side of red
_SKIN_HI2 = np.array([180, 255, 255], dtype=np.uint8)


def skin_ratio(crop_bgr: np.ndarray) -> float:
    """Fraction of pixels in typical skin-tone HSV range.  Real faces ≥ 0.3-0.7."""
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _SKIN_LO1, _SKIN_HI1)
    m2 = cv2.inRange(hsv, _SKIN_LO2, _SKIN_HI2)
    return float(np.count_nonzero(cv2.bitwise_or(m1, m2))) / m1.size


# ─────────────────────────────────────────────────────────────────────────────
# Background detection worker
# ─────────────────────────────────────────────────────────────────────────────

class DetectionWorker:
    """
    Runs the full two-stage inference pipeline in a background daemon thread.

    The main display loop calls  worker.submit(frame)  (non-blocking — drops
    the frame if the worker is busy) and reads  worker.get_detections().
    This decouples camera FPS from inference FPS so the video is always smooth.

    Stage 1  FaceDetectorCNN sliding window on a 320×240 downscale.
    Stage 2  FaceDistanceNet on each cropped face region.
    """

    def __init__(self, detector, device, det_thresh, focal_px=None,
                 landmark_net=None):
        self.detector     = detector
        self.device       = device
        self.det_thresh   = det_thresh
        self.focal_px     = focal_px        # None → auto-estimate per frame
        self.landmark_net = landmark_net    # LandmarkCNN — eye + landmark regressor

        self._q       = queue.Queue(maxsize=1)
        self._lock    = threading.Lock()
        self._results = []
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def submit(self, frame: np.ndarray):
        """Send a frame to the worker.  Non-blocking; drops frame if busy."""
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def get_detections(self):
        with self._lock:
            return list(self._results)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                frame = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            dets = self._process(frame)
            with self._lock:
                self._results = dets

    @torch.no_grad()
    def _process(self, frame_bgr: np.ndarray):
        H_orig, W_orig = frame_bgr.shape[:2]

        # ── Stage 1: detect faces on a small copy ────────────────────────────
        # Downscale for speed; the detector still finds faces reliably.
        det_frame = cv2.resize(frame_bgr, (DET_W, DET_H))
        rgb_small = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)

        # Convert to normalised tensor ONCE — no per-patch PIL conversion
        t = torch.from_numpy(rgb_small).float().permute(2, 0, 1).div(255.0)
        t = (t - 0.5) / 0.5
        t = t.to(self.device)

        patches, meta = extract_patches(t, DET_SCALES, DET_STRIDE, DETECTOR_PATCH_SIZE)
        if patches is None:
            return []

        # Batch forward through FaceDetectorCNN
        scores = []
        for i in range(0, len(patches), 512):
            logits = self.detector(patches[i:i+512])
            scores.extend(torch.sigmoid(logits).cpu().tolist())

        # Connected components → one box per face
        boxes_det = scores_to_boxes(meta, scores, DET_H, DET_W)
        if not boxes_det:
            return []

        # Scale boxes back to original resolution
        sx, sy = W_orig / DET_W, H_orig / DET_H
        boxes = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), wsz)
                 for x1, y1, x2, y2, _score, wsz in boxes_det]

        # ── Stage 2: distance estimation ─────────────────────────────────────
        # Priority order:
        #   1. Landmark-based IPD  (works cross-user without calibration)
        #   2. Fallback geometric formula using face-width assumption
        # Focal auto-estimated from frame width; override with --focal or press C.
        focal_px = self.focal_px if self.focal_px else W_orig * FOCAL_RATIO

        detections = []
        for x1, y1, x2, y2, wsz in boxes:
            ipd_px   = None
            eyes_abs = None

            yaw_deg = 0.0
            if self.landmark_net is not None:
                # Pad the crop by 20% around the display box — the landmark
                # model was trained on padded crops, so inference must match.
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = int(bw * 0.20), int(bh * 0.20)
                cx1 = max(x1 - pad_x, 0);          cy1 = max(y1 - pad_y, 0)
                cx2 = min(x2 + pad_x, W_orig);     cy2 = min(y2 + pad_y, H_orig)

                lm_crop = frame_bgr[cy1:cy2, cx1:cx2]
                if lm_crop.size > 0:
                    from PIL import Image as _PIL
                    pil    = _PIL.fromarray(lm_crop[:, :, ::-1])
                    cw, ch = pil.size
                    t_in   = LANDMARK_INFER_TF(pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out10 = self.landmark_net(t_in).cpu().squeeze(0).tolist()
                    # Order: [lex, ley, rex, rey, nox, noy, lmx, lmy, rmx, rmy]
                    lx, ly, rx, ry, nx_, ny_, lmx, lmy, rmx, rmy = out10

                    # Geometric plausibility — constraints that real faces satisfy
                    # but random wall/background patches almost never do.
                    # Eye ordering (lx is always the LEFT landmark in image coords,
                    # which sits to the VIEWER's right since CelebA labels from
                    # the subject's perspective).
                    eye_left_x, eye_right_x = min(lx, rx), max(lx, rx)
                    eye_avg_y       = (ly + ry) * 0.5
                    mouth_avg_y     = (lmy + rmy) * 0.5
                    mouth_avg_x     = (lmx + rmx) * 0.5
                    ipd_norm        = abs(lx - rx)
                    mouth_w_norm    = abs(lmx - rmx)

                    # Core geometric plausibility — loosened so valid faces at
                    # slight angles, with headphones, hair over forehead, etc.
                    # still pass.  False positives usually fail multiple of these
                    # rather than just one.
                    eyes_ok = (
                        # Bounds — landmarks inside the crop (generous margin)
                        0.02 < lx  < 0.98 and 0.02 < rx  < 0.98 and
                        0.02 < ly  < 0.85 and 0.02 < ry  < 0.85 and
                        0.02 < nx_ < 0.98 and 0.15 < ny_ < 0.98 and
                        0.02 < lmx < 0.98 and 0.02 < rmx < 0.98 and
                        0.25 < lmy < 0.99 and 0.25 < rmy < 0.99 and
                        # Eyes roughly horizontal (loose)
                        abs(ly - ry) < 0.18 and
                        # Sensible eye spacing
                        0.12 < ipd_norm < 0.80 and
                        # Nose below eye line, above mouth line
                        ny_ > eye_avg_y and
                        ny_ < mouth_avg_y and
                        # Mouth below nose
                        mouth_avg_y > ny_
                    )
                    if not eyes_ok:
                        continue

                    # Visual plausibility filters — soft-scored, not hard-rejected.
                    # Candidates accumulate a "face-ness" score; if it's too low
                    # we skip, but we don't require perfection on every signal.
                    sym = symmetry_score(lm_crop)
                    skn = skin_ratio(lm_crop)
                    face_signals = int(sym >= SYMMETRY_THRESH) + int(skn >= SKIN_RATIO_THRESH)
                    # Need at least one of the two visual signals to pass.
                    # Together with geometric plausibility this is enough.
                    if face_signals < 1:
                        continue

                    lx_abs  = cx1 + lx  * cw;  ly_abs  = cy1 + ly  * ch
                    rx_abs  = cx1 + rx  * cw;  ry_abs  = cy1 + ry  * ch
                    nx_abs  = cx1 + nx_ * cw;  ny_abs  = cy1 + ny_ * ch
                    lmx_abs = cx1 + lmx * cw;  lmy_abs = cy1 + lmy * ch
                    rmx_abs = cx1 + rmx * cw;  rmy_abs = cy1 + rmy * ch
                    ipd_raw_px = ((lx_abs - rx_abs)**2 + (ly_abs - ry_abs)**2) ** 0.5
                    eyes_abs   = (int(lx_abs), int(ly_abs), int(rx_abs), int(ry_abs))

                    # ── Yaw correction (compute first — needed for head-width) ──
                    # When the face turns by θ:
                    #   nose_offset / ipd_measured = (nose_depth / ipd_real) × tan(θ)
                    # For adult faces: nose_depth / ipd_real ≈ 20mm / 63mm ≈ 0.317
                    # Then corrected_ipd = ipd_measured / cos(θ).
                    eye_mid_x   = (lx_abs + rx_abs) * 0.5
                    nose_off    = nx_abs - eye_mid_x
                    ratio       = NOSE_DEPTH_M / IPD_METRES          # ~0.317
                    tan_yaw     = (nose_off / max(ipd_raw_px, 1.0)) / ratio
                    tan_yaw     = max(min(tan_yaw, 2.75), -2.75)     # clamp ≤ ±70°
                    cos_yaw     = 1.0 / (1.0 + tan_yaw * tan_yaw) ** 0.5
                    yaw_deg     = float(np.degrees(np.arctan(tan_yaw)))
                    ipd_px      = ipd_raw_px / max(cos_yaw, 0.35)    # un-foreshorten

                    # ── Full-head box from landmarks ──────────────────────────
                    # Anthropometric proportions (Farkas craniofacial norms):
                    #   head width / IPD ≈ 2.3   (includes ears)
                    #   forehead above eyes ≈ 1.1 × (eye-to-mouth)
                    #   chin below mouth    ≈ 0.7 × (eye-to-mouth)
                    # Uses yaw-corrected IPD so box width stays constant when
                    # the head turns — the head's 3D width is rotation-invariant.
                    eye_mid_y   = (ly_abs + ry_abs) * 0.5
                    mouth_mid_y = (lmy_abs + rmy_abs) * 0.5
                    eye_to_mouth = max(mouth_mid_y - eye_mid_y, 10.0)

                    head_w_px = max(2.30 * ipd_px, 40.0)
                    face_top  = eye_mid_y  - 1.10 * eye_to_mouth
                    face_bot  = mouth_mid_y + 0.70 * eye_to_mouth
                    face_cx   = eye_mid_x                            # head centre

                    tx1 = max(int(face_cx - head_w_px * 0.5), 0)
                    tx2 = min(int(face_cx + head_w_px * 0.5), W_orig - 1)
                    ty1 = max(int(face_top), 0)
                    ty2 = min(int(face_bot), H_orig - 1)
                    if tx2 > tx1 and ty2 > ty1:
                        x1, y1, x2, y2 = tx1, ty1, tx2, ty2

            if ipd_px and ipd_px > 4:
                dist_m = (IPD_METRES * focal_px) / ipd_px
            else:
                # Fallback: face-width geometric formula
                face_px = max(wsz * sx * FALLBACK_FACE_FILL, 1.0)
                dist_m  = (FALLBACK_FACE_W_M * focal_px) / face_px

            dist_m = round(max(0.3, min(dist_m, 10.0)), 2)

            # 10-tuple of landmark pixel coords in original frame — used by tracker
            # for frame-to-frame landmark-stability gate.
            lmks_abs = None
            if eyes_abs is not None:
                lmks_abs = (lx_abs, ly_abs, rx_abs, ry_abs,
                            nx_abs, ny_abs,
                            lmx_abs, lmy_abs, rmx_abs, rmy_abs)
            detections.append({
                'box':      (x1, y1, x2, y2),
                'name':     'FACE',
                'conf':     1.0,
                'dist':     dist_m,
                'ipd_px':   ipd_px if ipd_px else 0.0,
                'yaw':      yaw_deg,
                'eyes':     eyes_abs,
                'lmks_abs': lmks_abs,
                'wsz':      wsz * sx,
            })

        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_label(frame, box, name, conf, dist_m):
    """Draw a bounding box and distance label on the frame."""
    x1, y1, x2, y2 = box
    colour = PALETTE['box']
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    txt   = f"{dist_m:.2f} m"
    pad, lh = 4, 18
    bg_y1 = max(y1 - lh - pad * 2, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x2, y1), colour, -1)
    cv2.putText(frame, txt, (x1 + pad, bg_y1 + pad + lh - 2),
                FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main capture + display loop
# ─────────────────────────────────────────────────────────────────────────────

def run(camera_idx:  int   = 0,
        det_ckpt:    str   = 'checkpoints/face_detector.pth',
        det_thresh:  float = DET_THRESH,
        focal_px:    float = 0.0):      # 0 = auto from image width

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")

    print(f"Detector   : {det_ckpt}")
    detector = load_detector(det_ckpt, device)
    print(f"Det thresh : {det_thresh}")

    landmark_net = load_landmark_net('checkpoints/landmark_net.pth', device)
    if landmark_net is None:
        print("Landmark net : not found — using face-width fallback  "
              "(train with: python train_landmarks.py)")

    # Start background detection thread
    # focal_px=0 means auto-estimate per frame from image width
    worker = DetectionWorker(detector, device, det_thresh,
                             focal_px if focal_px > 0 else None,
                             landmark_net=landmark_net)
    worker.start()

    tracker = CentroidTracker()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        worker.stop()
        raise RuntimeError(f"Cannot open camera {camera_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Running — Q/ESC to quit  S to screenshot  C to calibrate distance at 1.0m")

    fps_t = time.time(); fps_n = 0; fps_val = 0.0
    shot_n  = 0
    cal_msg = ''   # on-screen calibration feedback
    cal_t   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        # Feed frame to background worker (non-blocking)
        worker.submit(frame)

        # Pull latest detections and update the sticky single-face tracker
        tracks = tracker.update(worker.get_detections())

        h = frame.shape[0]
        for tr in tracks:
            draw_label(frame, tr['box'], tr['name'], tr['conf'], tr['dist'])

        # FPS overlay
        fps_n += 1
        now = time.time()
        if now - fps_t >= 0.5:
            fps_val = fps_n / (now - fps_t)
            fps_n   = 0
            fps_t   = now

        focal_label = f"{worker.focal_px:.0f}px" if worker.focal_px else "auto"
        cv2.putText(frame, f"FPS: {fps_val:.0f}  focal: {focal_label}",
                    (10, h-12), FONT, 0.5, PALETTE['fps'], 1, cv2.LINE_AA)
        cv2.putText(frame, "Q/ESC quit   S screenshot   C calibrate@1m",
                    (10, h-30), FONT, 0.4, PALETTE['fps'], 1, cv2.LINE_AA)
        if cal_msg and time.time() - cal_t < 3.0:
            cv2.putText(frame, cal_msg, (10, 60),
                        FONT, 0.7, (0, 230, 118), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition + Distance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            fname = f"screenshot_{shot_n:04d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            shot_n += 1
        if key == ord('c'):
            # Calibrate: stand exactly 1.0 m away, press C.
            # Prefer landmark IPD; fall back to face-width if landmarks unavailable.
            #   focal = (ipd_px × known_dist) / IPD_METRES
            dets = worker.get_detections()
            if dets:
                d         = dets[0]
                ipd_px    = d.get('ipd_px', 0)
                wsz_orig  = d.get('wsz', 0)
                known_dist = 1.0   # metres
                new_focal = 0.0

                if ipd_px and ipd_px > 4:
                    new_focal = (ipd_px * known_dist) / IPD_METRES
                    method    = 'IPD'
                elif wsz_orig > 0:
                    new_focal = (wsz_orig * FALLBACK_FACE_FILL * known_dist) / FALLBACK_FACE_W_M
                    method    = 'face-width'

                if new_focal > 0:
                    worker.focal_px = new_focal
                    cal_msg = f"Calibrated ({method}): focal={new_focal:.0f}px"
                    cal_t   = time.time()
                    print(cal_msg)
                    print(f"  Tip: --focal {new_focal:.0f}  to skip calibration next time")
                else:
                    cal_msg = "No face detected — move closer and try again"
                    cal_t   = time.time()
            else:
                cal_msg = "No detection — move into frame first"
                cal_t   = time.time()

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time face recognition + distance estimation')
    parser.add_argument('--camera',     type=int,   default=0,
                        help='Camera index (default 0)')
    parser.add_argument('--det_ckpt',   type=str,   default='checkpoints/face_detector.pth',
                        help='FaceDetectorCNN checkpoint')
    parser.add_argument('--det_thresh', type=float, default=DET_THRESH,
                        help='Detector sigmoid threshold (default 0.40)')
    parser.add_argument('--focal',     type=float, default=0.0,
                        help='Camera focal length in pixels. 0 = auto-estimate from frame width. '
                             'Press C at 1.0m to calibrate, then pass the printed value.')
    args = parser.parse_args()
    run(camera_idx=args.camera, det_ckpt=args.det_ckpt,
        det_thresh=args.det_thresh, focal_px=args.focal)


if __name__ == '__main__':
    main()
