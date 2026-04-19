"""
facetrack.pipeline  —  detection → landmark → distance glue
============================================================

Three stages run per frame in a background worker thread:

    Stage 1  FaceDetectorCNN sliding window on a 320×240 downscale
             → a heatmap → top-K peaks → candidate boxes.

    Stage 2  For each candidate box: pad by 20%, crop from full resolution,
             run LandmarkCNN → 5 landmarks.  Apply geometric plausibility +
             symmetry + skin-tone filters.  Discard candidates that fail.

    Stage 3  From landmarks: estimate yaw, correct IPD, compute distance,
             derive a tight head-bounding box.

The worker is fed via `submit(frame)` and read via `get_detections()` — the
main display loop never blocks on inference.
"""

from __future__ import annotations

import queue
import threading
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from facetrack.config import (
    DET_W, DET_H, DET_SCALES, DET_STRIDE, DET_THRESH, DET_MAX_FACES,
    IPD_METRES, NOSE_DEPTH_M, FOCAL_RATIO,
    FALLBACK_FACE_W_M, FALLBACK_FACE_FILL,
    SYMMETRY_THRESH, SKIN_RATIO_THRESH,
)
from facetrack.detector  import FaceDetectorCNN, DETECTOR_PATCH_SIZE
from facetrack.landmarks import LandmarkCNN, LANDMARK_INFER_TF
from facetrack.filters   import (
    plausible_face_geometry,
    symmetry_score,
    skin_ratio,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor-based sliding-window patch extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_patches(frame_t: torch.Tensor, scales: list, stride_ratio: float,
                     patch_size: int, roi=None):
    """
    Build the full batch of sliding-window crops in GPU-friendly form.

    Instead of per-patch PIL resize, we slice the frame tensor directly and
    batch-resize with F.interpolate — 10–20× faster.

    Args
        frame_t       : [3, H, W] float tensor, normalised to [-1, 1].
        scales        : list of scales ∈ (0, 1]; window_size = scale × min(H, W).
        stride_ratio  : stride = window_size × stride_ratio.
        patch_size    : model input size (e.g. 64).
        roi           : optional (x1, y1, x2, y2) in frame-tensor coords. When
                        set, only windows whose CENTERS fall inside the ROI
                        are kept. Used for ROI-constrained detection once a
                        track is locked.

    Returns
        patches : [N, 3, patch_size, patch_size] on the same device as frame_t.
        meta    : list of (x1, y1, x2, y2) pixel coords for each patch.
    """
    _, H, W     = frame_t.shape
    batches, metas = [], []

    for scale in scales:
        wsz = int(min(H, W) * scale)
        if wsz < 16:
            continue
        stride = max(int(wsz * stride_ratio), 4)

        crops, scale_metas = [], []
        for y in range(0, H - wsz + 1, stride):
            for x in range(0, W - wsz + 1, stride):
                if roi is not None:
                    cx = x + wsz // 2
                    cy = y + wsz // 2
                    rx1, ry1, rx2, ry2 = roi
                    if not (rx1 <= cx < rx2 and ry1 <= cy < ry2):
                        continue
                crops.append(frame_t[:, y:y + wsz, x:x + wsz])
                scale_metas.append((x, y, x + wsz, y + wsz))
        if not crops:
            continue

        batch = torch.stack(crops)
        batch = F.interpolate(batch, size=(patch_size, patch_size),
                              mode='bilinear', align_corners=False)
        batches.append(batch)
        metas.extend(scale_metas)

    if not batches:
        return None, []
    return torch.cat(batches, dim=0), metas


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap → bounding boxes (top-K peaks)
# ─────────────────────────────────────────────────────────────────────────────

def scores_to_boxes(meta: list, scores: list, H: int, W: int,
                    max_faces: int = None) -> list:
    """
    Convert per-patch sigmoid scores into up to `max_faces` bounding boxes.

    Algorithm
        1. Paint max score per pixel into a heatmap.
        2. For each window that fires, remember the best-scoring window size.
        3. Gaussian-blur lightly; find top-K local maxima by iterative argmax
           + local suppression.

    `max_faces` defaults to the DET_MAX_FACES config value. Override to 1
    when tracking a known face (avoids the top-K picker jittering between
    multiple overlapping peaks on the same face).

    Returns
        List of (x1, y1, x2, y2, peak_score, wsz) in detection-frame coords.
    """
    if max_faces is None:
        max_faces = DET_MAX_FACES
    score_map = np.zeros((H, W), dtype=np.float32)
    size_map  = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2), s in zip(meta, scores):
        if s < DET_THRESH:
            continue
        wsz  = x2 - x1
        old  = score_map[y1:y2, x1:x2]
        improved = s > old
        if not improved.any():
            continue
        score_map[y1:y2, x1:x2] = np.where(improved, s,   old)
        size_map [y1:y2, x1:x2] = np.where(improved, wsz, size_map[y1:y2, x1:x2])

    if score_map.max() < 1e-6:
        return []

    # Gentle blur smooths noise without spreading one face's blob into another
    sigma   = max(int(min(H, W) * 0.04), 3)
    blurred = cv2.GaussianBlur(score_map, (0, 0), sigma)

    boxes   = []
    working = blurred.copy()
    for _ in range(max_faces):
        peak_idx    = int(np.argmax(working))
        cy, cx      = divmod(peak_idx, W)
        peak_score  = float(working[cy, cx])
        if peak_score < DET_THRESH:
            break

        wsz = float(size_map[cy, cx]) or float(min(H, W) * 0.25)

        hw = int(wsz * 0.48)
        hh = int(wsz * 0.55)
        x1 = max(cx - hw, 0);      y1 = max(cy - hh, 0)
        x2 = min(cx + hw, W - 1);  y2 = min(cy + hh, H - 1)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, peak_score, wsz))

        # Suppress ≈ one face width around this peak before next argmax
        r = max(int(wsz * 0.7), 10)
        working[max(cy - r, 0): min(cy + r, H),
                max(cx - r, 0): min(cx + r, W)] = 0.0

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Background detection worker
# ─────────────────────────────────────────────────────────────────────────────

class DetectionWorker:
    """
    Runs the three-stage inference pipeline in a daemon thread.

    API
        start()                — launch the worker
        stop()                 — ask it to exit after current frame
        submit(frame)          — enqueue a frame for processing
                                 (non-blocking, drops if busy)
        get_detections()       — read the latest list of detections
    """

    def __init__(self, detector: FaceDetectorCNN, device: torch.device,
                 det_thresh: float = DET_THRESH,
                 focal_px: Optional[float] = None,
                 landmark_net: Optional[LandmarkCNN] = None,
                 ipd_m: float = IPD_METRES):
        self.detector     = detector
        self.device       = device
        self.det_thresh   = det_thresh
        self.focal_px     = focal_px         # None ⇒ auto-estimate per frame
        self.landmark_net = landmark_net
        # Per-user IPD override. Default is the 0.063 m adult population
        # average; an individual's real IPD is ±3 mm away, and using the
        # user's measured IPD removes up to ~5% of distance-accuracy bias.
        self.ipd_m        = float(ipd_m)

        self._q       = queue.Queue(maxsize=1)
        self._lock    = threading.Lock()
        self._results: List[dict] = []
        self._running = False

    # ------------------------------------------------------------------ thread

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def submit(self, frame: np.ndarray, track_box=None) -> None:
        """
        Enqueue a frame for processing. `track_box` is an optional
        (x1, y1, x2, y2) of the currently-locked face in ORIGINAL frame
        coords. When provided, the sliding window runs in ROI+single-scale
        mode around the box — faster, and immune to distant-background
        false positives.
        """
        try:
            self._q.put_nowait((frame, track_box))
        except queue.Full:
            pass   # drop — the display must never block on inference

    def get_detections(self) -> List[dict]:
        with self._lock:
            return list(self._results)

    def _loop(self) -> None:
        while self._running:
            try:
                frame, track_box = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            dets = self._process(frame, track_box=track_box)
            with self._lock:
                self._results = dets

    # ------------------------------------------------------------- processing

    @torch.no_grad()
    def _process(self, frame_bgr: np.ndarray, track_box=None) -> List[dict]:
        H_orig, W_orig = frame_bgr.shape[:2]

        # ── Stage 1: detect on a small downscale ─────────────────────────────
        det_frame = cv2.resize(frame_bgr, (DET_W, DET_H))
        rgb_small = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb_small).float().permute(2, 0, 1).div(255.0)
        t = ((t - 0.5) / 0.5).to(self.device)

        # When locked: ROI around the track + single best-matching scale +
        # top-1 peak. Eliminates the cross-scale / cross-peak jitter that
        # makes the displayed box flicker at close range. When searching
        # (no track_box): full pyramid and top-K, so we can find a face
        # we don't yet know about.
        sx_d, sy_d = DET_W / W_orig, DET_H / H_orig
        if track_box is not None:
            bx1, by1, bx2, by2 = track_box
            bw = max(bx2 - bx1, 1); bh = max(by2 - by1, 1)
            # Expand 2.2x around the box for motion headroom, clamp to frame
            cx = (bx1 + bx2) // 2; cy = (by1 + by2) // 2
            rw = int(bw * 2.2); rh = int(bh * 2.2)
            roi_full = (max(cx - rw // 2, 0), max(cy - rh // 2, 0),
                        min(cx + rw // 2, W_orig - 1),
                        min(cy + rh // 2, H_orig - 1))
            roi_det  = (int(roi_full[0] * sx_d), int(roi_full[1] * sy_d),
                        int(roi_full[2] * sx_d), int(roi_full[3] * sy_d))

            # Face size in detection-frame coords — use the LARGER of
            # width / height so a tall (landmark-derived) box doesn't
            # pick a too-small window. max(bw, bh) is the clean signal.
            face_px_det = max(int(max(bw, bh) * sx_d), 1)

            # Single scale whose window size is closest to face_px_det
            best_scale = min(DET_SCALES,
                             key=lambda s: abs(int(min(DET_H, DET_W) * s) - face_px_det))
            scales_used        = [best_scale]
            max_faces_override = 1
        else:
            roi_det            = None
            scales_used        = DET_SCALES
            max_faces_override = None    # = DET_MAX_FACES default

        patches, meta = extract_patches(t, scales_used, DET_STRIDE,
                                        DETECTOR_PATCH_SIZE, roi=roi_det)
        if patches is None:
            return []

        scores: List[float] = []
        for i in range(0, len(patches), 512):
            logits = self.detector(patches[i:i + 512])
            scores.extend(torch.sigmoid(logits).cpu().tolist())

        boxes_det = scores_to_boxes(meta, scores, DET_H, DET_W,
                                    max_faces=max_faces_override)
        if not boxes_det:
            return []

        # Scale the detection-frame boxes back to the original resolution
        sx, sy = W_orig / DET_W, H_orig / DET_H
        boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), wsz)
                 for x1, y1, x2, y2, _score, wsz in boxes_det]

        # ── Stage 2 + 3: landmark + distance per candidate ───────────────────
        focal_px = self.focal_px if self.focal_px else W_orig * FOCAL_RATIO
        detections: List[dict] = []

        for x1, y1, x2, y2, wsz in boxes:
            d = self._make_detection(frame_bgr, x1, y1, x2, y2, wsz,
                                     sx, W_orig, H_orig, focal_px)
            if d is not None:
                detections.append(d)

        return detections

    # ................................................. single-candidate path

    def _make_detection(self, frame_bgr, x1, y1, x2, y2, wsz,
                        sx, W_orig, H_orig, focal_px):
        """Try to promote one detection candidate into a full detection dict."""
        # Pad by 20% around the detection box (matches landmark training dist.)
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(x1 - int(bw * 0.20), 0)
        cy1 = max(y1 - int(bh * 0.20), 0)
        cx2 = min(x2 + int(bw * 0.20), W_orig)
        cy2 = min(y2 + int(bh * 0.20), H_orig)
        lm_crop = frame_bgr[cy1:cy2, cx1:cx2]
        if lm_crop.size == 0:
            return None

        ipd_px = None
        eyes_abs = None
        lmks_abs = None
        yaw_deg  = 0.0

        if self.landmark_net is not None:
            pil    = Image.fromarray(lm_crop[:, :, ::-1])        # BGR → RGB
            cw, ch = pil.size
            t_in   = LANDMARK_INFER_TF(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out10 = self.landmark_net(t_in).cpu().squeeze(0).tolist()
            lx, ly, rx, ry, nx_, ny_, lmx, lmy, rmx, rmy = out10

            # ── Filters ──────────────────────────────────────────────────────
            if not plausible_face_geometry(lx, ly, rx, ry,
                                           nx_, ny_,
                                           lmx, lmy, rmx, rmy):
                return None

            sym = symmetry_score(lm_crop)
            skn = skin_ratio(lm_crop)
            # Require at least one of the two visual signals (loose by design)
            if (sym < SYMMETRY_THRESH) and (skn < SKIN_RATIO_THRESH):
                return None

            # ── Landmark → absolute frame coords ─────────────────────────────
            lx_abs  = cx1 + lx  * cw;  ly_abs  = cy1 + ly  * ch
            rx_abs  = cx1 + rx  * cw;  ry_abs  = cy1 + ry  * ch
            nx_abs  = cx1 + nx_ * cw;  ny_abs  = cy1 + ny_ * ch
            lmx_abs = cx1 + lmx * cw;  lmy_abs = cy1 + lmy * ch
            rmx_abs = cx1 + rmx * cw;  rmy_abs = cy1 + rmy * ch
            ipd_raw_px = float(np.hypot(lx_abs - rx_abs, ly_abs - ry_abs))
            eyes_abs   = (int(lx_abs), int(ly_abs), int(rx_abs), int(ry_abs))
            lmks_abs   = (lx_abs, ly_abs, rx_abs, ry_abs,
                          nx_abs, ny_abs,
                          lmx_abs, lmy_abs, rmx_abs, rmy_abs)

            # ── Yaw correction ───────────────────────────────────────────────
            eye_mid_x   = (lx_abs + rx_abs) * 0.5
            nose_off    = nx_abs - eye_mid_x
            tan_yaw     = (nose_off / max(ipd_raw_px, 1.0)) / (NOSE_DEPTH_M / IPD_METRES)
            tan_yaw     = max(min(tan_yaw, 2.75), -2.75)       # clamp ≤ ±70°
            cos_yaw     = 1.0 / (1.0 + tan_yaw * tan_yaw) ** 0.5
            yaw_deg     = float(np.degrees(np.arctan(tan_yaw)))
            ipd_px      = ipd_raw_px / max(cos_yaw, 0.35)

            # ── Tight head box derived from landmarks ────────────────────────
            eye_mid_y    = (ly_abs + ry_abs) * 0.5
            mouth_mid_y  = (lmy_abs + rmy_abs) * 0.5
            eye_to_mouth = max(mouth_mid_y - eye_mid_y, 10.0)
            head_w_px    = max(2.30 * ipd_px, 40.0)
            face_top     = eye_mid_y  - 1.10 * eye_to_mouth
            face_bot     = mouth_mid_y + 0.70 * eye_to_mouth
            face_cx      = eye_mid_x

            tx1 = max(int(face_cx - head_w_px * 0.5), 0)
            tx2 = min(int(face_cx + head_w_px * 0.5), W_orig - 1)
            ty1 = max(int(face_top), 0)
            ty2 = min(int(face_bot), H_orig - 1)
            if tx2 > tx1 and ty2 > ty1:
                x1, y1, x2, y2 = tx1, ty1, tx2, ty2

        # ── Distance from IPD (or fallback geometric formula) ────────────────
        # Uses self.ipd_m (per-user configurable) rather than the population
        # average IPD_METRES — lets users remove ~±5% bias by passing --ipd.
        if ipd_px and ipd_px > 4:
            dist_m = (self.ipd_m * focal_px) / ipd_px
        else:
            face_px = max(wsz * sx * FALLBACK_FACE_FILL, 1.0)
            dist_m  = (FALLBACK_FACE_W_M * focal_px) / face_px
        dist_m = round(max(0.3, min(dist_m, 10.0)), 2)

        return {
            'box':      (x1, y1, x2, y2),
            'name':     'FACE',
            'conf':     1.0,
            'dist':     dist_m,
            'ipd_px':   ipd_px if ipd_px else 0.0,
            'yaw':      yaw_deg,
            'eyes':     eyes_abs,
            'lmks_abs': lmks_abs,
            'wsz':      wsz * sx,
        }
