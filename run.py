"""
run.py  —  real-time face tracking + distance estimation
=========================================================

Entry point.  Opens the camera, runs the three-stage pipeline
(detect → landmark → distance), overlays a box + distance, displays live.

Usage
─────
    python run.py                       # camera 0, auto-calibrated focal
    python run.py --camera 1            # external webcam / phone camera
    python run.py --focal 850           # use a previously measured focal
    python run.py --det_thresh 0.30     # more sensitive detector

Controls
────────
    Q / Esc   quit
    S         save a PNG screenshot
    C         calibrate focal (stand exactly 1.0 m from camera first)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch

from facetrack              import SingleFaceTracker, DetectionWorker
from facetrack.detector     import FaceDetectorCNN
from facetrack.landmarks    import LandmarkCNN
from facetrack.config       import (
    DET_THRESH, IPD_METRES, FOCAL_RATIO,
    FALLBACK_FACE_W_M, FALLBACK_FACE_FILL,
    PALETTE,
)

FONT = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_detector(path: str, device: torch.device) -> FaceDetectorCNN:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = FaceDetectorCNN().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def load_landmark_net(path: str, device: torch.device):
    if not Path(path).exists():
        return None
    try:
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = LandmarkCNN().to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        err = ckpt.get('val_err', float('nan'))
        print(f"Landmark net  : {path}  (val err = {err:.4f} ≈ {err*64:.1f}px on 64×64)")
        return model
    except Exception as e:
        print(f"Landmark net  : failed to load ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_label(frame, box, dist_m: float) -> None:
    x1, y1, x2, y2 = box
    colour = PALETTE['box']
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    txt   = f"{dist_m:.2f} m"
    pad, lh = 4, 18
    bg_y1   = max(y1 - lh - pad * 2, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x2, y1), colour, -1)
    cv2.putText(frame, txt, (x1 + pad, bg_y1 + pad + lh - 2),
                FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(camera_idx: int   = 0,
        det_ckpt:   str   = 'checkpoints/face_detector.pth',
        lmk_ckpt:   str   = 'checkpoints/landmark_net.pth',
        det_thresh: float = DET_THRESH,
        focal_px:   float = 0.0) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device        : {device}")

    print(f"Detector      : {det_ckpt}")
    detector = load_detector(det_ckpt, device)
    print(f"Det threshold : {det_thresh}")

    landmark_net = load_landmark_net(lmk_ckpt, device)
    if landmark_net is None:
        print("Landmark net  : not found — falling back to face-width geometry")

    worker = DetectionWorker(
        detector, device, det_thresh,
        focal_px     = focal_px if focal_px > 0 else None,
        landmark_net = landmark_net,
    )
    worker.start()

    tracker = SingleFaceTracker()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        worker.stop()
        raise RuntimeError(f"Cannot open camera {camera_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Running — Q/Esc quit, S screenshot, C calibrate @ 1.0 m")

    fps_t, fps_n, fps_val = time.time(), 0, 0.0
    shot_n   = 0
    cal_msg  = ''
    cal_t    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        worker.submit(frame)
        tracks = tracker.update(worker.get_detections())

        for tr in tracks:
            draw_label(frame, tr['box'], tr['dist'])

        # FPS + status overlay
        fps_n += 1
        now    = time.time()
        if now - fps_t >= 0.5:
            fps_val = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now

        h = frame.shape[0]
        focal_label = f"{worker.focal_px:.0f}px" if worker.focal_px else "auto"
        cv2.putText(frame, f"FPS: {fps_val:.0f}  focal: {focal_label}",
                    (10, h - 12), FONT, 0.5, PALETTE['fps'], 1, cv2.LINE_AA)
        cv2.putText(frame, "Q/Esc quit   S screenshot   C calibrate@1m",
                    (10, h - 30), FONT, 0.4, PALETTE['fps'], 1, cv2.LINE_AA)
        if cal_msg and (time.time() - cal_t) < 3.0:
            cv2.putText(frame, cal_msg, (10, 60), FONT, 0.7, PALETTE['box'],
                        2, cv2.LINE_AA)

        cv2.imshow('Face Tracking + Distance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            fname = f"screenshot_{shot_n:04d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            shot_n += 1
        if key == ord('c'):
            cal_msg, cal_t = _calibrate(worker, tracker)

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration  —  user stands 1.0 m away, presses C
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate(worker: DetectionWorker, tracker: SingleFaceTracker):
    """Measure the current IPD (or face-width fallback) and back-solve focal."""
    tracks = tracker.update([]) if tracker.track else []
    # Prefer current track readings; fall back to latest worker detection
    source = tracks[0] if tracks else (worker.get_detections() + [{}])[0]

    ipd_px   = source.get('ipd_px', 0)
    wsz_orig = source.get('wsz',    0)

    known_dist = 1.0
    new_focal  = 0.0
    method     = ''
    if ipd_px and ipd_px > 4:
        new_focal = (ipd_px * known_dist) / IPD_METRES
        method    = 'IPD'
    elif wsz_orig > 0:
        new_focal = (wsz_orig * FALLBACK_FACE_FILL * known_dist) / FALLBACK_FACE_W_M
        method    = 'face-width'

    if new_focal <= 0:
        return 'No face detected — move closer and retry', time.time()

    worker.focal_px = new_focal
    msg = f"Calibrated ({method}): focal = {new_focal:.0f} px"
    print(msg)
    print(f"  Tip: --focal {new_focal:.0f}  to skip calibration next time")
    return msg, time.time()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description='Real-time face tracking + distance')
    p.add_argument('--camera',     type=int,   default=0)
    p.add_argument('--det_ckpt',   type=str,   default='checkpoints/face_detector.pth')
    p.add_argument('--lmk_ckpt',   type=str,   default='checkpoints/landmark_net.pth')
    p.add_argument('--det_thresh', type=float, default=DET_THRESH,
                   help='Detector sigmoid threshold (default 0.40)')
    p.add_argument('--focal',      type=float, default=0.0,
                   help='Camera focal (px). 0 = auto-estimate. Press C at 1.0m to calibrate.')
    args = p.parse_args()

    run(camera_idx = args.camera,
        det_ckpt   = args.det_ckpt,
        lmk_ckpt   = args.lmk_ckpt,
        det_thresh = args.det_thresh,
        focal_px   = args.focal)


if __name__ == '__main__':
    main()
