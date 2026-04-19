# Face Tracking + Distance Estimation

Real-time face tracking and monocular distance estimation from a single webcam — custom detector, custom 5-point landmark regressor, sticky single-face tracker, geometric distance from inter-pupillary distance. Runs at 30+ FPS on a modest GPU.

```bash
python run.py --camera 0
```

- **Green box** locks onto your face and glides with your motion.
- **Distance in metres** shown live, yaw-corrected so it stays accurate when you turn your head.
- **Press `C`** once at exactly 1 m from the camera to calibrate focal length → ~±3 % accuracy from then on.

---

## How it works — per frame, in plain English

```
┌─ 1 ── GRAB ─────────────────────────────────────────────────────┐
│  Pull a 1280×720 frame from the webcam. Downscale a copy to     │
│  320×240 for detection (the detector doesn't need full res).    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 2 ── FIND THE FACE ────────────────────────────────────────────┐
│                                                                 │
│  SEARCHING mode (no face locked yet):                           │
│    Slide a window across the frame at 6 scales (big-close-up    │
│    → tiny-far). Classify every window with the detector CNN.    │
│    Keep the top 3 peaks above threshold.                        │
│                                                                 │
│  TRACKING mode (face already locked):                           │
│    Scan only a small box around the current track, and only     │
│    at the one scale that matches the face size. ~20 ms/frame    │
│    and can't false-fire on distant background.                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 3 ── LANDMARKS ────────────────────────────────────────────────┐
│  For each candidate, pad the crop by 20% and run the landmark   │
│  CNN. Output: 5 points — left eye, right eye, nose, left mouth, │
│  right mouth — normalised to [0,1] inside the crop.             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 4 ── REJECT IMPOSTERS ─────────────────────────────────────────┐
│  Three cheap filters; all three must pass:                      │
│    • Geometry — eyes horizontal, nose between them, mouth below │
│    • Symmetry — flipped crop still looks similar (NCC)          │
│    • Skin tone — enough skin-colour pixels in HSV               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 5 ── MEASURE DISTANCE ─────────────────────────────────────────┐
│  yaw  = angle derived from how far the nose has shifted off     │
│         the eye midline                                         │
│  ipd  = pixel distance between eye centres, un-foreshortened    │
│         by cos(yaw)                                             │
│  dist = (0.063 m × focal_px) / ipd                              │
│                                                                 │
│  0.063 m is the average adult inter-pupillary distance.         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 6 ── TRACK + SMOOTH ───────────────────────────────────────────┐
│  The SingleFaceTracker keeps the box stable across frames:      │
│    • Lock only after 2 consecutive confirming frames            │
│    • Match new detections by nearest-centroid                   │
│    • If a detection appears far from the lock for 2 frames,     │
│      switch to it (re-lock) instead of coasting on the old one  │
│    • EMA-smooth the box + 7-frame rolling median on distance    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         Display frame
```

**Why IPD instead of face width?** Inter-pupillary distance is ±3 mm across adults; face width is ±10 mm. IPD cuts cross-user distance error roughly in half and is robust to expression, hair, facial hair, glasses.

---

## Models

| Model             | File                     | Params  | Input         | Output                                 |
| ----------------- | ------------------------ | ------- | ------------- | -------------------------------------- |
| `FaceDetectorCNN` | `facetrack/detector.py`  | ~3.46 M | 96 × 96 RGB   | scalar logit — face vs background      |
| `LandmarkCNN`     | `facetrack/landmarks.py` | ~150 k  | 64 × 64 RGB   | 10 floats — 5 (x, y) landmark coords   |

Both are **custom** architectures. The detector is a 5-block conv backbone (Conv → BatchNorm → SiLU, MaxPool between blocks) with double conv per block, channels [32, 64, 128, 256, 384], and a small MLP head. The landmark net is a narrower 4-block variant of the same pattern. Detector and landmark patch sizes are decoupled, so you can retrain the detector at a different resolution without invalidating the landmark checkpoint.

### How we got here

| Version | Params | Input | Key change                                                                 |
| ------- | ------ | ----- | -------------------------------------------------------------------------- |
| v1      | ~200 k | 64×64 | LFW + CelebA only; overfit the celebrity-frontal distribution              |
| v2      | ~1.0 M | 96×96 | Heavier augmentation + UTKFace                                             |
| v4      | ~1.9 M | 96×96 | + FairFace (capped)                                                        |
| **v5**  | **~3.5 M** | **96×96** | **Full uncapped data (349 k positives); lazy-loading training pipeline** |

The landmark net was also retrained (v2) with horizontal-flip augmentation (labels mirrored in sync) + heavier photometric aug.

---

## Datasets

### Used for training

| Dataset            | Role                                                  | Size       | How to obtain                                                  |
| ------------------ | ----------------------------------------------------- | ---------- | -------------------------------------------------------------- |
| **LFW**            | Diverse face positives                                | 13 k       | `python download_datasets.py`                                  |
| **CelebA aligned** | Face positives + 5-point landmarks                    | 200 k      | `kaggle datasets download -d jessicali9530/celeba-dataset`     |
| **WIDER FACE**     | Face positives at varied scales, occlusions, poses    | ~390 k     | `python download_datasets.py --wider`                          |
| **UTKFace**        | Broad demographic spread (age / gender / ethnicity)   | 23 k       | `python download_datasets.py --utk`                            |
| **FairFace**       | Demographically balanced (7 race groups × age × sex)  | 108 k      | `python download_datasets.py --fairface` (manual instructions) |
| **Hard negatives** | False positives mined by Phase-1 detector             | ≤ 6 k      | Auto-generated during Phase-2 training                         |

Training degrades gracefully if any dataset is missing — you just lose the corresponding generalisation boost.

---

## Aggregation & voting

Temporal aggregation is used; cross-model ensembling is **not**.

| Layer                 | What happens                                                 | Kind                                     |
| --------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| Distance readings     | 7-frame rolling median                                       | Temporal — robust to spikes              |
| Bounding box          | EMA with `TRACK_BOX_ALPHA`                                   | Temporal — smooths jitter                |
| Track lock            | 2 consecutive confirming frames required                     | Temporal — rejects single-frame FPs      |
| Filter stage          | Geometry ∧ Symmetry ∧ Skin — all three must pass             | Unanimous AND, not weighted voting       |
| Detection peaks       | Top-K (searching) or top-1 + single scale (tracking)         | Top-1 once locked kills cross-peak race  |
| Model ensemble / TTA  | —                                                            | **None** — single detector, single pass  |

---

## Known limitations

- **Monocular distance is calibration-bound.** ±20 % out-of-the-box, ±3–5 % *after* pressing `C` at 1 m. No way to drive error to zero without a reference — fundamental to monocular depth from an unknown camera.
- **IPD varies ≈ ±3 mm across adults** → residual per-person error floor.
- **Single-face only.** By design — no multi-person support.
- **`DET_THRESH` is environment-dependent** and may need tuning (see *Tuning* below).
- **Top-K heatmap peaks, not NMS / connected components** — can miss overlapping faces.

---

## Quick start

### Inference

```bash
pip install -r requirements.txt
python run.py --camera 0
```

The shipped `checkpoints/face_detector.pth` is the trained v5 model and `checkpoints/landmark_net.pth` is landmark v2 — no training needed to run.

Controls while running:

| Key       | Action                                    |
| --------- | ----------------------------------------- |
| `Q` / Esc | Quit                                      |
| `S`       | Save a screenshot (`screenshot_NNNN.png`) |
| `C`       | Calibrate focal — stand 1.0 m away first  |

### CLI flags

| Flag           | Default                           | Purpose                                                              |
| -------------- | --------------------------------- | -------------------------------------------------------------------- |
| `--camera`     | `0`                               | Camera index (use `1` for external webcam / phone)                   |
| `--focal`      | `0` (= auto-estimate)             | Override focal length in pixels. Calibrate once with `C`, paste here. |
| `--det_thresh` | `0.40`                            | Detector sigmoid threshold. Lower = more sensitive.                  |
| `--det_ckpt`   | `checkpoints/face_detector.pth`   | Detector weights                                                     |
| `--lmk_ckpt`   | `checkpoints/landmark_net.pth`    | Landmark weights                                                     |

---

## Training

### Download datasets (one-time setup)

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# LFW + WIDER + UTKFace in one go; FairFace prints manual instructions
python download_datasets.py --all

# CelebA (separate — big)
kaggle datasets download -d jessicali9530/celeba-dataset -p datasets/tmp_celeba --unzip
```

### Face detector — two-phase training

```bash
python train_detector.py
# or, if Phase 1 already ran and you just want to redo mining + Phase 2:
python train_detector.py --resume-from-phase1
```

Phase 1 bootstraps on positives + random negatives. Phase 2 mines the Phase-1 model's own false positives as hard negatives and re-trains from scratch. Samples are stored as file-paths (lazy loaded in `__getitem__`) so RAM scales with sample *count*, not decoded pixel data — no dataset size caps needed.

### Landmark regressor

```bash
python train_landmarks.py
```

30 epochs with horizontal-flip + photometric augmentation. Reaches ≈ 0.0037 normalised error (~0.24 px on a 64 × 64 crop).

---

## Accuracy

| Setup                              | Distance error |
| ---------------------------------- | -------------- |
| Out-of-box, default focal          | ±20 %          |
| After one `C` calibration at 1 m   | ±3 – 5 %       |
| Friend's laptop, no calibration    | ±25 %          |
| Friend's laptop, one calibration   | ±3 – 5 %       |

Remaining error is dominated by focal-length uncertainty (camera-specific) and secondarily IPD variation (anatomy-specific).

---

## Tuning

| Symptom                           | Fix                                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Box flickers on / off             | Already damped by tracker EMA and the 2-frame lock confirm. If still flickering, lower `DET_THRESH`. |
| Box drifts to walls or furniture  | Raise `DET_THRESH` to 0.55+. Filters are the second line of defence.                                 |
| Face never detected               | Lower `DET_THRESH` to 0.30. Also press `C` to calibrate focal.                                       |
| Distance off                      | Press `C` at 1.0 m. Or pass `--focal N`.                                                             |

All tunables live in `facetrack/config.py`.

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── run.py                         Real-time inference entry point
├── train_detector.py              Trains FaceDetectorCNN (two-phase)
├── train_landmarks.py             Trains LandmarkCNN
├── download_datasets.py           LFW / WIDER / UTKFace / FairFace
│
├── facetrack/                     Library package
│   ├── __init__.py
│   ├── config.py                  All tunable constants
│   ├── detector.py                FaceDetectorCNN architecture + transforms
│   ├── landmarks.py               LandmarkCNN architecture + transforms
│   ├── filters.py                 Geometric + symmetry + skin FP filters
│   ├── tracker.py                 SingleFaceTracker (sticky 1-face state machine)
│   └── pipeline.py                Sliding window + DetectionWorker thread
│
└── checkpoints/
    ├── face_detector.pth          Trained detector
    └── landmark_net.pth           Trained landmark regressor
```

---

## Design decisions

- **Everything from scratch.** Build and train every piece end-to-end on open face datasets.
- **ROI + single-scale when locked.** Once the tracker has a face, the detector scans only a tight box around it at one scale — cuts compute ~15× and eliminates the distant-background false-positive class.
- **EMA smoothing over Kalman.** Two multiplies per coord per frame vs. dozens, and head motion isn't complex enough to need a dynamical model.
- **Single-face tracker.** The classic "box jumps to a wall" failure comes from multi-track logic treating each detection independently. A single sticky track is simpler and more robust for the one-person use case.
- **Ship the trained checkpoints.** Small (~16 MB total). `git clone` + `python run.py` should just work.

---

```
┌────────────────────────────────────────────┐
│ C:\blitz>                                  │
│                                            │
│   ██████╗ ██╗     ██╗████████╗███████╗     │
│   ██╔══██╗██║     ██║╚══██╔══╝╚══███╔╝     │
│   ██████╔╝██║     ██║   ██║     ███╔╝      │
│   ██╔══██╗██║     ██║   ██║    ███╔╝       │
│   ██████╔╝███████╗██║   ██║   ███████╗     │
│   ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚══════╝     │
│                                            │
│ C:\blitz> _                                │
└────────────────────────────────────────────┘
```
