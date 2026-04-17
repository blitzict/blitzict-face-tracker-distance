# Face Tracking + Distance Estimation

Real-time face tracking and distance estimation from a single webcam, built from scratch — custom face detector, custom landmark regressor, sticky single-face tracker, geometric distance from inter-pupillary distance. Runs at 30+ FPS on a modest GPU.

```
python run.py --camera 0
```

- **Green box** locks onto your face with smooth temporal tracking.
- **Distance in metres** shown live, yaw-corrected so it stays accurate when you turn your head.
- **Press `C`** once at exactly 1 m from the camera to calibrate focal length → ~±3 % accuracy from then on.

No MediaPipe, no Haar cascade at inference, no pre-trained black boxes.

---

## Pipeline

```
┌──────────────────┐
│ Camera frame     │ 1280 × 720 BGR
└────────┬─────────┘
         ▼
┌────────────────────────┐
│ Stage 1  FaceDetectorCNN                                              │
│   Sliding window on a 320 × 240 downscale at 4 scales.                │
│   Top-K heatmap peaks → candidate boxes.                              │
│   (~200 k parameters, binary face / background classifier)            │
└────────┬──────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────┐
│ Stage 2  LandmarkCNN                                                  │
│   Pad candidate crop by 20 %, run CNN → 5 landmarks:                  │
│     (left eye, right eye, nose, left mouth, right mouth)              │
│   (~150 k parameters)                                                 │
└────────┬──────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────┐
│ Stage 3  Filters       │ Reject non-faces:
│   • Geometric plausibility of the 5 landmarks                         │
│   • Bilateral symmetry  (flipped-crop NCC)                            │
│   • Skin-tone ratio     (HSV)                                         │
└────────┬──────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────┐
│ Stage 4  Distance      │
│   yaw   = estimated from nose offset vs eye midline                   │
│   ipd   = measured IPD in pixels, un-foreshortened by cos(yaw)        │
│   dist  = (0.063 m × focal_px) / ipd                                  │
└────────┬──────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────┐
│ Stage 5  SingleFaceTracker                                            │
│   • Requires LOCK_CONFIRM consecutive confirming frames to lock       │
│   • Nearest-detection match only (ignores far detections)             │
│   • Temporal landmark-stability gate                                  │
│   • EMA box smoothing + median-filtered distance                      │
└────────┬──────────────────────────────────────────────────────────────┘
         ▼
      Display
```

**Why IPD instead of face width?**
Inter-pupillary distance is ±3 mm across adults; face width is ±10 mm. IPD cuts cross-user distance error roughly in half and is robust to expression, hair, facial hair, glasses.

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── run.py                         Real-time inference entry point
├── train_detector.py              Trains FaceDetectorCNN
├── train_landmarks.py             Trains LandmarkCNN
├── download_datasets.py           LFW + WIDER FACE downloader
│
├── facetrack/                     Library package
│   ├── __init__.py
│   ├── config.py                  All tunable constants
│   ├── detector.py                FaceDetectorCNN architecture + transforms
│   ├── landmarks.py               LandmarkCNN architecture + transforms
│   ├── filters.py                 Geometric + symmetry + skin false-positive filters
│   ├── tracker.py                 SingleFaceTracker (sticky 1-face state machine)
│   └── pipeline.py                Sliding window + heatmap + DetectionWorker thread
│
└── checkpoints/
    ├── face_detector.pth          Trained detector
    └── landmark_net.pth           Trained landmark regressor
```

---

## Quick start

### Inference (no training required — checkpoints are shipped in the repo)

```bash
pip install -r requirements.txt
python run.py --camera 0
```

Controls while running:

| Key       | Action                              |
| --------- | ----------------------------------- |
| `Q` / Esc | Quit                                |
| `S`       | Save a screenshot (`screenshot_NNNN.png`) |
| `C`       | Calibrate focal — stand 1.0 m away first |

### CLI flags

| Flag             | Default                      | Purpose                                                            |
| ---------------- | ---------------------------- | ------------------------------------------------------------------ |
| `--camera`       | `0`                          | Camera index (use `1` for external webcam / phone)                 |
| `--focal`        | `0` (= auto-estimate)        | Override focal length in pixels. Calibrate once with `C`, paste here. |
| `--det_thresh`   | `0.40`                       | Detector sigmoid threshold. Lower = more sensitive.               |
| `--det_ckpt`     | `checkpoints/face_detector.pth` | Detector weights                                              |
| `--lmk_ckpt`     | `checkpoints/landmark_net.pth`  | Landmark weights                                              |

---

## Models

| Model             | File                     | Params  | Input       | Output                                 |
| ----------------- | ------------------------ | ------- | ----------- | -------------------------------------- |
| `FaceDetectorCNN` | `facetrack/detector.py`  | ~200 k  | 64 × 64 RGB | scalar logit — face vs background      |
| `LandmarkCNN`     | `facetrack/landmarks.py` | ~150 k  | 64 × 64 RGB | 10 floats — 5 (x, y) landmark coords   |

Both share the same 4-block conv backbone. Trained checkpoints are in `checkpoints/`.

---

## Datasets

### Used for training

| Dataset            | Role                                                  | Size   | How to obtain                                                          |
| ------------------ | ----------------------------------------------------- | ------ | ---------------------------------------------------------------------- |
| **LFW**            | Diverse face positives                                | 13 k   | `python download_datasets.py`                                          |
| **CelebA aligned** | Face positives + 5-point landmarks                    | 200 k  | `kaggle datasets download -d jessicali9530/celeba-dataset`             |
| **WIDER-selected** | Face positives at varied scales + 68-point landmarks  | 4.3 k imgs | `kaggle datasets download -d alirezakay/wider-selected`            |
| **Hard negatives** | False positives mined by Phase-1 detector             | ≤ 6 k  | Auto-generated during Phase-2 training                                 |

### Explicitly not used — and why

| Not used                   | Reason                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| BIWI Kinect                | All subjects at a fixed ~0.9 m distance — no variety for distance training.                                  |
| HRRFaceD / LRRFaceD        | "Range" in the dataset name refers to wavelength, not distance. No distance annotations.                     |
| HPE-360                    | Fisheye-distorted — distortion would actively hurt the detector.                                             |
| FFHQ                       | Over-curated high-res portraits. After 64 × 64 downsample, adds nothing that LFW / CelebA doesn't already give. |
| Scale-augmented synthetics | Tried, made the detector fire on plain backgrounds. Removed.                                                 |

---

## Training

### Download datasets (one-time setup)

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python download_datasets.py
kaggle datasets download -d alirezakay/wider-selected -p datasets/tmp_wider --unzip
kaggle datasets download -d jessicali9530/celeba-dataset -p datasets/tmp_celeba --unzip
```

### Train the face detector

```bash
python train_detector.py
```

Two-phase training. Phase 1 bootstraps on positives + random negatives for 40 epochs. Phase 2 mines the phase-1 model's own false positives as "hard negatives" and re-trains from scratch for another 40 epochs. Reaches P = R = 1.00 on the held-out val split.

### Train the landmark regressor

```bash
python train_landmarks.py
```

30 epochs, ~25 s each on an RTX-class GPU. Reaches ≈ 0.003 normalised error — about **0.2 pixels on a 64 × 64 crop**.

---

## Tuning

| Symptom                                | Fix                                                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Box flickers on / off                  | Already damped by tracker EMA + 45-frame persistence. If it's still flickering, lower `DET_THRESH` to 0.30.  |
| Box drifts to walls or furniture       | Raise `DET_THRESH` to 0.55+. The filters are the second line of defence.                                     |
| Face never detected                    | Lower `DET_THRESH` to 0.30. Also check `FOCAL_RATIO` / press `C` to calibrate.                               |
| Distance off                           | Press `C` at 1.0 m. Or pass `--focal N`.                                                                     |

All tunables live in `facetrack/config.py`.

---

## Honest accuracy

| Setup                                              | Distance error |
| -------------------------------------------------- | -------------- |
| Out-of-box, default focal                          | ±20 %          |
| After one `C` calibration at 1 m                   | ±3 – 5 %       |
| Friend's laptop, no calibration                    | ±25 %          |
| Friend's laptop, one calibration                   | ±3 – 5 %       |

Remaining error is dominated by focal-length uncertainty (camera-specific) and secondarily IPD variation (anatomy-specific). Monocular depth from an unknown camera is fundamentally a calibration problem — there is no way to drive error to zero without any reference.

---

## Design decisions

- **No MediaPipe / Haar cascade at inference.** The whole point was to build every piece from scratch.
- **Top-K peaks instead of connected components.** Simpler; the landmark plausibility + filters do the false-positive rejection better than CC regions ever did.
- **EMA smoothing instead of Kalman.** Two multiplies per coord per frame versus dozens, and the motion isn't complex enough to need a dynamical model.
- **Single-face tracker.** The classic "box jumps to a wall" failure comes from multi-track logic treating each detection independently. Forcing a single sticky track is a simpler, more robust design for the one-person use case.
- **Shipping the trained checkpoints.** They're small (~5 MB total) — cloning and running `python run.py` should just work. Training is an optional rebuild.
