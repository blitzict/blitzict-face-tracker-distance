# Face Tracking + Distance Estimation

Real-time face tracking and monocular distance estimation from a single webcam, built from scratch — custom face detector, custom 5-point landmark regressor, sticky single-face tracker, geometric distance from inter-pupillary distance. Runs at 30+ FPS on a modest GPU.

```bash
python run.py --camera 0
```

- **Green box** locks onto your face with smooth temporal tracking.
- **Distance in metres** shown live, yaw-corrected so it stays accurate when you turn your head.
- **Press `C`** once at exactly 1 m from the camera to calibrate focal length → ~±3 % accuracy from then on.

Everything is built from scratch — custom detector, custom landmark regressor, trained end-to-end on open face datasets.

### What this does

Face **detection** + 5-point **landmark regression** + monocular **distance estimation**, in a single real-time pipeline.

---

## Pipeline

```
┌──────────────────┐
│ Camera frame     │ 1280 × 720 BGR
└────────┬─────────┘
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1  FaceDetectorCNN                                               │
│   Sliding window on a 320 × 240 downscale at 4 scales.                 │
│   Top-K heatmap peaks → candidate boxes.                               │
│   (~200 k params, binary face / background classifier)                 │
└────────┬───────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2  LandmarkCNN                                                   │
│   Pad candidate crop by 20 %, run CNN → 5 landmarks:                   │
│     (left eye, right eye, nose, left mouth, right mouth)               │
│   (~150 k params)                                                      │
└────────┬───────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3  Filters  (unanimous AND gate — all three must pass)           │
│   • Geometric plausibility of the 5 landmarks                          │
│   • Bilateral symmetry (flipped-crop NCC)                              │
│   • Skin-tone ratio (HSV)                                              │
└────────┬───────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4  Distance                                                      │
│   yaw   = estimated from nose offset vs eye midline                    │
│   ipd   = measured IPD in pixels, un-foreshortened by cos(yaw)         │
│   dist  = (0.063 m × focal_px) / ipd                                   │
└────────┬───────────────────────────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 5  SingleFaceTracker                                             │
│   • LOCK_CONFIRM consecutive confirming frames required to lock        │
│   • Nearest-detection match only (ignores far detections)              │
│   • Temporal landmark-stability gate                                   │
│   • EMA box smoothing + 7-frame median on distance                     │
└────────┬───────────────────────────────────────────────────────────────┘
         ▼
      Display
```

**Why IPD instead of face width?** Inter-pupillary distance is ±3 mm across adults; face width is ±10 mm. IPD cuts cross-user distance error roughly in half and is robust to expression, hair, facial hair, glasses.

---

## Models

| Model             | File                     | Params  | Input         | Output                                 |
| ----------------- | ------------------------ | ------- | ------------- | -------------------------------------- |
| `FaceDetectorCNN` | `facetrack/detector.py`  | ~1.04 M | 96 × 96 RGB   | scalar logit — face vs background      |
| `LandmarkCNN`     | `facetrack/landmarks.py` | ~150 k  | 64 × 64 RGB   | 10 floats — 5 (x, y) landmark coords   |

Both are **custom** architectures. The detector is a 5-block conv backbone (Conv → BatchNorm → SiLU, MaxPool between blocks) with double conv per block and a small MLP head; the landmark net is a narrower 4-block variant of the same pattern, trained at 64×64. Detector and landmark patch sizes are decoupled: you can retrain the detector at a different resolution without invalidating the landmark checkpoint.

> **v2 upgrade note.** An earlier v1 of the detector was ~200 k params at 64×64. It overfit to the LFW + CelebA distribution and generalised poorly to arbitrary webcams / people. The v2 above increases capacity 5×, moves to 96×96, and trains with much heavier augmentation + two demographic-balanced datasets (UTKFace, FairFace). The shipped v1 checkpoint is not compatible with the v2 architecture — retrain via `python train_detector.py`.

---

## Datasets

### Used for training

| Dataset            | Role                                                  | Size       | How to obtain                                                |
| ------------------ | ----------------------------------------------------- | ---------- | ------------------------------------------------------------ |
| **LFW**            | Diverse face positives                                | 13 k       | `python download_datasets.py`                                |
| **CelebA aligned** | Face positives + 5-point landmarks                    | 200 k      | `kaggle datasets download -d jessicali9530/celeba-dataset`   |
| **WIDER FACE**     | Face positives at varied scales, occlusions, poses    | ~390 k     | `python download_datasets.py --wider`                        |
| **UTKFace**        | Broad demographic spread (age / gender / ethnicity)   | 23 k       | `python download_datasets.py --utk`                          |
| **FairFace**       | Demographically balanced (7 race groups × age × sex)  | 108 k      | `python download_datasets.py --fairface` (manual instructions) |
| **Hard negatives** | False positives mined by Phase-1 detector             | ≤ 6 k      | Auto-generated during Phase-2 training                       |

UTKFace and FairFace were added in the v2 upgrade specifically to defeat the demographic bias baked into LFW + CelebA. Training will degrade gracefully if either (or both) are missing — you just lose the corresponding generalisation boost.

### Explicitly not used — and why

| Not used                   | Reason                                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| BIWI Kinect                | All subjects at a fixed ~0.9 m distance — no variety for distance training.                                     |
| HRRFaceD / LRRFaceD        | "Range" in the dataset name refers to wavelength, not distance. No distance annotations.                        |
| HPE-360                    | Fisheye-distorted — distortion would actively hurt the detector.                                                |
| FFHQ                       | Over-curated high-res portraits. After 64 × 64 downsample, adds nothing LFW / CelebA doesn't already give.      |
| Scale-augmented synthetics | Tried, made the detector fire on plain backgrounds. Removed.                                                    |

---

## Aggregation & voting

Temporal aggregation is used; cross-model ensembling is **not**.

| Layer                  | What happens                                                     | What kind of aggregation   |
| ---------------------- | ---------------------------------------------------------------- | -------------------------- |
| Distance readings      | 7-frame rolling median                                           | Temporal — robust to spikes |
| Bounding box           | EMA with `TRACK_BOX_ALPHA`                                       | Temporal — smooths jitter   |
| Track lock             | `LOCK_CONFIRM = 2` consecutive confirming frames required        | Temporal — rejects single-frame FPs |
| Filter stage           | Geometry ∧ Symmetry ∧ Skin — **all three must pass**             | Unanimous AND, not weighted voting |
| Multi-scale detection  | Heatmap peaks taken top-K per scale (no cross-scale fusion)      | None                        |
| Model ensemble / TTA   | —                                                                | **None** — single detector, single forward pass per crop |

Cheap future wins: weighted filter scoring instead of AND, flip-TTA on the landmark net to tighten IPD, multi-scale score fusion before top-K.

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

> The shipped `checkpoints/face_detector.pth` is the **v1** 200 k-param model at 64×64 and **does not load** into the v2 architecture above — you'll see a `state_dict` size-mismatch on startup. Retrain once with `python train_detector.py` to produce a fresh v2 checkpoint. The shipped `checkpoints/landmark_net.pth` is unchanged and still works.

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
```

Phase 1 bootstraps on positives + random negatives for 40 epochs. Phase 2 mines the Phase-1 model's own false positives as hard negatives and re-trains from scratch for another 40 epochs. With the v2 model (~1 M params, 96×96) and the full LFW + CelebA + WIDER + UTKFace + FairFace mix, expect ~2–3 hours on a single modern GPU.

### Landmark regressor

```bash
python train_landmarks.py
```

30 epochs, ~25 s each on an RTX-class GPU. Reaches ≈ 0.003 normalised error — about **0.2 pixels on a 64 × 64 crop**.

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

| Symptom                           | Fix                                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Box flickers on / off             | Already damped by tracker EMA + 45-frame persistence. If it's still flickering, lower `DET_THRESH` to 0.30. |
| Box drifts to walls or furniture  | Raise `DET_THRESH` to 0.55+. The filters are the second line of defence.                                    |
| Face never detected               | Lower `DET_THRESH` to 0.30. Also check `FOCAL_RATIO` / press `C` to calibrate.                              |
| Distance off                      | Press `C` at 1.0 m. Or pass `--focal N`.                                                                    |

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
├── train_detector.py              Trains FaceDetectorCNN
├── train_landmarks.py             Trains LandmarkCNN
├── download_datasets.py           LFW + WIDER FACE downloader
│
├── facetrack/                     Library package
│   ├── __init__.py
│   ├── config.py                  All tunable constants
│   ├── detector.py                FaceDetectorCNN architecture + transforms
│   ├── landmarks.py               LandmarkCNN architecture + transforms
│   ├── filters.py                 Geometric + symmetry + skin FP filters
│   ├── tracker.py                 SingleFaceTracker (sticky 1-face state machine)
│   └── pipeline.py                Sliding window + heatmap + DetectionWorker thread
│
└── checkpoints/
    ├── face_detector.pth          Trained detector
    └── landmark_net.pth           Trained landmark regressor
```

---

## Design decisions

- **Everything from scratch.** The whole point was to build and train every piece end-to-end on open face datasets.
- **Top-K peaks over the detector heatmap.** Simpler than connected components; landmark plausibility + filters do the FP rejection just as well.
- **EMA smoothing instead of Kalman.** Two multiplies per coord per frame vs. dozens, and the motion isn't complex enough to need a dynamical model.
- **Single-face tracker.** The classic "box jumps to a wall" failure comes from multi-track logic treating each detection independently. A single sticky track is simpler and more robust for the one-person use case.
- **Shipping the trained checkpoints.** They're small (~5 MB total) — cloning and running `python run.py` should just work. Training is an optional rebuild.
