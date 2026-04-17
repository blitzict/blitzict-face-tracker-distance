# Real-Time Face Tracking + Distance Estimation

A self-contained computer-vision pipeline that detects a face in a webcam feed, locates the eyes, and reports how far the person is from the camera in metres — at 30–60 FPS on a modest GPU.

Everything under the hood is custom: the face detector, the eye-landmark regressor, the tracker, the distance formula. No MediaPipe, no Haar cascade at inference, no pre-trained black boxes.

---

## 1. What it does

Open `inference.py`, point a webcam at yourself:

- A **green bounding box** locks onto your face and follows it smoothly.
- The **distance in metres** is shown at the top of the box, updated live as you move.
- The detection stays stable across frames (temporal smoothing kills flicker).
- Works on anyone's webcam — a one-time calibration press gives ±5 % accuracy.

```
python inference.py --camera 1
```

**Controls**

| Key      | Action                                               |
| -------- | ---------------------------------------------------- |
| `Q` / Esc | Quit                                                |
| `S`      | Save a screenshot as `screenshot_NNNN.png`          |
| `C`      | Calibrate focal length — stand exactly 1.0 m away first |

---

## 2. The pipeline

```
┌─────────────────┐
│ Webcam frame    │ 1280 × 720 BGR
└────────┬────────┘
         │
         ▼
┌────────────────────────┐
│ FaceDetectorCNN        │ Stage 1 — find the face
│ (binary classifier,    │ sliding window on 320×240 downscale
│  ~200 k params)        │ returns one tight box per frame
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ LandmarkCNN            │ Stage 2 — find the two eyes
│ (regression,           │ inside the detected crop
│  ~150 k params)        │ outputs normalised (lex,ley,rex,rey)
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Plausibility check     │ Eyes must land in the upper 70 % of
│ (reject false positives)│ the crop, be roughly level, IPD ≥ 15 %
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Distance formula       │ dist = (0.063 × focal_px) / ipd_px
│ (IPD-based geometry)   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ CentroidTracker + EMA  │ Smooths box + distance across frames
│ (persistent IDs,       │ so the display doesn't flicker
│  MAX_DISAPPEARED = 12) │
└────────┬───────────────┘
         │
         ▼
      Display
```

**Why IPD instead of face width?**  Inter-pupillary distance is ±3 mm across adults; face width is ±10 mm. Using IPD cuts cross-user distance error roughly in half.

---

## 3. Models

| Model                 | File                          | Params   | Input       | Output                       |
| --------------------- | ----------------------------- | -------- | ----------- | ---------------------------- |
| **FaceDetectorCNN**   | `face_detector_model.py`      | ~200 k   | 64 × 64 RGB | Scalar logit — face vs bg    |
| **LandmarkCNN**       | `landmark_model.py`           | ~150 k   | 64 × 64 RGB | 4 floats — left/right eye xy |

Both share the same 4-block conv backbone. The landmark model is warm-startable from the detector backbone (not done by default because training is fast anyway).

Checkpoints live in `checkpoints/`:

- `face_detector.pth` — detector
- `landmark_net.pth`  — eye regressor

---

## 4. Datasets

### Used during training

| Dataset              | Role                                       | Size        | Source                                                                         |
| -------------------- | ------------------------------------------ | ----------- | ------------------------------------------------------------------------------ |
| **DnHFaces**         | Positives (11 subjects at 31 distances)    | ~1.4 k crops | Included in repo                                                               |
| **Haar-annotated scenes** | Positives from full scenes + easy negatives | ~5 k         | Derived from DnHFaces `photos_all/`                                            |
| **LFW**              | Diverse positive face crops (in the wild)  | 13 k        | `python download_datasets.py`                                                 |
| **WIDER-selected**   | Positives at varied scales, 68-pt landmarks | 4.3 k imgs  | `kaggle datasets download -d alirezakay/wider-selected`                       |
| **CelebA aligned**   | Positives (large diversity, capped at 30 k) | 30 k / 200 k | `kaggle datasets download -d jessicali9530/celeba-dataset`                    |
| **Hard negatives (mined)** | False positives found by phase-1 model | ≤ 6 k       | Auto-generated during Phase-2 training                                         |

### Used only for the landmark model

The landmark regressor uses the subset of positives that has eye annotations:

| Dataset            | Eye annotations             | Size used |
| ------------------ | --------------------------- | --------- |
| **WIDER-selected** | 68-point landmarks → eye centres = mean of indices 36-41 / 42-47 | 6.2 k |
| **CelebA**         | 5-point landmarks (includes both eye centres directly)           | 100 k   |

### Explicitly excluded / removed

| Not used            | Why                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| BIWI Kinect         | All subjects sit at one fixed distance (~0.9 m). No distance variety → useless for distance training.               |
| HRRFaceD / LRRFaceD | "Range" in the name refers to wavelength, not distance. No distance annotations.                                    |
| HPE-360             | Fisheye-distorted pose dataset — distortion would actively hurt the detector.                                       |
| FFHQ                | High-resolution portraits that would just be downsampled to 64 × 64 anyway; adds bulk without new variety.          |
| Scale augmentation  | A synthetic augmentation I tried that composited mini-face crops on random backgrounds. It taught the detector to fire on any textured background → removed. |

---

## 5. Training

Every script is self-contained. Run them in this order:

### 5.1 Download datasets

```bash
# One-time Kaggle setup (skip if already done)
pip install kaggle
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# LFW (13 k faces)
python download_datasets.py

# WIDER-selected (needed for landmarks)
kaggle datasets download -d alirezakay/wider-selected -p datasets/tmp_wider --unzip

# CelebA aligned (~1.4 GB — needed for detector + landmarks)
kaggle datasets download -d jessicali9530/celeba-dataset -p datasets/tmp_celeba --unzip
```

Once downloaded, run `python train_face_detector.py` — it will also extract 68-landmark face crops from WIDER-selected into `datasets/wider_face_crops/` the first time.

### 5.2 Train the face detector

```bash
python train_face_detector.py
```

Two phases:

1. **Phase 1 — bootstrap.** Trains on all positives + random negatives for 40 epochs.
2. **Phase 2 — hard negative retrain.** Runs phase-1 model on full scenes, collects its false positives, adds them as "hard negatives", trains a fresh detector from scratch for another 40 epochs.

Target metrics: val precision ≥ 0.98, recall ≥ 0.99.

### 5.3 Train the landmark regressor

```bash
python train_landmarks.py
```

- 30 epochs, ~25 s each on an RTX-class GPU.
- Final val error ≈ 0.003 normalised, or **< 0.2 px on a 64 × 64 crop**.

### 5.4 (Optional) Collect your own face

Adding your face to DnHFaces before training usually improves recognition of you specifically, though it is **not** required — the pipeline is identity-agnostic by design.

```bash
python collect_my_face.py --subject l --camera 1
```

Walk through the prompts: stand at each distance, press Space 8 times per distance.

---

## 6. Inference — usage

```bash
python inference.py --camera 1                 # default focal = 900 px
python inference.py --camera 1 --focal 850     # use your measured focal
python inference.py --camera 1 --det_thresh 0.3  # looser detector
```

### Flags

| Flag            | Default | Meaning                                                                 |
| --------------- | ------- | ----------------------------------------------------------------------- |
| `--camera`      | 0       | Camera index (use 1 for an external webcam / phone camera)              |
| `--det_ckpt`    | `checkpoints/face_detector.pth` | Detector weights                                        |
| `--det_thresh`  | 0.45    | Sigmoid score threshold — lower = more detections, more false positives |
| `--focal`       | 900     | Camera focal length in pixels — calibrate with `C` for best accuracy    |

### Calibration

Stand exactly 1.0 m from the camera, wait for the green box to stabilise on your face, then press `C`. The console will print something like:

```
Calibrated (IPD): focal=742px
  Tip: --focal 742  to skip calibration next time
```

Add `--focal 742` the next time you run the script and you're done — accuracy usually lands within ±5 % across the 0.5 – 5 m range.

---

## 7. Tuning

| Symptom                                | Fix                                                    |
| -------------------------------------- | ------------------------------------------------------ |
| Box flickers on / off                  | Already handled — `MAX_DISAPPEARED=12`, EMA smoothing. If still flickering, lower `DET_THRESH` to 0.35.           |
| Box drifts to walls (false positives)  | Raise `DET_THRESH` to 0.55+. The landmark plausibility check is your second line of defence.                                         |
| Face never detected                    | Lower `DET_THRESH` to 0.30. If that doesn't help, the model hasn't seen someone like you — run `collect_my_face.py` and retrain.                 |
| Distance reads wildly wrong            | Press `C` at 1.0 m. Or pass `--focal N` where N matches your camera.                                             |

---

## 8. Repo layout

```
ai/
├── inference.py                 Main real-time loop
├── face_detector_model.py       FaceDetectorCNN architecture
├── landmark_model.py            LandmarkCNN architecture
├── distance_model.py            (legacy) FaceDistanceRegressor — not used at inference
├── train_face_detector.py       Two-phase detector training
├── train_landmarks.py           Landmark regressor training
├── train_distance.py            (legacy) distance regressor training — no longer recommended
├── collect_my_face.py           Webcam utility to add your face to DnHFaces
├── download_datasets.py         LFW + WIDER FACE downloader (Kaggle)
├── dataset.py                   Shared dataset constants (DIST_MIN_M etc.)
├── model.py                     (legacy) FaceDistanceNet — superseded by landmark pipeline
│
├── checkpoints/
│   ├── face_detector.pth        Current production detector
│   └── landmark_net.pth         Current production landmark model
│
├── datasets/
│   ├── lfw/                     LFW positives (after download_datasets.py)
│   ├── wider_face_crops/        Extracted WIDER face crops for detector training
│   ├── tmp_wider/               Raw WIDER-selected images + annotations
│   └── tmp_celeba/              Raw CelebA aligned images + landmarks CSV
│
└── DnHFaces/open_data_set/
    ├── photos_all/              Full-scene photos (used for neg mining + Haar anno)
    └── photos_all_faces/        Pre-cropped DnHFaces faces (primary positives)
```

---

## 9. Honest accuracy expectations

| Setup                                             | Distance accuracy |
| ------------------------------------------------- | ----------------- |
| Out-of-box, default focal = 900 px                | ±20 %             |
| After one `C` calibration at 1.0 m                | ±5 %              |
| Your friend's laptop, no calibration              | ±25 %             |
| Your friend's laptop, one calibration             | ±5 %              |

The remaining error is dominated by focal-length uncertainty (camera-specific) and, secondarily, IPD variation (anatomy-specific). There is no way to drive this to zero without a reference — monocular depth from one unknown camera is mathematically a calibration problem.

---

## 10. Design decisions (for future-me)

- **Why no MediaPipe?**  The whole point of this project was to build every model from scratch. A MediaPipe solution would be more accurate and smaller — and also not what we wanted to build.
- **Why top-1 peak instead of connected components?**  Simpler. False positives are handled by the landmark plausibility check downstream.
- **Why EMA smoothing in the tracker instead of a Kalman filter?**  EMA is two multiplies per coord per frame. Kalman is overkill for box smoothing at 30 FPS.
- **Why drop the identity (recognition) model?**  It only recognised 11 DnHFaces subjects and was the main source of "0.xx m" misfires when it gave low confidence on unknown faces. The pipeline is strictly better without it — if identity ever matters, re-wire it as an optional stage 3.
- **Why CelebA capped at 30 k, not 200 k?**  Diminishing returns past ~30 k and a 7× training-time penalty. The detector already saturates near P=R=1.00.
