"""
facetrack.landmarks  —  5-point face landmark regressor
========================================================

Architecture
────────────
 Small CNN (~150k params) that takes a 64×64 face crop and predicts:
     [ left_eye_xy, right_eye_xy, nose_xy, left_mouth_xy, right_mouth_xy ]
     = 10 values normalised to [0, 1] relative to the crop dimensions.

Why 5 landmarks?
────────────────
 * Eye-to-eye distance → IPD (for distance estimation)
 * Nose position relative to eye-midpoint → yaw angle
       When the face turns sideways, nose shifts off the eye midline while
       the eyes project closer together (IPD shrinks by cos(yaw)).
       Knowing yaw lets us un-shrink IPD → correct distance even when the
       user isn't facing the camera head-on.
 * Mouth corners → future pitch/roll correction; available free from CelebA.

 All five are exactly what CelebA annotates, so 100k labelled samples
 come essentially for free.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

LANDMARK_PATCH_SIZE = 64   # decoupled from detector patch size — the shipped
                           # landmark checkpoint was trained at 64×64, so this
                           # stays fixed even if DETECTOR_PATCH_SIZE changes.
NUM_LANDMARKS       = 5    # left eye, right eye, nose, left mouth, right mouth
NUM_OUTPUTS         = NUM_LANDMARKS * 2     # (x, y) per landmark


class LandmarkCNN(nn.Module):
    """
    Small CNN: 64×64 face crop → 4 scalars (left_eye, right_eye).

    Output is squashed to [0, 1] with a sigmoid so it cannot predict
    positions outside the crop.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1  64 → 32
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2),

            # Block 2  32 → 16
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(2),

            # Block 3  16 → 8
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.MaxPool2d(2),

            # Block 4  8 → 4
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(2),     # 2×2
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, NUM_OUTPUTS),
            nn.Sigmoid(),                # normalise to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))   # [B, NUM_OUTPUTS]


# ── Image transforms ──────────────────────────────────────────────────────────
#
# Photometric transforms only — geometric transforms (flip / rotation / affine)
# would invalidate landmark coordinates. Horizontal flip IS supported, but has
# to happen inside the Dataset so the corresponding left/right labels can be
# swapped at the same time (see LANDMARK_FLIP_PAIRS + LandmarkDataset).

from facetrack.detector import AddGaussianNoise   # reuse the detector's noise aug

LANDMARK_INFER_TF = T.Compose([
    T.Resize((LANDMARK_PATCH_SIZE, LANDMARK_PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

LANDMARK_TRAIN_TF = T.Compose([
    T.Resize((LANDMARK_PATCH_SIZE, LANDMARK_PATCH_SIZE)),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4,
                                 saturation=0.3, hue=0.08)], p=0.9),
    T.RandomGrayscale(p=0.15),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.25),
    T.RandomAutocontrast(p=0.20),
    T.ToTensor(),
    AddGaussianNoise(p=0.25, sigma_max=0.03),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# When horizontally flipping an image, these landmark-index pairs must be
# swapped in the 5-point label vector (lex, ley, rex, rey, nox, noy,
# lmx, lmy, rmx, rmy). Left eye ↔ right eye; left mouth ↔ right mouth.
# Nose (index 2) stays in place; only its x flips.
LANDMARK_FLIP_PAIRS = [(0, 1), (3, 4)]   # (left_eye, right_eye), (left_mouth, right_mouth)


def flip_landmarks_horizontal(label_vec):
    """
    In-place mirror for a 10-float label vector after a horizontal image flip.
        input:  [lex, ley, rex, rey, nox, noy, lmx, lmy, rmx, rmy]
                (normalised to [0, 1])
        output: same layout, with left/right swapped and x-coords mirrored.
    """
    out = list(label_vec)
    # Mirror x-coordinates
    for i in (0, 1, 2, 3, 4):
        out[2*i] = 1.0 - out[2*i]
    # Swap symmetrical pairs (eyes, mouth corners)
    for a, b in LANDMARK_FLIP_PAIRS:
        out[2*a],   out[2*b]   = out[2*b],   out[2*a]
        out[2*a+1], out[2*b+1] = out[2*b+1], out[2*a+1]
    return out


def predict_eyes(model: LandmarkCNN, face_crop_pil, device) -> tuple:
    """
    Run the landmark model on one face crop.

    Returns (left_eye, right_eye) where each is (x, y) in crop pixels.
    """
    W, H = face_crop_pil.size
    tens = LANDMARK_INFER_TF(face_crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tens).cpu().squeeze(0).tolist()
    lx, ly, rx, ry = out
    return (int(lx * W), int(ly * H)), (int(rx * W), int(ry * H))
