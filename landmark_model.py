"""
landmark_model.py  —  5-point face landmark regressor
======================================================

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

from face_detector_model import DETECTOR_PATCH_SIZE

LANDMARK_PATCH_SIZE = DETECTOR_PATCH_SIZE   # 64 — same as detector input
NUM_LANDMARKS       = 5                     # left eye, right eye, nose, left mouth, right mouth
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

LANDMARK_INFER_TF = T.Compose([
    T.Resize((LANDMARK_PATCH_SIZE, LANDMARK_PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Training transform: we cannot use HFlip or rotation here — those would
# invalidate the landmark coordinates.  Only photometric augs are safe.
LANDMARK_TRAIN_TF = T.Compose([
    T.Resize((LANDMARK_PATCH_SIZE, LANDMARK_PATCH_SIZE)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.1),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


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
