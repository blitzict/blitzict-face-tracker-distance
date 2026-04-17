"""
Custom Face Detector
─────────────────────────────────────────────────────────────
Lightweight binary CNN: face vs background.
Input  : 64×64 RGB patch
Output : scalar logit (positive = face)

At inference we run this as a GPU-batched sliding window across
multiple scales, then apply NMS to get clean bounding boxes.
"""

import torch
import torch.nn as nn


class FaceDetectorCNN(nn.Module):
    """
    ~200k parameter face/background classifier.
    Fast enough to process hundreds of patches per frame in one GPU batch.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 64×64 → 32×32
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),

            # Block 2 — 32×32 → 16×16
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),

            # Block 3 — 16×16 → 8×8
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),

            # Block 4 — 8×8 → 4×4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(2),   # → 2×2
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),        # raw logit; BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)   # [B]


# ── Image normalisation (same stats as recognition model) ────────────────────
import torchvision.transforms as T

DETECTOR_PATCH_SIZE = 64

DETECTOR_TRAIN_TF = T.Compose([
    T.Resize((DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    T.RandomGrayscale(p=0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

DETECTOR_INFER_TF = T.Compose([
    T.Resize((DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
