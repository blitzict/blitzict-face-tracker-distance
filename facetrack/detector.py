"""
facetrack.detector  —  binary face / background classifier
===========================================================
Wider CNN (~1.0M parameters) at 96×96 input.
    Input   : 96×96 RGB patch
    Output  : scalar logit (positive ⇒ face)

At inference we run this as a GPU-batched sliding window across multiple
scales and take the local maxima of the resulting heatmap — see
`facetrack.pipeline` for the wiring.

v2 note
───────
Upgraded from 64×64, ~200k-param 4-block net. The prior model overfit to
LFW + CelebA (frontal, narrow demographics) and generalised poorly to
arbitrary webcams / people. This version adds capacity (5 blocks, double
conv per block), a larger input (96×96), and heavier augmentation
(perspective, blur, additive noise, erasing) to force invariance to
lighting, compression, and camera differences.
"""

import random

import torch
import torch.nn as nn
import torchvision.transforms as T


DETECTOR_PATCH_SIZE = 96


class FaceDetectorCNN(nn.Module):
    """
    ~1.0M parameter face/background classifier.
    5-block conv backbone with double conv per block, BN + SiLU, MaxPool
    between blocks. Global average pool + small MLP head.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(  3,  32),   # 96 → 48
            block( 32,  64),   # 48 → 24
            block( 64, 128),   # 24 → 12
            block(128, 192),   # 12 → 6
            block(192, 256),   #  6 → 3
            nn.AdaptiveAvgPool2d(1),   # → 1×1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),        # raw logit; BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)   # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation — aggressive, targets real-world webcam failure modes
# ─────────────────────────────────────────────────────────────────────────────
#
# Each transform here addresses a specific class of train/test gap that
# hurt the prior model:
#   ColorJitter / RandomGrayscale / RandomAutocontrast  → lighting, white-balance
#   GaussianBlur                                         → cheap cameras, defocus
#   RandomPerspective / RandomAffine                     → pose, tilt, off-axis
#   AddGaussianNoise                                     → low-light sensor noise
#   RandomErasing                                        → occlusion (hand, glasses frame)

class AddGaussianNoise:
    """Add gaussian noise to a [0, 1] tensor (post-ToTensor, pre-Normalize)."""
    def __init__(self, p: float = 0.3, sigma_max: float = 0.04):
        self.p = p
        self.sigma_max = sigma_max

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return t
        sigma = random.uniform(0.0, self.sigma_max)
        return (t + torch.randn_like(t) * sigma).clamp_(0.0, 1.0)


DETECTOR_TRAIN_TF = T.Compose([
    T.Resize((DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5,
                                 saturation=0.4, hue=0.10)], p=0.9),
    T.RandomGrayscale(p=0.15),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.30),
    T.RandomAutocontrast(p=0.20),
    T.RandomAffine(degrees=20, translate=(0.05, 0.05),
                   scale=(0.90, 1.10), shear=5),
    T.RandomPerspective(distortion_scale=0.20, p=0.30),
    T.ToTensor(),
    AddGaussianNoise(p=0.30, sigma_max=0.04),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.5, 2.0)),
])

DETECTOR_INFER_TF = T.Compose([
    T.Resize((DETECTOR_PATCH_SIZE, DETECTOR_PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
