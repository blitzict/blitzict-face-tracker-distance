"""
facetrack.landmarks  —  5-point face landmark regressor
========================================================

Architecture
────────────
 Small CNN (~500k params) that takes a 64×64 face crop and predicts:
     [ left_eye_xy, right_eye_xy, nose_xy, left_mouth_xy, right_mouth_xy ]
     = 10 values normalised to [0, 1] relative to the crop dimensions.

Why 5 landmarks?
────────────────
 * Eye-to-eye distance → IPD (for distance estimation)
 * Nose position relative to eye-midpoint → yaw angle
 * Mouth corners → mouth-width cue for multi-cue distance fusion

Why 64×64 (not 128×128)?
────────────────────────
 A 128×128 variant was tried (see git history around commit 81485cf).
 The theoretical win — 4× more input pixels → sharper eye centres — did
 not survive the real detection pipeline. Inference-time detector boxes
 are sometimes much wider than the actual face, and the sharper 128×128
 net collapses all landmarks to the crop centre in that case, failing
 plausible_face_geometry. The 64×64 net's implicit blur from downsampling
 was tolerant of that framing variance. Lesson: landmark training needs
 random-scale / random-position augmentation before a larger input can
 beat 64×64 in practice.
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
    Small CNN: 64×64 face crop → 10 normalised landmark coords.

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


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap-regression variant (DSNT-style soft-argmax)
# ─────────────────────────────────────────────────────────────────────────────
#
# Direct-coord regression (LandmarkCNN above) floors out around 0.2-0.3 px of
# landmark error at 64×64 — the last FC layer has to compress all spatial
# information into 10 scalars and ~0.005 normalised error is about as tight
# as that bottleneck allows. Heatmap regression moves the spatial inference
# back into the feature map: the head outputs one heatmap per landmark, and
# a soft-argmax decodes sub-pixel coordinates by taking the expected value
# under the spatial softmax. The argmax is fully differentiable so the model
# trains with the same coord-space loss as the direct-regression variant.
#
# Public API is the same as LandmarkCNN — `forward(x) -> [B, 10]` normalised
# coords — so this class is a drop-in replacement for pipeline.py.

HEATMAP_SIZE = 32      # decoder output H×W; soft-argmax is sub-pixel so this
                       # does not bound accuracy the way the input patch does


class LandmarkHeatmapCNN(nn.Module):
    """
    64×64 face crop → 5 heatmaps @ HEATMAP_SIZE × HEATMAP_SIZE → soft-argmax →
    10 normalised coords. Same [B, 10] output shape as LandmarkCNN so the
    inference pipeline needs no changes.
    """

    def __init__(self, heatmap_size: int = HEATMAP_SIZE):
        super().__init__()
        self.heatmap_size = heatmap_size

        # Encoder — keeps spatial resolution at 8×8 after block 3 (no final
        # adaptive-pool, unlike the direct-regression LandmarkCNN).
        self.features = nn.Sequential(
            # Block 1  64 → 32
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
            nn.MaxPool2d(2),
            # Block 2  32 → 16
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
            nn.MaxPool2d(2),
            # Block 3  16 → 8  (no pool after — decoder upsamples from here)
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.SiLU(),
        )

        # Decoder: 8 → 16 → 32, final 1×1 conv to NUM_LANDMARKS channels.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, NUM_LANDMARKS, kernel_size=1),     # 5 heatmaps
        )

    def forward_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logit heatmaps — [B, 5, H, W]. Used by training for the DSNT
        regulariser and by visualisation."""
        return self.decoder(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-argmax → normalised coords [B, 10] in (x0,y0,x1,y1,...) order."""
        return soft_argmax(self.forward_heatmap(x))


# ── Heatmap decode + DSNT regulariser helpers ────────────────────────────────

def soft_argmax(heatmap_logits: torch.Tensor) -> torch.Tensor:
    """
    Decode [B, K, H, W] logit heatmaps to [B, 2K] normalised (x, y) coords,
    using the softmax-weighted expected value (differentiable sub-pixel argmax).
    """
    B, K, H, W = heatmap_logits.shape
    prob = torch.softmax(heatmap_logits.view(B, K, -1), dim=-1).view(B, K, H, W)
    ys = torch.linspace(0.0, 1.0, H, device=heatmap_logits.device).view(1, 1, H, 1)
    xs = torch.linspace(0.0, 1.0, W, device=heatmap_logits.device).view(1, 1, 1, W)
    x_coord = (prob * xs).sum(dim=(2, 3))                    # [B, K]
    y_coord = (prob * ys).sum(dim=(2, 3))
    coords  = torch.stack([x_coord, y_coord], dim=-1)        # [B, K, 2]
    return coords.view(B, K * 2)


def _gaussian_target(target_coords_norm: torch.Tensor,
                     H: int, W: int, sigma_px: float) -> torch.Tensor:
    """
    Build a [B, K, H, W] Gaussian prior centred on each ground-truth landmark,
    normalised to sum=1 per heatmap.
        target_coords_norm : [B, K, 2] in [0, 1]
        sigma_px           : Gaussian sigma in heatmap-pixel units
    """
    B, K, _ = target_coords_norm.shape
    gx = target_coords_norm[..., 0] * (W - 1)                # [B, K]
    gy = target_coords_norm[..., 1] * (H - 1)
    ys = torch.arange(H, device=target_coords_norm.device,
                      dtype=target_coords_norm.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=target_coords_norm.device,
                      dtype=target_coords_norm.dtype).view(1, 1, 1, W)
    g = torch.exp(-((xs - gx.view(B, K, 1, 1)) ** 2
                    + (ys - gy.view(B, K, 1, 1)) ** 2)
                  / (2.0 * sigma_px * sigma_px))
    g = g / g.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return g                                                  # [B, K, H, W]


def dsnt_regulariser(heatmap_logits: torch.Tensor,
                     target_coords_norm: torch.Tensor,
                     sigma_px: float = 1.0) -> torch.Tensor:
    """
    Jensen-Shannon divergence between the softmax'd heatmap and a Gaussian
    prior centred on the ground-truth coord. Pushes heatmaps to be peaky and
    correctly located — without it, soft-argmax operates on a blurry blob and
    loses the sub-pixel advantage of heatmap regression over direct coords.

    Reference: Nibali et al., "Numerical Coordinate Regression with
    Convolutional Neural Networks" (arXiv:1801.07372).

    Args
        heatmap_logits     : [B, K, H, W] — model.forward_heatmap output
        target_coords_norm : [B, K, 2]    — ground-truth coords in [0, 1]
        sigma_px           : target Gaussian sigma in heatmap pixels

    Returns
        scalar loss (batch + landmark mean).
    """
    B, K, H, W = heatmap_logits.shape
    p = torch.softmax(heatmap_logits.view(B, K, -1), dim=-1).view(B, K, H, W)
    q = _gaussian_target(target_coords_norm, H, W, sigma_px)
    m = 0.5 * (p + q)

    eps  = 1e-8
    logp = p.clamp(min=eps).log()
    logq = q.clamp(min=eps).log()
    logm = m.clamp(min=eps).log()

    kl_pm = (p * (logp - logm)).sum(dim=(-2, -1))            # [B, K]
    kl_qm = (q * (logq - logm)).sum(dim=(-2, -1))
    return 0.5 * (kl_pm + kl_qm).mean()


def build_landmark_net(arch: str = 'direct'):
    """Factory: 'direct' (LandmarkCNN) or 'heatmap' (LandmarkHeatmapCNN)."""
    if arch == 'heatmap':
        return LandmarkHeatmapCNN()
    if arch == 'direct':
        return LandmarkCNN()
    raise ValueError(f"Unknown landmark arch: {arch!r}")


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
