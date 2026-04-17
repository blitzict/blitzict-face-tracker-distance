"""
facetrack — real-time face tracking and distance estimation.

Public API
──────────
    detector   — FaceDetectorCNN  (binary face / background classifier)
    landmarks  — LandmarkCNN      (5-point facial landmark regressor)
    tracker    — SingleFaceTracker (sticky single-face state machine)
    pipeline   — DetectionWorker   (glue: detector → landmark → distance)
    filters    — symmetry_score, skin_ratio (false-positive filters)
    config     — tunable constants in one place
"""

from facetrack import config
from facetrack.detector  import FaceDetectorCNN, DETECTOR_PATCH_SIZE
from facetrack.landmarks import LandmarkCNN, LANDMARK_INFER_TF
from facetrack.tracker   import SingleFaceTracker
from facetrack.pipeline  import DetectionWorker, scores_to_boxes
from facetrack.filters   import symmetry_score, skin_ratio

__all__ = [
    'config',
    'FaceDetectorCNN', 'DETECTOR_PATCH_SIZE',
    'LandmarkCNN', 'LANDMARK_INFER_TF',
    'SingleFaceTracker',
    'DetectionWorker', 'scores_to_boxes',
    'symmetry_score', 'skin_ratio',
]
