"""
facetrack.tracker  —  sticky single-face tracker
=================================================

Why single-face?
    The use case is one person in front of the webcam.  Generic multi-track
    logic (Hungarian matching, N IDs, split/merge) adds complexity that buys
    us nothing — and the classic "box jumps to a wall" failure mode comes
    precisely from treating each detection as independent.

This tracker:
    • Requires N consecutive confirming frames to lock on (avoids single-
      frame false positives stealing the lock).
    • Once locked, accepts only the detection nearest the current track.
    • Rejects matches where the 5 facial landmarks shift by more than a
      fraction of the box size between frames (temporal stability gate).
    • EMA-blends the box for a smooth display; median-filters distance
      across the last N frames.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from facetrack.config import (
    TRACK_BOX_ALPHA,
    LMK_TEMPORAL_MAX,
)


class SingleFaceTracker:
    """One face, one track, one distance — sticky state machine."""

    MAX_DISAPPEARED = 45     # ~1.5 s at 30 FPS
    MAX_DISTANCE    = 200    # pixel radius for "same face"
    DIST_WINDOW     = 7      # frames in the distance median window
    LOCK_CONFIRM    = 2      # consecutive confirming frames required to lock on

    def __init__(self) -> None:
        self.track:      Optional[dict] = None
        self.age:        int            = 0
        self.dist_q:     List[float]    = []
        self.candidate:  Optional[dict] = None
        self.cand_hits:  int            = 0

    # ------------------------------------------------------------------ helpers

    def _smooth_dist(self, new_dist: float) -> float:
        """Median filter across `DIST_WINDOW` recent readings."""
        self.dist_q.append(new_dist)
        if len(self.dist_q) > self.DIST_WINDOW:
            self.dist_q.pop(0)
        return round(sorted(self.dist_q)[len(self.dist_q) // 2], 2)

    def _reset_all(self) -> None:
        self.track       = None
        self.dist_q      = []
        self.candidate   = None
        self.cand_hits   = 0
        self.age         = 0

    # --------------------------------------------------------------------- API

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Feed this frame's detections and get back the current track (0 or 1).

        detections: list of dicts produced by DetectionWorker; each must have
            'box', 'cx'/'cy' (or we compute), 'dist', 'lmks_abs' (optional).
        """
        # ── No detections this frame → age any existing track ────────────────
        if not detections:
            if self.track is not None:
                self.age += 1
                if self.age > self.MAX_DISAPPEARED:
                    self._reset_all()
                    return []
                return [self.track]
            self.candidate, self.cand_hits = None, 0
            return []

        # Centroids for distance matching
        cents = [((d['box'][0] + d['box'][2]) // 2,
                  (d['box'][1] + d['box'][3]) // 2) for d in detections]

        # ── First-time lock-on: need LOCK_CONFIRM confirming frames ──────────
        if self.track is None:
            return self._handle_lock_on(detections, cents)

        # ── Existing track: match to nearest detection ───────────────────────
        tx, ty = self.track['cx'], self.track['cy']
        dists  = [float(np.hypot(cx - tx, cy - ty)) for cx, cy in cents]
        best_i = int(np.argmin(dists))

        if dists[best_i] > self.MAX_DISTANCE:
            # Closest detection is too far to be the same face. Age the
            # current track, but also accumulate a re-lock candidate at
            # the distant detection's position. If it confirms for
            # LOCK_CONFIRM consecutive frames, abandon the (probably
            # wrong) current lock immediately rather than coasting all
            # the way to MAX_DISAPPEARED.
            self.age += 1

            new    = detections[best_i]
            cx, cy = cents[best_i]
            if self.candidate is None:
                self.candidate = {**new, 'cx': cx, 'cy': cy}
                self.cand_hits = 1
            else:
                dx = cx - self.candidate['cx']
                dy = cy - self.candidate['cy']
                if float(np.hypot(dx, dy)) < self.MAX_DISTANCE:
                    self.cand_hits += 1
                    self.candidate = {**new, 'cx': cx, 'cy': cy}
                    if self.cand_hits >= self.LOCK_CONFIRM:
                        # Re-lock: jump to the new position
                        self.track     = self.candidate
                        self.dist_q    = [self.track['dist']]
                        self.age       = 0
                        self.candidate, self.cand_hits = None, 0
                        return [self.track]
                else:
                    # Re-lock candidate itself drifted — restart accumulation
                    self.candidate = {**new, 'cx': cx, 'cy': cy}
                    self.cand_hits = 1

            if self.age > self.MAX_DISAPPEARED:
                self._reset_all()
                return []
            return [self.track] if self.track else []

        new = detections[best_i]

        # ── Temporal landmark-stability gate ─────────────────────────────────
        # Real faces: landmarks glide smoothly.  False positives on a different
        # object: landmarks jump by >30 % of box size.  Reject those.
        if not self._landmarks_stable(new):
            self.age += 1
            if self.age > self.MAX_DISAPPEARED:
                self._reset_all()
                return []
            return [self.track]

        # ── Matched: blend box, median distance, keep everything else fresh ──
        self._merge_match(new, cents[best_i])
        return [self.track]

    # ------------------------------------------------------------ internal ops

    def _handle_lock_on(self, detections: List[dict], cents: List[tuple]) -> List[dict]:
        """Accumulate confirmations before committing to a first track."""
        best_i = max(range(len(detections)),
                     key=lambda i: detections[i].get('conf', 1.0))
        cx, cy = cents[best_i]

        if self.candidate is None:
            self.candidate = {**detections[best_i], 'cx': cx, 'cy': cy}
            self.cand_hits = 1
            return []

        dx = cx - self.candidate['cx']
        dy = cy - self.candidate['cy']
        if float(np.hypot(dx, dy)) < self.MAX_DISTANCE:
            self.cand_hits += 1
            self.candidate = {**detections[best_i], 'cx': cx, 'cy': cy}
            if self.cand_hits >= self.LOCK_CONFIRM:
                self.track  = self.candidate
                self.dist_q = [self.track['dist']]
                self.age    = 0
                self.candidate, self.cand_hits = None, 0
                return [self.track]
        else:
            # Drift — restart candidate accumulation
            self.candidate = {**detections[best_i], 'cx': cx, 'cy': cy}
            self.cand_hits = 1

        return []

    def _landmarks_stable(self, new: dict) -> bool:
        """Return False if any landmark shifted more than LMK_TEMPORAL_MAX × box."""
        old_lmks = self.track.get('lmks_abs')  if self.track else None
        new_lmks = new.get('lmks_abs')
        if old_lmks is None or new_lmks is None:
            return True

        ox1, oy1, ox2, oy2 = self.track['box']
        box_size = max(ox2 - ox1, oy2 - oy1, 1)
        max_shift = 0.0
        for i in range(5):
            dx = old_lmks[2*i]   - new_lmks[2*i]
            dy = old_lmks[2*i+1] - new_lmks[2*i+1]
            s  = float(np.hypot(dx, dy))
            if s > max_shift:
                max_shift = s
        return max_shift <= LMK_TEMPORAL_MAX * box_size

    def _merge_match(self, new: dict, centroid: tuple) -> None:
        """EMA-blend box, median-filter distance, adopt new landmarks."""
        a = TRACK_BOX_ALPHA
        ox1, oy1, ox2, oy2 = self.track['box']
        nx1, ny1, nx2, ny2 = new['box']
        smooth_box = (
            int((1 - a) * ox1 + a * nx1), int((1 - a) * oy1 + a * ny1),
            int((1 - a) * ox2 + a * nx2), int((1 - a) * oy2 + a * ny2),
        )
        self.track = {
            **new,
            'box':  smooth_box,
            'dist': self._smooth_dist(new['dist']),
            'cx':   centroid[0],
            'cy':   centroid[1],
        }
        self.age = 0
