"""Obstacle representations for local_opt.

All obstacles expose the same interface so cost.py can iterate uniformly:

  signed_dist(p, t) -> float
      Negative inside the obstacle, 0 on the boundary, positive outside.
      `t` is parameter along the trajectory — static obstacles ignore it,
      dynamic ones use it to evaluate their predicted pose at time t.

  class_name : str
      Links the obstacle to AvoidParams in the avoid_config dict.

Per the chosen first-version representations:
  walls  → AABBObstacle (axis-aligned box)
  humans → SphereObstacle (centre + radius, optional linear velocity)
"""
from __future__ import annotations

import numpy as np


class Obstacle:
    """Base interface. Subclasses must override signed_dist + class_name."""
    class_name: str = ""

    def signed_dist(self, p: np.ndarray, t: float) -> float:
        raise NotImplementedError


class SphereObstacle(Obstacle):
    """Sphere with optional constant linear velocity (linear extrapolation).

    centre(t) = centre0 + vel * t
    signed_dist = ||p - centre(t)|| - radius
    """

    def __init__(self, centre0, radius: float, vel=None, class_name: str = "human"):
        self.centre0 = np.asarray(centre0, dtype=np.float64).reshape(3)
        self.radius = float(radius)
        self.vel = (np.zeros(3, dtype=np.float64) if vel is None
                    else np.asarray(vel, dtype=np.float64).reshape(3))
        self.class_name = class_name

    def predict(self, t: float) -> np.ndarray:
        return self.centre0 + self.vel * float(t)

    def signed_dist(self, p: np.ndarray, t: float) -> float:
        c = self.predict(t)
        return float(np.linalg.norm(np.asarray(p) - c) - self.radius)


class AABBObstacle(Obstacle):
    """Axis-aligned bounding box; static (no .predict(t)).

    signed_dist for points outside is the euclidean distance to the closest
    face; for points inside it is the negative penetration depth (max of
    per-axis penetrations, i.e. L∞), which is sufficient for the (d_safe - d)
    penalty since any inside-point yields a large positive penalty already.
    """

    def __init__(self, lo, hi, class_name: str = "wall"):
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if np.any(hi < lo):
            raise ValueError(f"AABB hi must be ≥ lo elementwise; lo={lo}, hi={hi}")
        self.lo = lo
        self.hi = hi
        self.class_name = class_name

    def signed_dist(self, p: np.ndarray, t: float = 0.0) -> float:
        p = np.asarray(p, dtype=np.float64)
        outside = np.maximum(self.lo - p, 0.0) + np.maximum(p - self.hi, 0.0)
        if np.any(outside > 0):
            return float(np.linalg.norm(outside))
        inside = float(np.max(np.maximum(self.lo - p, p - self.hi)))  # ≤ 0
        return inside
