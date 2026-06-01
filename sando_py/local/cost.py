"""Per-class obstacle-avoidance cost along a B-spline trajectory.

  EGO-Planner cubic penalty:   φ(d) = (d_safe - d)^3 if d < d_safe else 0
                               (C^2 continuous at the safety boundary)

  obstacle_cost(spline, obstacles, config, K=None) → scalar
      (1/K) · Σ_obs Σ_i  weight(obs.class) · φ(signed_dist(spline.eval(t_i), t_i),
                                               d_safe(obs.class))
      Averaged over K samples (uniform in t) so the cost is K-invariant —
      otherwise K and weight are entangled and grid refinement silently
      re-scales the optimisation.

Hard vs soft is just a weight knob — same shape — so the cost stays smooth
and differentiable (numerical gradients work fine for L-BFGS).
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .avoid_config import AvoidParams


def penalty(d: float, d_safe: float) -> float:
    diff = d_safe - d
    return diff * diff * diff if diff > 0.0 else 0.0


def obstacle_cost(spline, obstacles: Iterable, config: Dict[str, AvoidParams],
                  K: int = None) -> float:
    """Sum of per-class EGO cubic penalties over K samples of the spline."""
    obstacles = list(obstacles)
    if not obstacles:
        return 0.0
    if K is None:
        K = max(50, 10 * (spline.n + 1))
    ts = np.linspace(spline.t_start, spline.t_end, K)
    pts = spline.eval(ts)
    total = 0.0
    for obs in obstacles:
        params = config[obs.class_name]
        d_safe = params.d_safe
        w = params.weight
        for i in range(K):
            d = obs.signed_dist(pts[i], float(ts[i]))
            diff = d_safe - d
            if diff > 0.0:
                total += w * diff * diff * diff
    return float(total / K)
