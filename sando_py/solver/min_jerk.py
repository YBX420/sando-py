"""Closed-form cubic-Hermite trajectory through the global path's
waypoints. The v1 stand-in for the C++ Gurobi QP solver.

Per-segment polynomial (cubic, matching :class:`PieceWisePol`):
    p(u) = a u^3 + b u^2 + c u + d,    u in [0, T_i]

Boundary conditions per segment:
    p(0)   = waypoint i        p(T) = waypoint i+1
    p'(0)  = v_i               p'(T) = v_{i+1}

Junction velocities ``v_i`` are picked as the centred difference of the
two adjacent waypoints scaled by the sum of adjacent durations (the
classic Catmull-Rom-style tangent), clipped to ``v_max``. The first
velocity is the agent's current ``A.vel``; the last is zero (hover at
goal).

Per-segment duration ``T_i = L_i / (alpha * v_max)`` with a floor so a
near-zero-length segment doesn't blow up the polynomial coefficients.

V1 limitations
--------------
* Cubic per segment ⇒ start *acceleration* is not preserved (only 4
  DOFs per axis; we already spend them on pos+vel at both endpoints).
  This causes a one-tick accel jump at replan boundaries; the C++ also
  re-anchors per replan but matches accel via the quintic-equivalent
  Hermite parameterization in its QP. The v2 Gurobi solver in this
  port will fix it the same way.
* ``polys`` (safety corridors) are accepted for contract parity but
  not enforced. v1 trusts that the global path is already feasible —
  in practice the cubic spline can clip a sharp corner of a corridor.
  v2 turns that into inequality constraints.
* ``v_max`` / ``a_max`` / ``j_max`` are partially respected: junction
  velocities are clamped to ``v_max``, but no per-axis check is done on
  the intermediate part of each cubic. v2 enforces them as constraints.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Tuple

import numpy as np

from sando_py.core import Parameters, PieceWisePol, Polytope, RobotState

from .types import SolverInfo


# Cruise speed = ``_CRUISE_ALPHA * v_max`` — leaves headroom for
# overshoots on corners. Matches the C++ default-ish behaviour.
_CRUISE_ALPHA = 0.5

# Floor on per-segment duration so very short segments don't produce
# numerically explosive cubic coefficients.
_T_MIN = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MinJerkSolver:
    """Stateless solver — stash params, call ``solve`` per replan tick.

    Despite the file name, this is *not* min-jerk: cubic Hermite cannot
    minimize jerk (it has only 4 DOFs per axis). The class is named to
    match the file / README slot; ``v2`` (Gurobi QP) will be the actual
    minimum-jerk variant.
    """

    def __init__(self, params: Parameters) -> None:
        self.params = params

    def solve(
        self,
        A: RobotState,
        path: List[np.ndarray],
        polys: Iterable[Polytope],  # noqa: ARG002  v1 unused, kept for parity
    ) -> Tuple[PieceWisePol, SolverInfo]:
        t0 = time.perf_counter()
        pwp = PieceWisePol()

        if path is None or len(path) < 2:
            return pwp, SolverInfo(
                success=False,
                wall_time_s=time.perf_counter() - t0,
            )

        pts = [np.asarray(p, dtype=float).reshape(3) for p in path]
        Ts = _allocate_times(pts, self.params)
        if not Ts or any(T <= 0 for T in Ts):
            return pwp, SolverInfo(
                success=False,
                wall_time_s=time.perf_counter() - t0,
            )

        vs = _junction_velocities(pts, Ts, A.vel, self.params)

        t_cum = float(A.t)
        pwp.times.append(t_cum)
        total_jerk_sq = 0.0
        for i in range(len(pts) - 1):
            T = Ts[i]
            cx = _hermite_coeffs(pts[i][0], vs[i][0], pts[i + 1][0], vs[i + 1][0], T)
            cy = _hermite_coeffs(pts[i][1], vs[i][1], pts[i + 1][1], vs[i + 1][1], T)
            cz = _hermite_coeffs(pts[i][2], vs[i][2], pts[i + 1][2], vs[i + 1][2], T)
            pwp.coeff_x.append(cx)
            pwp.coeff_y.append(cy)
            pwp.coeff_z.append(cz)
            t_cum += T
            pwp.times.append(t_cum)
            # Jerk = 3rd derivative of cubic = 6 * coeff[0]. Constant per
            # segment, so integral of jerk² over [0, T] = 36 * a² * T.
            total_jerk_sq += 36.0 * (cx[0] ** 2 + cy[0] ** 2 + cz[0] ** 2) * T

        return pwp, SolverInfo(
            success=True,
            cost=float(total_jerk_sq),
            wall_time_s=time.perf_counter() - t0,
        )


def solve(
    A: RobotState,
    path: List[np.ndarray],
    polys: Iterable[Polytope],
    params: Parameters,
) -> Tuple[PieceWisePol, SolverInfo]:
    """Functional shim that matches the README contract verbatim."""
    return MinJerkSolver(params).solve(A, path, polys)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _allocate_times(pts: List[np.ndarray], params: Parameters) -> List[float]:
    v_max = max(0.1, float(params.v_max))
    cruise = max(_T_MIN * v_max, _CRUISE_ALPHA * v_max)
    Ts = []
    for i in range(len(pts) - 1):
        L = float(np.linalg.norm(pts[i + 1] - pts[i]))
        Ts.append(max(_T_MIN, L / cruise))
    return Ts


def _junction_velocities(
    pts: List[np.ndarray],
    Ts: List[float],
    v_start,
    params: Parameters,
) -> List[np.ndarray]:
    """Per-waypoint velocities for the cubic Hermite fit.

    The boundary waypoints are clamped: ``v[0]`` to the agent's current
    velocity, ``v[-1]`` to zero (hover at goal). Interior waypoints get
    a centred-difference tangent.
    """
    n = len(pts)
    v_max = max(0.0, float(params.v_max))

    vs: List[np.ndarray] = [np.zeros(3) for _ in range(n)]
    vs[0] = np.asarray(v_start, dtype=float).reshape(3).copy()
    if v_max > 0:
        vs[0] = _clip_norm(vs[0], v_max)
    # vs[-1] left at zeros — hover at terminal goal.

    for i in range(1, n - 1):
        dt = Ts[i - 1] + Ts[i]
        if dt <= 1e-9:
            continue
        v = (pts[i + 1] - pts[i - 1]) / dt
        if v_max > 0:
            v = _clip_norm(v, v_max)
        vs[i] = v
    return vs


def _hermite_coeffs(p0: float, v0: float, p1: float, v1: float, T: float) -> np.ndarray:
    """Return the (a, b, c, d) coefficients of ``a u^3 + b u^2 + c u + d``
    matching ``p(0) = p0, p'(0) = v0, p(T) = p1, p'(T) = v1``.

    Derived from the standard cubic-Hermite formulas; verified against
    the boundary conditions in :func:`tests.test_solver`.
    """
    T2 = T * T
    T3 = T2 * T
    dp = p1 - p0
    a = (-2.0 * dp + (v0 + v1) * T) / T3
    b = (3.0 * dp - (2.0 * v0 + v1) * T) / T2
    c = v0
    d = p0
    return np.array([a, b, c, d], dtype=float)


def _clip_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= vmax or n <= 1e-12:
        return v
    return v * (vmax / n)
