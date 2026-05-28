"""3D uniform B-spline geometry + LSQ fit from a polyline.

Convention:
  degree p   (default 5, quintic — C^{p-1} continuous)
  ctrl       (N+1, d)  control points P_0..P_N in R^d (d=3 here)
  uniform knot spacing dt:   t_i = (i - p) * dt   for i in 0..N+p+1
  valid parameter domain     t in [t_p, t_{N+1}], length T = (N - p + 1) * dt
  evaluation                 De Boor recursion (Wikipedia indexing)
  k-th derivative            itself a uniform B-spline of degree p-k with
                             control points Q_i^{(1)} = (P_{i+1} - P_i) / dt
                             (the uniform-knot identity; recurse for higher k)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class UniformBSpline:
    def __init__(self, ctrl: np.ndarray, degree: int = 5, dt: float = 1.0):
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.ndim != 2:
            raise ValueError(f"ctrl must be (N+1, d), got shape {ctrl.shape}")
        if int(degree) < 1:
            raise ValueError(f"degree must be ≥ 1, got {degree}")
        if float(dt) <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.ctrl = ctrl
        self.p = int(degree)
        self.dt = float(dt)
        self.n = ctrl.shape[0] - 1  # last index of P
        self.d = ctrl.shape[1]
        if self.n < self.p:
            raise ValueError(f"need ≥ p+1={self.p+1} control points, got {self.n+1}")
        self.m = self.n + self.p + 1  # last knot index
        # uniform knots so that t_p = 0 (clean parameter origin)
        self.knots = (np.arange(self.m + 1) - self.p) * self.dt
        self.t_start = self.knots[self.p]
        self.t_end = self.knots[self.n + 1]
        self.T = self.t_end - self.t_start

    # ---- domain helpers ---------------------------------------------------

    def _find_span(self, t: float) -> int:
        t = float(np.clip(t, self.t_start, self.t_end))
        # t == t_end belongs to the last valid span [t_n, t_{n+1}]
        if t >= self.t_end - 1e-12:
            return self.n
        # binary search for k with knots[k] <= t < knots[k+1], k in [p, n]
        lo, hi = self.p, self.n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.knots[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ---- evaluation -------------------------------------------------------

    def eval(self, t):
        """Position at t (scalar or 1-D array). Returns (d,) or (K, d)."""
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((ts.size, self.d), dtype=np.float64)
        for i, ti in enumerate(ts):
            out[i] = self._de_boor(ti, self.ctrl, self.p, self.knots)
        return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out

    def eval_deriv(self, t, order: int = 1):
        """k-th derivative at t. Returns same shape as eval."""
        if order < 0:
            raise ValueError("order must be ≥ 0")
        if order == 0:
            return self.eval(t)
        if order > self.p:
            zero = np.zeros(self.d)
            ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
            return zero if (np.isscalar(t) or np.ndim(t) == 0) else np.tile(zero, (ts.size, 1))
        # build derivative control points + degree + knots via the uniform identity
        Q = self.ctrl
        knots = self.knots
        p = self.p
        for _ in range(order):
            Q = (Q[1:] - Q[:-1]) / self.dt
            knots = knots[1:-1]
            p -= 1
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((ts.size, self.d), dtype=np.float64)
        for i, ti in enumerate(ts):
            out[i] = self._de_boor_with(ti, Q, p, knots, self.t_start, self.t_end)
        return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out

    # ---- nonzero basis (for LSQ assembly) --------------------------------

    def nonzero_basis(self, t: float) -> Tuple[int, np.ndarray]:
        """Return (span k, basis values [N_{k-p,p}(t) .. N_{k,p}(t)])."""
        t = float(np.clip(t, self.t_start, self.t_end))
        k = self._find_span(t)
        N = np.zeros(self.p + 1, dtype=np.float64)
        N[0] = 1.0
        left = np.zeros(self.p + 1)
        right = np.zeros(self.p + 1)
        for j in range(1, self.p + 1):
            left[j] = t - self.knots[k + 1 - j]
            right[j] = self.knots[k + j] - t
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = N[r] / denom if denom != 0 else 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        return k, N

    # ---- endpoint clamping (geometric) -----------------------------------

    def lock_endpoint(self, side: str, position: np.ndarray) -> None:
        """Pin the endpoint to `position` with zero velocity / accel / … up to
        (p-1)-th derivative — by setting the first or last p+1 control points
        all equal to `position` (geometric identity for uniform B-splines).
        """
        position = np.asarray(position, dtype=np.float64).reshape(self.d)
        if side == "start":
            self.ctrl[: self.p + 1] = position
        elif side == "end":
            self.ctrl[-(self.p + 1) :] = position
        else:
            raise ValueError("side must be 'start' or 'end'")

    # ---- LSQ fit from a polyline -----------------------------------------

    @classmethod
    def fit_path(
        cls,
        path: np.ndarray,
        num_ctrl: int,
        degree: int = 5,
        dt: float = 1.0,
        *,
        clamp_endpoints: bool = False,
    ) -> "UniformBSpline":
        """Least-squares fit a uniform B-spline to a polyline.

        path: (M, d) waypoints; treated as arc-length samples mapped uniformly
              onto the valid domain [t_p, t_{N+1}].
        num_ctrl: number of control points (must be ≥ degree+1).
        clamp_endpoints: after LSQ, pin both endpoints to path[0] / path[-1]
              with zero derivatives (via lock_endpoint).
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2:
            raise ValueError(f"path must be (M, d), got shape {path.shape}")
        M, d = path.shape
        N = num_ctrl - 1
        if N < degree:
            raise ValueError(f"num_ctrl ≥ degree+1={degree+1}, got {num_ctrl}")
        if clamp_endpoints and num_ctrl < 2 * (degree + 1):
            # locking both ends needs disjoint first/last p+1 ctrl points
            raise ValueError(
                f"clamp_endpoints needs num_ctrl ≥ 2*(degree+1)={2 * (degree + 1)}, got {num_ctrl}"
            )
        # build a temporary spline with zeros to use its basis machinery
        bs = cls(np.zeros((num_ctrl, d)), degree=degree, dt=dt)
        ts = np.linspace(bs.t_start, bs.t_end, M)
        # assemble (M x (N+1)) basis matrix; each row has only p+1 nonzeros
        B = np.zeros((M, N + 1), dtype=np.float64)
        for i, ti in enumerate(ts):
            k, vals = bs.nonzero_basis(ti)
            B[i, k - degree : k + 1] = vals
        ctrl, *_ = np.linalg.lstsq(B, path, rcond=None)
        bs.ctrl = ctrl
        if clamp_endpoints:
            bs.lock_endpoint("start", path[0])
            bs.lock_endpoint("end", path[-1])
        return bs

    # ---- internal: De Boor on (ctrl, p, knots) ---------------------------

    def _de_boor(self, t: float, ctrl: np.ndarray, p: int, knots: np.ndarray) -> np.ndarray:
        return self._de_boor_with(t, ctrl, p, knots, self.t_start, self.t_end)

    @staticmethod
    def _de_boor_with(t, ctrl, p, knots, t_start, t_end):
        t = float(np.clip(t, t_start, t_end))
        n_local = ctrl.shape[0] - 1
        if t >= t_end - 1e-12:
            k = n_local
        else:
            # binary search in [p, n_local]
            lo, hi = p, n_local
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if knots[mid] <= t:
                    lo = mid
                else:
                    hi = mid - 1
            k = lo
        d = np.array([ctrl[k - p + j] for j in range(p + 1)], dtype=np.float64)
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i_lo = k - p + j  # knot index t_{j+k-p}
                # alpha = (t - knots[i_lo]) / (knots[i_lo + p + 1 - r] - knots[i_lo])
                denom = knots[i_lo + p + 1 - r] - knots[i_lo]
                alpha = 0.0 if denom == 0 else (t - knots[i_lo]) / denom
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        return d[p]
