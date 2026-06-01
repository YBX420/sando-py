"""MINCO minimum-jerk (s=3, quintic) trajectory backbone.

Replacement for the B-spline backbone of the local optimizer. MINCO
(Wang et al., IEEE T-RO 2022, GCOPTER framework) parameterizes a piecewise
quintic by intermediate waypoints q and segment durations T; given (q, T) the
minimum-jerk trajectory is UNIQUE and is recovered by solving a sparse banded
linear system  M(T) c = b(q)  (one row per scalar constraint, 6M rows).

Conventions (mirrors GCOPTER MINCO_S3NU exactly):
  M segments, segment i is  p_i(t) = sum_{j=0..5} c[6i+j] * t^j,  t in [0, T_i]
  coefficients c are stored ASCENDING in power (c[6i+0] is the constant term).
  banded system has lower/upper bandwidth 6.

Constraint row layout (6M rows total):
  rows 0,1,2          : p(0)=p_start, v(0)=v0, a(0)=a0      (head block)
  per junction i in 1..M-1  (loop var i = 0..M-2):
    6i+3 : C^3 (jerk)  continuity   p_i'''(T_i)  = p_{i+1}'''(0)
    6i+4 : C^4 (snap)  continuity   p_i''''(T_i) = p_{i+1}''''(0)
    6i+5 : p_i(T_i) = q_i           (EXACT waypoint interpolation)
    6i+6 : C^0 (pos)   continuity   p_i(T_i)  = p_{i+1}(0)
    6i+7 : C^1 (vel)   continuity   p_i'(T_i) = p_{i+1}'(0)
    6i+8 : C^2 (acc)   continuity   p_i''(T_i)= p_{i+1}''(0)
  rows 6M-3,-2,-1     : p(T_M)=p_goal, v=vf, a=af           (tail block)

eval_deriv(ts, order) mirrors UniformBSpline.eval_deriv: order 0 = position.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# Power -> Bernstein conversion for a degree-5 polynomial on [0, 1].
# A degree-5 polynomial p(tau) = sum_{j=0..5} a_j tau^j on tau in [0,1] equals
# sum_{k=0..5} P_k Bernstein_k(tau), Bernstein_k = C(5,k) tau^k (1-tau)^(5-k).
# The control points are P = C2B @ a with the lower-triangular constant matrix
#   C2B[k, j] = C(k, j) / C(5, j)   for j <= k else 0
# (verified eval-equivalent to 3e-15; P_0 = a_0 = p(0), P_5 = sum a_j = p(1)).
# Bernstein control points satisfy the convex-hull property: the whole curve
# lies in conv{P_k}, which is the continuous-time clearance certificate basis.
# ---------------------------------------------------------------------------
C2B = (1.0 / 60.0) * np.array([
    [60.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [60.0, 12.0,  0.0,  0.0,  0.0,  0.0],
    [60.0, 24.0,  6.0,  0.0,  0.0,  0.0],
    [60.0, 36.0, 18.0,  6.0,  0.0,  0.0],
    [60.0, 48.0, 36.0, 24.0, 12.0,  0.0],
    [60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
], dtype=np.float64)


class MinjerkTraj:
    """Minimum-jerk MINCO trajectory, s=3 (quintic segments), 3D.

    Build from full waypoints (start, intermediates..., goal) and per-segment
    durations. Boundary derivatives (v0,a0 at start, vf,af at goal) default to
    zero; pass arrays to override.
    """

    DEG = 5          # quintic
    NC = 6           # coefficients per segment

    def __init__(
        self,
        waypoints_full: np.ndarray,
        durations: np.ndarray,
        v0: Optional[np.ndarray] = None,
        a0: Optional[np.ndarray] = None,
        vf: Optional[np.ndarray] = None,
        af: Optional[np.ndarray] = None,
    ):
        wp = np.asarray(waypoints_full, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] != 3:
            raise ValueError(f"waypoints_full must be (M+1, 3), got {wp.shape}")
        T = np.asarray(durations, dtype=np.float64).reshape(-1)
        M = T.size
        if wp.shape[0] != M + 1:
            raise ValueError(
                f"need M+1 waypoints for M durations: got {wp.shape[0]} pts, {M} durations"
            )
        if M < 1:
            raise ValueError("need at least one segment")
        if np.any(T <= 0.0):
            raise ValueError("all durations T_i must be > 0")

        self.M = M
        self.d = 3
        self.T = T
        self.waypoints = wp
        self.p_start = wp[0].copy()
        self.p_goal = wp[-1].copy()
        self.q = wp[1:-1].copy()  # (M-1, 3) interior waypoints

        z = np.zeros(3)
        self.v0 = z.copy() if v0 is None else np.asarray(v0, dtype=np.float64).reshape(3)
        self.a0 = z.copy() if a0 is None else np.asarray(a0, dtype=np.float64).reshape(3)
        self.vf = z.copy() if vf is None else np.asarray(vf, dtype=np.float64).reshape(3)
        self.af = z.copy() if af is None else np.asarray(af, dtype=np.float64).reshape(3)

        self.t_start = 0.0
        self.t_end = float(np.sum(T))
        # cumulative segment-start times, length M+1; last = t_end
        self._cum = np.concatenate([[0.0], np.cumsum(T)])

        self._solve()

    # ---- alternative constructor ----------------------------------------

    @classmethod
    def from_endpoints(
        cls,
        start: np.ndarray,
        goal: np.ndarray,
        q: np.ndarray,
        durations: np.ndarray,
        v0=None, a0=None, vf=None, af=None,
    ) -> "MinjerkTraj":
        """Build from explicit start, goal, interior waypoints q (M-1, 3)."""
        start = np.asarray(start, dtype=np.float64).reshape(1, 3)
        goal = np.asarray(goal, dtype=np.float64).reshape(1, 3)
        q = np.asarray(q, dtype=np.float64)
        if q.size == 0:
            wp = np.vstack([start, goal])
        else:
            wp = np.vstack([start, q.reshape(-1, 3), goal])
        return cls(wp, durations, v0=v0, a0=a0, vf=vf, af=af)

    # ---- banded M(T) construction + solve --------------------------------

    def _build_banded(self):
        """Build M(T) in scipy banded storage (l=u=6) plus dense version.

        Returns (ab, b) where ab is (13, 6M) banded form for solve_banded and
        b is the (6M, 3) RHS.  ab[u + i - j, j] = A[i, j].
        Also stash a sparse/dense A for testing introspection.
        """
        M, NC = self.M, self.NC
        n = NC * M
        l = u = 6
        ab = np.zeros((l + u + 1, n), dtype=np.float64)
        b = np.zeros((n, 3), dtype=np.float64)
        # keep a COO-style record for an exact dense reconstruction in tests
        rows, cols, vals = [], [], []

        def setA(i, j, v):
            ab[u + i - j, j] = v
            rows.append(i); cols.append(j); vals.append(v)

        T1 = self.T
        T2 = T1 * T1
        T3 = T2 * T1
        T4 = T2 * T2
        T5 = T4 * T1

        # head block: p(0)=p_start, v(0)=v0, a(0)=a0
        setA(0, 0, 1.0)
        setA(1, 1, 1.0)
        setA(2, 2, 2.0)
        b[0] = self.p_start
        b[1] = self.v0
        b[2] = self.a0

        # interior junctions
        for i in range(M - 1):
            base = 6 * i
            setA(base + 3, base + 3, 6.0)
            setA(base + 3, base + 4, 24.0 * T1[i])
            setA(base + 3, base + 5, 60.0 * T2[i])
            setA(base + 3, base + 9, -6.0)

            setA(base + 4, base + 4, 24.0)
            setA(base + 4, base + 5, 120.0 * T1[i])
            setA(base + 4, base + 10, -24.0)

            setA(base + 5, base + 0, 1.0)
            setA(base + 5, base + 1, T1[i])
            setA(base + 5, base + 2, T2[i])
            setA(base + 5, base + 3, T3[i])
            setA(base + 5, base + 4, T4[i])
            setA(base + 5, base + 5, T5[i])

            setA(base + 6, base + 0, 1.0)
            setA(base + 6, base + 1, T1[i])
            setA(base + 6, base + 2, T2[i])
            setA(base + 6, base + 3, T3[i])
            setA(base + 6, base + 4, T4[i])
            setA(base + 6, base + 5, T5[i])
            setA(base + 6, base + 6, -1.0)

            setA(base + 7, base + 1, 1.0)
            setA(base + 7, base + 2, 2.0 * T1[i])
            setA(base + 7, base + 3, 3.0 * T2[i])
            setA(base + 7, base + 4, 4.0 * T3[i])
            setA(base + 7, base + 5, 5.0 * T4[i])
            setA(base + 7, base + 7, -1.0)

            setA(base + 8, base + 2, 2.0)
            setA(base + 8, base + 3, 6.0 * T1[i])
            setA(base + 8, base + 4, 12.0 * T2[i])
            setA(base + 8, base + 5, 20.0 * T3[i])
            setA(base + 8, base + 8, -2.0)

            b[base + 5] = self.q[i]

        # tail block: p(T_M)=p_goal, v=vf, a=af  (segment M-1)
        last = M - 1
        bN = 6 * M
        setA(bN - 3, bN - 6, 1.0)
        setA(bN - 3, bN - 5, T1[last])
        setA(bN - 3, bN - 4, T2[last])
        setA(bN - 3, bN - 3, T3[last])
        setA(bN - 3, bN - 2, T4[last])
        setA(bN - 3, bN - 1, T5[last])

        setA(bN - 2, bN - 5, 1.0)
        setA(bN - 2, bN - 4, 2.0 * T1[last])
        setA(bN - 2, bN - 3, 3.0 * T2[last])
        setA(bN - 2, bN - 2, 4.0 * T3[last])
        setA(bN - 2, bN - 1, 5.0 * T4[last])

        setA(bN - 1, bN - 4, 2.0)
        setA(bN - 1, bN - 3, 6.0 * T1[last])
        setA(bN - 1, bN - 2, 12.0 * T2[last])
        setA(bN - 1, bN - 1, 20.0 * T3[last])

        b[bN - 3] = self.p_goal
        b[bN - 2] = self.vf
        b[bN - 1] = self.af

        self._coo = (np.asarray(rows), np.asarray(cols), np.asarray(vals))
        return ab, b

    def _solve(self):
        ab, b = self._build_banded()
        # banded LU solve, O(M) for fixed bandwidth
        c = solve_banded((6, 6), ab, b)
        self.c = np.ascontiguousarray(c)  # (6M, 3)
        self._ab = ab
        self._b = b

    # ---- dense / sparse introspection for tests --------------------------

    def dense_M(self) -> np.ndarray:
        """Reconstruct the full dense M(T) from the recorded entries."""
        n = self.NC * self.M
        A = np.zeros((n, n), dtype=np.float64)
        r, c, v = self._coo
        A[r, c] = v
        return A

    @property
    def banded_M(self) -> np.ndarray:
        """The banded storage (13, 6M) passed to solve_banded."""
        return self._ab

    # ---- domain helpers --------------------------------------------------

    def _locate(self, t: float):
        """Return (segment index i, local tau in [0, T_i]) for global time t."""
        t = float(np.clip(t, self.t_start, self.t_end))
        if t >= self.t_end:
            i = self.M - 1
            return i, self.T[i]
        # find i with cum[i] <= t < cum[i+1]
        i = int(np.searchsorted(self._cum, t, side="right") - 1)
        if i < 0:
            i = 0
        if i > self.M - 1:
            i = self.M - 1
        tau = t - self._cum[i]
        return i, tau

    # ---- evaluation ------------------------------------------------------

    @staticmethod
    def _basis(tau: float, order: int) -> np.ndarray:
        """Powers basis row for the `order`-th derivative of [1,t,...,t^5].

        d^o/dt^o (t^j) = j!/(j-o)! * t^(j-o) for j>=o else 0.
        """
        row = np.zeros(6, dtype=np.float64)
        for j in range(order, 6):
            coeff = 1.0
            for k in range(order):
                coeff *= (j - k)
            row[j] = coeff * (tau ** (j - order))
        return row

    def eval_deriv(self, t, order: int = 0):
        """`order`-th derivative at t (scalar or 1-D array). Returns (d,) or (K,d).

        order 0 = position; matches UniformBSpline.eval_deriv semantics.
        """
        if order < 0:
            raise ValueError("order must be >= 0")
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if order > self.DEG:
            out = np.zeros((ts.size, self.d), dtype=np.float64)
        else:
            # vectorised _locate: clip, segment via searchsorted, local tau
            tc = np.clip(ts, self.t_start, self.t_end)
            idx = np.searchsorted(self._cum, tc, side="right") - 1
            idx = np.clip(idx, 0, self.M - 1)
            tau = tc - self._cum[idx]                      # (K,)
            # vectorised _basis row per tau: col p = (p!/(p-o)!) tau^(p-o)
            B = np.zeros((ts.size, 6), dtype=np.float64)
            for p in range(order, 6):
                coeff = 1.0
                for m in range(order):
                    coeff *= (p - m)
                B[:, p] = coeff * (tau ** (p - order))     # same `**` for bit-parity
            cseg = self.c.reshape(self.M, 6, self.d)[idx]  # (K,6,d) gathered coeffs
            out = np.einsum("kj,kjd->kd", B, cseg)         # row @ ci per point
        scalar = np.isscalar(t) or np.ndim(t) == 0
        return out[0] if scalar else out

    def eval(self, t):
        """Position at t. Alias of eval_deriv(t, 0) for B-spline parity."""
        return self.eval_deriv(t, 0)

    # =====================================================================
    # Bernstein control points (Stage 3 — continuous-time clearance basis)
    # =====================================================================
    #
    # Segment i is p_i(t) = sum_j c[6i+j] t^j on the LOCAL clock t in [0,T_i].
    # Reparametrize t = tau*T_i, tau in [0,1]:  a_{i,j} = c[6i+j] * T_i^j, so
    # p_i(tau) = sum_j a_{i,j} tau^j.  Control points  P_i = C2B @ a_i (6,3).
    # The T_i^j scaling makes P_i depend on T_i EXPLICITLY (besides implicitly
    # via c = M(T)^-1 b); the explicit derivative is dP_i/dT_i = C2B @ dD/dT_i @ c_i
    # with dD/dT_i = diag([0,1,2T_i,3T_i^2,4T_i^3,5T_i^4]).

    @staticmethod
    def _D_powers(Ti: float) -> np.ndarray:
        """diag scaling vector [1,T,T^2,T^3,T^4,T^5] for one segment."""
        return np.array([1.0, Ti, Ti**2, Ti**3, Ti**4, Ti**5], dtype=np.float64)

    @staticmethod
    def _dD_powers(Ti: float) -> np.ndarray:
        """d/dT of the diag scaling: [0,1,2T,3T^2,4T^3,5T^4]."""
        return np.array([0.0, 1.0, 2.0 * Ti, 3.0 * Ti**2,
                         4.0 * Ti**3, 5.0 * Ti**4], dtype=np.float64)

    def control_points(self) -> np.ndarray:
        """Bernstein control points of every segment, shape (M, 6, 3).

        P[i,k,:] is the k-th degree-5 Bezier control point of segment i on the
        normalized domain tau in [0,1]. The continuous curve of segment i lies
        in conv{P[i,0..5]} (convex-hull property).  Vectorized: a = D(T)c per
        segment then P = C2B @ a as a single batched matmul."""
        M = self.M
        cseg = self.c.reshape(M, 6, 3)                    # (M,6,3) per-segment coeffs
        T = self.T
        Dvec = np.stack([np.ones(M), T, T**2, T**3, T**4, T**5], axis=1)  # (M,6)
        a = Dvec[:, :, None] * cseg                       # (M,6,3) = D(T) c
        return np.einsum("kj,mjd->mkd", C2B, a)           # (M,6,3) = C2B @ a

    def control_point_times(self) -> np.ndarray:
        """Wall-clock time of every Bernstein control point, shape (M, 6).

        Bernstein control point k of segment i sits at normalized tau_k = k/5,
        i.e. local time (k/5)*T_i, so its absolute time relative to the replan
        instant is t_{i,k} = _cum[i] + (k/5)*T_i.  Single source of truth for
        the space-time ALM: t[:,0]==_cum[:M] (segment starts) and
        t[:,5]==_cum[1:M+1] (segment ends == the Stage-3 _seg_rep_time)."""
        M = self.M
        kfrac = (np.arange(6) / float(self.DEG))[None, :]   # (1,6) = k/5
        return self._cum[:M, None] + kfrac * self.T[:, None]   # (M,6)

    def control_points_dT_explicit(self) -> np.ndarray:
        """Explicit dP_i/dT_i (holding c fixed), shape (M, 6, 3).

        dP_i/dT_i = C2B @ (dD/dT_i) @ c_i.  This is the explicit-T piece that
        the dCost/dT chain MUST include (dropping it reproduces the M2 trap)."""
        M = self.M
        cseg = self.c.reshape(M, 6, 3)
        T = self.T
        dDvec = np.stack([np.zeros(M), np.ones(M), 2.0 * T, 3.0 * T**2,
                          4.0 * T**3, 5.0 * T**4], axis=1)  # (M,6)
        da = dDvec[:, :, None] * cseg                       # (M,6,3)
        return np.einsum("kj,mjd->mkd", C2B, da)

    # =====================================================================
    # M1 — analytic gradient through the MINCO map  M(T) c = b(q)
    # =====================================================================
    #
    # Control cost (canonical MINCO energy):  J = integral ||jerk||^2 dt.
    # For segment i, p_i(t)=sum_j c[6i+j] t^j (ASCENDING), jerk=p'''=
    #   6 c3 + 24 c4 t + 60 c5 t^2  (c3,c4,c5 == c[6i+3],c[6i+4],c[6i+5]).
    #   J_i = 720 T^5 c5.c5 + 720 T^4 c4.c5 + 240 T^3 c3.c5
    #         + 192 T^3 c4.c4 + 144 T^2 c3.c4 + 36 T c3.c3
    # (derived by symbolic integration; matches GCOPTER MINCO_S3NU::getEnergy
    #  with its internal ascending coefficient order). The . is dot over the
    #  3 spatial dims. dJ/dc and the explicit dJ/dT below are the symbolic
    #  partials of this expression; verified row-by-row against sympy and
    #  end-to-end against central finite differences in the M1 tests.
    #
    # Adjoint backprop (chain rule through M(T) c = b):
    #   solve  M^T lambda = dJ/dc
    #   dJ/dq_i = lambda[6i+5]           (q_i sits in RHS row 6i+5)
    #   dJ/dT_k = (explicit energy dJ/dT_k) - lambda^T (dM/dT_k) c
    # Mirrors GCOPTER getGrad / propogateGrad (solveAdj == solve M^T).

    def energy(self) -> float:
        """Total jerk-squared integral of the solved trajectory (scalar)."""
        return float(self._energy_and_gradc()[0])

    def _energy_and_gradc(self):
        """Return (J, dJ/dc) with dJ/dc shape (6M, 3). Closed form."""
        M = self.M
        c = self.c
        T = self.T
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        T5 = T4 * T
        gdC = np.zeros_like(c)
        J = 0.0
        for i in range(M):
            b = 6 * i
            c3 = c[b + 3]
            c4 = c[b + 4]
            c5 = c[b + 5]
            J += (36.0 * T[i] * c3.dot(c3)
                  + 144.0 * T2[i] * c3.dot(c4)
                  + 192.0 * T3[i] * c4.dot(c4)
                  + 240.0 * T3[i] * c3.dot(c5)
                  + 720.0 * T4[i] * c4.dot(c5)
                  + 720.0 * T5[i] * c5.dot(c5))
            gdC[b + 3] = 72.0 * T[i] * c3 + 144.0 * T2[i] * c4 + 240.0 * T3[i] * c5
            gdC[b + 4] = 144.0 * T2[i] * c3 + 384.0 * T3[i] * c4 + 720.0 * T4[i] * c5
            gdC[b + 5] = 240.0 * T3[i] * c3 + 720.0 * T4[i] * c4 + 1440.0 * T5[i] * c5
        return J, gdC

    def _energy_grad_time_explicit(self) -> np.ndarray:
        """Explicit dJ/dT_i (holding c fixed), shape (M,). Symbolic partials."""
        M = self.M
        c = self.c
        T = self.T
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        gdT = np.zeros(M, dtype=np.float64)
        for i in range(M):
            b = 6 * i
            c3 = c[b + 3]
            c4 = c[b + 4]
            c5 = c[b + 5]
            gdT[i] = (36.0 * c3.dot(c3)
                      + 288.0 * T[i] * c3.dot(c4)
                      + 576.0 * T2[i] * c4.dot(c4)
                      + 720.0 * T2[i] * c3.dot(c5)
                      + 2880.0 * T3[i] * c4.dot(c5)
                      + 3600.0 * T4[i] * c5.dot(c5))
        return gdT

    # ---- adjoint solve M^T x = rhs ---------------------------------------

    def _solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Solve  M(T)^T x = rhs  (rhs (6M,3) -> x (6M,3)) via banded LU.

        Reuses the same banded factor structure; M^T also has bandwidth (6,6)
        so this stays O(M).
        """
        n = self.NC * self.M
        l = u = 6
        ab = self._ab  # storage of A:  ab[u + i - j, j] = A[i, j]
        # banded storage of A^T:  abT[u + i - j, j] = A[j, i] = A_T[i, j]
        # A[j,i] = ab[u + j - i, i]  ->  abT[u + i - j, j] = ab[u + j - i, i]
        abT = np.zeros_like(ab)
        for i in range(n):
            jlo = max(0, i - l)
            jhi = min(n - 1, i + u)
            for j in range(jlo, jhi + 1):
                abT[u + i - j, j] = ab[u + j - i, i]
        return solve_banded((l, u), abT, rhs)

    # ---- dM/dT_k applied to c  -------------------------------------------

    def _dM_dT_times_c(self, k: int) -> np.ndarray:
        """Return the sparse vector  (dM/dT_k) c  of shape (6M, 3).

        Only the rows of M that contain T_k are nonzero. For an interior
        segment k < M-1 these are the junction rows 6k+3..6k+8; for k==M-1
        they are the three tail rows. Each entry is d(M_row)/dT_k . c.
        """
        n = self.NC * self.M
        c = self.c
        T = self.T[k]
        out = np.zeros((n, 3), dtype=np.float64)
        if k < self.M - 1:
            b = 6 * k
            ci = c[b:b + 6]  # coeffs of segment k (the one ending at T_k)
            # d/dT of the segment-k basis rows (see header derivation):
            # jerk-cont row 6k+3: [6,24T,60T^2] -> d: [0,24,120T] on cols 3,4,5
            out[b + 3] = 24.0 * ci[4] + 120.0 * T * ci[5]
            # snap-cont row 6k+4: [24,120T] -> d: [0,120] on cols 4,5
            out[b + 4] = 120.0 * ci[5]
            # waypoint row 6k+5: [1,T,T^2,T^3,T^4,T^5] -> d:[0,1,2T,3T^2,4T^3,5T^4]
            out[b + 5] = (ci[1] + 2.0 * T * ci[2] + 3.0 * T**2 * ci[3]
                          + 4.0 * T**3 * ci[4] + 5.0 * T**4 * ci[5])
            # C0 pos row 6k+6: same pos basis (the -c_{k+1,0} term is T-free)
            out[b + 6] = out[b + 5]
            # C1 vel row 6k+7: [1,2T,3T^2,4T^3,5T^4] -> d:[0,2,6T,12T^2,20T^3]
            out[b + 7] = (2.0 * ci[2] + 6.0 * T * ci[3] + 12.0 * T**2 * ci[4]
                          + 20.0 * T**3 * ci[5])
            # C2 acc row 6k+8: [2,6T,12T^2,20T^3] -> d:[0,6,24T,60T^2]
            out[b + 8] = 6.0 * ci[3] + 24.0 * T * ci[4] + 60.0 * T**2 * ci[5]
        else:
            b = 6 * (self.M - 1)
            ci = c[b:b + 6]
            bN = n
            # tail pos row bN-3: pos basis -> [0,1,2T,3T^2,4T^3,5T^4]
            out[bN - 3] = (ci[1] + 2.0 * T * ci[2] + 3.0 * T**2 * ci[3]
                           + 4.0 * T**3 * ci[4] + 5.0 * T**4 * ci[5])
            # tail vel row bN-2: vel basis -> [0,2,6T,12T^2,20T^3]
            out[bN - 2] = (2.0 * ci[2] + 6.0 * T * ci[3] + 12.0 * T**2 * ci[4]
                           + 20.0 * T**3 * ci[5])
            # tail acc row bN-1: acc basis -> [0,6,24T,60T^2]
            out[bN - 1] = 6.0 * ci[3] + 24.0 * T * ci[4] + 60.0 * T**2 * ci[5]
        return out

    # ---- generic backprop hook -------------------------------------------

    def grad_from_dcost_dc(self, dcost_dc: np.ndarray,
                           dcost_dT_explicit: Optional[np.ndarray] = None):
        """Backprop an external dCost/dc (and optional explicit dCost/dT).

        Given the partial gradient of ANY scalar cost w.r.t. the trajectory
        coefficients c (shape (6M,3)) -- e.g. obtained by sampling an obstacle
        cost on the trajectory and accumulating its sensitivity onto c -- and
        optionally the explicit partial dCost/dT (shape (M,), the part of the
        cost that depends on T directly, not through c), return:

            (dCost/dq, dCost/dT)   shapes ((M-1,3), (M,))

        This is the chain rule through the MINCO map M(T) c = b(q):
            M^T lambda = dCost/dc
            dCost/dq_i = lambda[6i+5]
            dCost/dT_k = dCost/dT_k|explicit - lambda^T (dM/dT_k) c
        """
        dcost_dc = np.asarray(dcost_dc, dtype=np.float64)
        if dcost_dc.shape != (self.NC * self.M, 3):
            raise ValueError(
                f"dcost_dc must be ({self.NC * self.M}, 3), got {dcost_dc.shape}")
        M = self.M
        lam = self._solve_adjoint(dcost_dc)  # (6M, 3)

        # waypoint gradient: q_i lives in RHS row 6i+5 (junction loop i=0..M-2)
        gq = np.zeros((M - 1, 3), dtype=np.float64)
        for i in range(M - 1):
            gq[i] = lam[6 * i + 5]

        gT = np.zeros(M, dtype=np.float64)
        if dcost_dT_explicit is not None:
            gT += np.asarray(dcost_dT_explicit, dtype=np.float64).reshape(M)
        for k in range(M):
            dMc = self._dM_dT_times_c(k)  # (6M, 3)
            gT[k] -= float(np.sum(lam * dMc))
        return gq, gT

    def energy_grad(self):
        """Analytic gradient of the jerk-energy cost J.

        Returns (J, dJ/dq, dJ/dT) with shapes (scalar, (M-1,3), (M,)).
        Combines the closed-form energy partials with the MINCO adjoint.
        """
        J, gdC = self._energy_and_gradc()
        gdT_explicit = self._energy_grad_time_explicit()
        gq, gT = self.grad_from_dcost_dc(gdC, dcost_dT_explicit=gdT_explicit)
        return J, gq, gT
