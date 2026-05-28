"""Gurobi local-trajectory solver — Python port of src/sando/gurobi_solver.cpp.

Decision variables (SANDO mode, with variable elimination):
  d_k_axis  ∈ R for axis ∈ {0,1,2}, k ∈ {3, 4, ..., N-1}
            — the constant coefficient (== start-of-segment position) of
              segments t=3..N-1. 3*(N-3) free continuous variables total.
  b[t][p]   ∈ {0, 1}, polytope-assignment binaries for every segment t and
              spatial polytope p. Σ_p b[t][p] >= 1 per segment.

Constraints:
  * The 4N power-basis coefficients (a_t, b_t, c_t, d_t) per axis are written
    as linear expressions in (P0, V0, A0, Pf, Vf, Af, d_k_axis) via a one-time
    numerical elimination of the 4N×4N continuity + boundary linear system.
  * Indicator constraints (Gurobi internally compiles to big-M):
        b[t][p] == 1  ⇒  A_p · CP_j(t) ≤ bb_p   for each Bezier position CP j.
  * Map-bound box on the 4 Bezier position CPs per segment.
  * Dynamic Linf / L1 / L2 limits on Bezier velocity / acceleration / jerk
    control points.

Objective:
  minimize  jerk_smooth_weight * Σ_t Σ_axis (6 * a_t_axis)²

Time allocation:
  dt_[t] = factor * max(initial_dt, 2 * dc)    (uniform across segments)
  initial_dt = max axis time / N, where per-axis time is the worst of
  velocity-dash / accel-ramp / jerk-cubic single-axis root.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import BasisConverter, Parameters, PieceWisePol, Polytope, RobotState


# Gurobi import is deferred — we want `import sando_py` to work even on systems
# without a Gurobi license.
def _import_gurobi():
    try:
        import gurobipy as gp  # noqa: F401
        from gurobipy import GRB  # noqa: F401
    except ImportError as ex:
        raise RuntimeError(
            "gurobipy is required to run the Gurobi solver. Install with "
            "`pip install gurobipy` and ensure a valid license is available."
        ) from ex
    return gp, GRB


# ---------------------------------------------------------------------------
# Coefficient elimination
# ---------------------------------------------------------------------------

def _build_elimination_matrix(dts: Sequence[float], N: int):
    """Build the (4N × 4N) matrix M and (4N × (6 + N-3)) selector S such that:

        M @ q = S @ knowns
        knowns = [P0, V0, A0, Pf, Vf, Af, d_3, d_4, ..., d_{N-1}]

    where q is the flat coefficient vector with q[4t+0..3] = (a_t, b_t, c_t, d_t).

    Returns (M_inv @ S) — a (4N × (6 + N-3)) numpy array C such that
    q = C @ knowns gives every coefficient as a linear combination of knowns.
    """
    M = np.zeros((4 * N, 4 * N), dtype=np.float64)
    S = np.zeros((4 * N, 6 + max(0, N - 3)), dtype=np.float64)
    row = 0

    # 1) Initial conditions at segment 0, local time τ=0:
    #   d_0 = P0
    M[row, 4 * 0 + 3] = 1.0; S[row, 0] = 1.0; row += 1
    #   c_0 = V0
    M[row, 4 * 0 + 2] = 1.0; S[row, 1] = 1.0; row += 1
    #   2 b_0 = A0
    M[row, 4 * 0 + 1] = 2.0; S[row, 2] = 1.0; row += 1

    # 2) Continuity at every junction t → t+1 (t = 0 .. N-2)
    for t in range(N - 1):
        T = float(dts[t])
        # C0:  a_t T^3 + b_t T^2 + c_t T + d_t  -  d_{t+1} = 0
        M[row, 4 * t + 0] = T ** 3
        M[row, 4 * t + 1] = T ** 2
        M[row, 4 * t + 2] = T
        M[row, 4 * t + 3] = 1.0
        M[row, 4 * (t + 1) + 3] = -1.0
        row += 1
        # C1:  3 a_t T^2 + 2 b_t T + c_t  -  c_{t+1} = 0
        M[row, 4 * t + 0] = 3.0 * T ** 2
        M[row, 4 * t + 1] = 2.0 * T
        M[row, 4 * t + 2] = 1.0
        M[row, 4 * (t + 1) + 2] = -1.0
        row += 1
        # C2:  3 a_t T + b_t  -  b_{t+1} = 0
        M[row, 4 * t + 0] = 3.0 * T
        M[row, 4 * t + 1] = 1.0
        M[row, 4 * (t + 1) + 1] = -1.0
        row += 1

    # 3) Final conditions at segment N-1, local time τ = T_{N-1}
    T = float(dts[N - 1])
    M[row, 4 * (N - 1) + 0] = T ** 3
    M[row, 4 * (N - 1) + 1] = T ** 2
    M[row, 4 * (N - 1) + 2] = T
    M[row, 4 * (N - 1) + 3] = 1.0
    S[row, 3] = 1.0; row += 1  # = Pf
    M[row, 4 * (N - 1) + 0] = 3.0 * T ** 2
    M[row, 4 * (N - 1) + 1] = 2.0 * T
    M[row, 4 * (N - 1) + 2] = 1.0
    S[row, 4] = 1.0; row += 1  # = Vf
    M[row, 4 * (N - 1) + 0] = 6.0 * T
    M[row, 4 * (N - 1) + 1] = 2.0
    S[row, 5] = 1.0; row += 1  # = Af

    # 4) Free variable assignments: d_t = d_t_var, t = 3 .. N-1
    for k in range(3, N):
        M[row, 4 * k + 3] = 1.0
        S[row, 6 + (k - 3)] = 1.0
        row += 1

    assert row == 4 * N, f"row count mismatch: {row} vs {4 * N}"
    C = np.linalg.solve(M, S)
    return C  # (4N, 6 + N-3)


# ---------------------------------------------------------------------------
# Initial dt heuristic (mirrors getInitialDt in C++)
# ---------------------------------------------------------------------------

def _smallest_positive_real_root(coeffs: Sequence[float]) -> float:
    """Smallest positive real root of polynomial with coefficients in increasing
    order [c0, c1, c2, ...] = c0 + c1*t + c2*t² + ...  Returns 0 if none.
    """
    rev = list(reversed(coeffs))  # numpy expects highest-degree first
    try:
        rs = np.roots(rev)
    except Exception:
        return 0.0
    best = float("inf")
    for r in rs:
        if abs(r.imag) > 1e-6:
            continue
        rv = float(r.real)
        if rv > 1e-9 and rv < best:
            best = rv
    return 0.0 if best == float("inf") else best


def compute_initial_dt(
    x0: np.ndarray, v0: np.ndarray, a0: np.ndarray,
    xf: np.ndarray, v_max: float, a_max: float, j_max: float, N: int,
) -> float:
    times = []
    for axis in range(3):
        dx = xf[axis] - x0[axis]
        # Velocity dash
        t_v = abs(dx) / max(v_max, 1e-6)
        # Acceleration ramp:  x0 - xf + v0 t + 0.5 * sign(dx) * a_max * t² = 0
        sign = math.copysign(1.0, dx) if abs(dx) > 1e-12 else 1.0
        t_a = _smallest_positive_real_root([x0[axis] - xf[axis], v0[axis], 0.5 * sign * a_max])
        # Jerk cubic:  x0 - xf + v0 t + (a0/2) t² + (sign * j_max / 6) t³ = 0
        t_j = _smallest_positive_real_root([
            x0[axis] - xf[axis], v0[axis], 0.5 * a0[axis], sign * j_max / 6.0,
        ])
        times.extend([t_v, t_a, t_j])
    dt = max(times) / max(1, N)
    if dt > 1e4 or not math.isfinite(dt):
        return 0.0
    return float(dt)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class SolveTiming:
    findDT_ms: float = 0.0
    setX_ms: float = 0.0
    polytopes_ms: float = 0.0
    dynamic_ms: float = 0.0
    objective_ms: float = 0.0
    mapsize_ms: float = 0.0
    callOptimizer_ms: float = 0.0
    postsolve_ms: float = 0.0


class SolverGurobi:
    def __init__(self):
        self.par = Parameters()
        self.N = 6
        self.dc = 0.01
        self.v_max = 10.0
        self.a_max = 20.0
        self.j_max = 100.0
        self.dynamic_constraint_type = "Linf"
        self.x_min = -200.0
        self.x_max = 200.0
        self.y_min = -200.0
        self.y_max = 200.0
        self.z_min = 0.5
        self.z_max = 6.0
        self.jerk_smooth_weight = 10.0
        self.initial_dt = 0.1
        self.t0 = 0.0
        self.x0 = np.zeros(3)
        self.v0 = np.zeros(3)
        self.a0 = np.zeros(3)
        self.xf = np.zeros(3)
        self.vf = np.zeros(3)
        self.af = np.zeros(3)
        self.dirf = 0.0
        self.dt = [0.0] * self.N
        self.total_traj_time = 0.0
        self.polytopes: List[Polytope] = []
        self.polytopes_time_layered: Optional[List[List[Polytope]]] = None

        self.basis = BasisConverter()

        # Set in solve(); reset between calls
        self._gp = None
        self._GRB = None
        self._env = None
        self._x_double = None  # (3, 4N) numeric coefficients after a solve
        self._goal_setpoints: List[RobotState] = []
        self.objective_value = float("nan")
        self.factor_that_worked = 0.0
        self.last_timing = SolveTiming()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def initialize(self, par: Parameters) -> None:
        self.par = par
        self.N = par.num_N
        self.dc = par.dc
        self.v_max = par.v_max
        self.a_max = par.a_max
        self.j_max = par.j_max
        self.dynamic_constraint_type = par.dynamic_constraint_type
        self.x_min = par.x_min; self.x_max = par.x_max
        self.y_min = par.y_min; self.y_max = par.y_max
        self.z_min = par.z_min; self.z_max = par.z_max
        self.jerk_smooth_weight = par.jerk_smooth_weight

    def set_X0(self, s: RobotState) -> None:
        self.x0 = s.pos.copy()
        self.v0 = s.vel.copy()
        self.a0 = s.accel.copy()

    def set_Xf(self, s: RobotState) -> None:
        self.xf = s.pos.copy()
        self.vf = s.vel.copy()
        self.af = s.accel.copy()

    def set_T0(self, t0: float) -> None:
        self.t0 = float(t0)

    def set_initial_dt(self, dt: float) -> None:
        self.initial_dt = float(dt)

    def set_dirf(self, yawf: float) -> None:
        self.dirf = float(yawf)

    def set_polytopes(self, polytopes: List[Polytope]) -> None:
        self.polytopes = polytopes
        self.polytopes_time_layered = None

    def set_polytopes_time_layered(self, layers: List[List[Polytope]]) -> None:
        self.polytopes_time_layered = layers
        # spatial polytopes per layer — assume all layers same length
        if layers:
            self.polytopes = layers[0]

    # ------------------------------------------------------------------
    def find_dt(self, factor: float) -> None:
        base = factor * max(self.initial_dt, 2.0 * self.dc)
        self.dt = [base] * self.N
        self.total_traj_time = sum(self.dt)

    def get_initial_dt(self) -> float:
        return compute_initial_dt(self.x0, self.v0, self.a0, self.xf,
                                  self.v_max, self.a_max, self.j_max, self.N)

    # ------------------------------------------------------------------
    # The optimization itself
    # ------------------------------------------------------------------
    def generate_new_trajectory(self, factor: float) -> Tuple[bool, float]:
        """Build and solve the QP for a single time-allocation factor.

        Returns (success, gurobi_compute_time_ms).
        """
        gp, GRB = _import_gurobi()
        N = self.N
        if N < 4:
            raise ValueError("SANDO requires N >= 4 segments")

        t_total = time.perf_counter()
        t0 = time.perf_counter()
        self.find_dt(factor)
        self.last_timing.findDT_ms = (time.perf_counter() - t0) * 1000.0

        # ----- Build elimination matrix C (4N × (6 + N-3))
        t0 = time.perf_counter()
        C = _build_elimination_matrix(self.dt, N)
        n_d_free = max(0, N - 3)
        # ----- Build Gurobi model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        m = gp.Model(env=env)
        m.setParam("TimeLimit", float(self.par.max_gurobi_comp_time_sec))
        m.setParam("MIPGap", 1e-2)

        # Free decision vars per axis: d_3, d_4, ..., d_{N-1}
        d_vars = []
        for axis in range(3):
            bounds_lo = self.x_min - 5.0 if axis == 0 else (self.y_min - 5.0 if axis == 1 else self.z_min - 2.0)
            bounds_hi = self.x_max + 5.0 if axis == 0 else (self.y_max + 5.0 if axis == 1 else self.z_max + 2.0)
            axis_vars = [m.addVar(lb=bounds_lo, ub=bounds_hi, vtype=GRB.CONTINUOUS,
                                  name=f"d{k}_{axis}") for k in range(3, N)]
            d_vars.append(axis_vars)
        m.update()

        # For each axis, build the 4N LinExpr coefficients via C @ knowns
        def coeffs_for_axis(axis: int):
            knowns_consts = np.array([
                self.x0[axis], self.v0[axis], self.a0[axis],
                self.xf[axis], self.vf[axis], self.af[axis],
            ], dtype=np.float64)
            base = C[:, :6] @ knowns_consts  # numeric constant component
            out = []
            for i in range(4 * N):
                expr = gp.LinExpr(float(base[i]))
                if n_d_free > 0:
                    for k in range(n_d_free):
                        coef = float(C[i, 6 + k])
                        if abs(coef) > 1e-15:
                            expr += coef * d_vars[axis][k]
                out.append(expr)
            return out

        x_coeffs = [coeffs_for_axis(axis) for axis in range(3)]  # list per axis of 4N LinExprs
        self.last_timing.setX_ms = (time.perf_counter() - t0) * 1000.0

        # ---- helpers to fetch (a,b,c,d) for segment t, axis ----
        def A(t, axis): return x_coeffs[axis][4 * t + 0]
        def B(t, axis): return x_coeffs[axis][4 * t + 1]
        def Cc(t, axis): return x_coeffs[axis][4 * t + 2]
        def D(t, axis): return x_coeffs[axis][4 * t + 3]

        T_seg = self.dt  # alias

        # Bezier position control points per segment, per axis (4 CPs)
        def cp_pos(t, axis, j):
            T = T_seg[t]
            # CP0 = d
            # CP1 = d + c*T/3
            # CP2 = d + 2*c*T/3 + b*T^2/3
            # CP3 = d + c*T + b*T^2 + a*T^3
            if j == 0:
                return D(t, axis) + gp.LinExpr(0.0)
            if j == 1:
                return D(t, axis) + (T / 3.0) * Cc(t, axis)
            if j == 2:
                return D(t, axis) + (2.0 * T / 3.0) * Cc(t, axis) + (T ** 2 / 3.0) * B(t, axis)
            if j == 3:
                return D(t, axis) + T * Cc(t, axis) + T ** 2 * B(t, axis) + T ** 3 * A(t, axis)
            raise IndexError(j)

        def cp_vel(t, axis, j):
            """Bezier velocity CPs: V0 = c, V1 = c + b*T, V2 = c + 2*b*T + 3*a*T²"""
            T = T_seg[t]
            if j == 0:
                return Cc(t, axis)
            if j == 1:
                return Cc(t, axis) + T * B(t, axis)
            if j == 2:
                return Cc(t, axis) + 2.0 * T * B(t, axis) + 3.0 * (T ** 2) * A(t, axis)
            raise IndexError(j)

        def cp_accel(t, axis, j):
            """A0 = 2b, A1 = 2b + 6aT"""
            T = T_seg[t]
            if j == 0:
                return 2.0 * B(t, axis)
            if j == 1:
                return 2.0 * B(t, axis) + 6.0 * T * A(t, axis)
            raise IndexError(j)

        def cp_jerk(t, axis):
            return 6.0 * A(t, axis)

        # ---- Polytope assignment binaries + at-least-one + indicator constraints ----
        t0 = time.perf_counter()
        b_vars: List[List[Optional[object]]] = []
        polys_per_seg = self._polys_for_segment  # bound below

        for t in range(N):
            polys_t = self._polys_for_segment(t)
            seg_bvars = []
            for p, poly in enumerate(polys_t):
                if poly is None or poly.A.shape[0] == 0:
                    seg_bvars.append(None)
                    continue
                bv = m.addVar(vtype=GRB.BINARY, name=f"b_{t}_{p}")
                seg_bvars.append(bv)
            b_vars.append(seg_bvars)
        m.update()

        for t in range(N):
            polys_t = self._polys_for_segment(t)
            active_bvars = [bv for bv in b_vars[t] if bv is not None]
            if active_bvars:
                m.addConstr(gp.quicksum(active_bvars) >= 1, name=f"atleast1_{t}")
            for p, (bv, poly) in enumerate(zip(b_vars[t], polys_t)):
                if bv is None or poly is None:
                    continue
                for j in range(4):  # 4 Bezier position CPs
                    cpx = cp_pos(t, 0, j)
                    cpy = cp_pos(t, 1, j)
                    cpz = cp_pos(t, 2, j)
                    for i in range(poly.A.shape[0]):
                        a_row = poly.A[i]
                        rhs = float(poly.b[i])
                        lhs = a_row[0] * cpx + a_row[1] * cpy + a_row[2] * cpz
                        m.addGenConstrIndicator(bv, True, lhs <= rhs,
                                                name=f"poly_{t}_{p}_{j}_{i}")
        self.last_timing.polytopes_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Dynamic constraints (Linf default; L1; L2) ----
        t0 = time.perf_counter()
        norm_type = self.dynamic_constraint_type
        signs1d = (-1.0, 1.0)
        for t in range(N):
            for j in range(3):  # vel CPs
                vx = cp_vel(t, 0, j); vy = cp_vel(t, 1, j); vz = cp_vel(t, 2, j)
                self._dyn_constrain(m, gp, vx, vy, vz, self.v_max, norm_type, f"v_{t}_{j}")
            for j in range(2):  # accel CPs
                ax = cp_accel(t, 0, j); ay = cp_accel(t, 1, j); az = cp_accel(t, 2, j)
                self._dyn_constrain(m, gp, ax, ay, az, self.a_max, norm_type, f"a_{t}_{j}")
            jx = cp_jerk(t, 0); jy = cp_jerk(t, 1); jz = cp_jerk(t, 2)
            self._dyn_constrain(m, gp, jx, jy, jz, self.j_max, norm_type, f"j_{t}")
        self.last_timing.dynamic_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Map-bound box on Bezier position CPs ----
        t0 = time.perf_counter()
        for t in range(N):
            for j in range(4):
                cpx = cp_pos(t, 0, j)
                cpy = cp_pos(t, 1, j)
                cpz = cp_pos(t, 2, j)
                m.addConstr(cpx <= self.x_max, name=f"mapx_hi_{t}_{j}")
                m.addConstr(cpx >= self.x_min, name=f"mapx_lo_{t}_{j}")
                m.addConstr(cpy <= self.y_max, name=f"mapy_hi_{t}_{j}")
                m.addConstr(cpy >= self.y_min, name=f"mapy_lo_{t}_{j}")
                m.addConstr(cpz <= self.z_max, name=f"mapz_hi_{t}_{j}")
                m.addConstr(cpz >= self.z_min, name=f"mapz_lo_{t}_{j}")
        self.last_timing.mapsize_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Objective: jerk smoothness ----
        t0 = time.perf_counter()
        obj = gp.QuadExpr(0.0)
        for t in range(N):
            for axis in range(3):
                ja = 6.0 * A(t, axis)
                obj += ja * ja
        m.setObjective(self.jerk_smooth_weight * obj, GRB.MINIMIZE)
        self.last_timing.objective_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Solve ----
        t0 = time.perf_counter()
        m.optimize()
        self.last_timing.callOptimizer_ms = (time.perf_counter() - t0) * 1000.0

        ok = m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0
        if not ok:
            self.objective_value = float("nan")
            try:
                m.dispose()
            except Exception:
                pass
            env.dispose()
            return False, self.last_timing.callOptimizer_ms

        # ---- Extract coefficients ----
        t0 = time.perf_counter()
        # Snapshot d_k_var values, then numerically recover all coefficients
        knowns = np.zeros((3, 6 + n_d_free), dtype=np.float64)
        for axis in range(3):
            knowns[axis, 0] = self.x0[axis]; knowns[axis, 1] = self.v0[axis]; knowns[axis, 2] = self.a0[axis]
            knowns[axis, 3] = self.xf[axis]; knowns[axis, 4] = self.vf[axis]; knowns[axis, 5] = self.af[axis]
            for k in range(n_d_free):
                knowns[axis, 6 + k] = float(d_vars[axis][k].X)
        x_double = np.zeros((3, 4 * N), dtype=np.float64)
        for axis in range(3):
            x_double[axis] = C @ knowns[axis]
        self._x_double = x_double
        self.objective_value = float(m.ObjVal)
        self.factor_that_worked = float(factor)

        # Goal setpoints sampled at dc
        self._fill_goal_setpoints()
        self.last_timing.postsolve_ms = (time.perf_counter() - t0) * 1000.0

        try:
            m.dispose()
        except Exception:
            pass
        env.dispose()
        return True, self.last_timing.callOptimizer_ms

    def _polys_for_segment(self, t: int) -> List[Polytope]:
        if self.polytopes_time_layered is not None:
            tl = self.polytopes_time_layered
            return tl[t] if t < len(tl) else tl[-1]
        return self.polytopes

    def _dyn_constrain(self, m, gp, ex, ey, ez, limit, norm_type, prefix):
        if norm_type == "Linf":
            m.addConstr(ex <= limit, name=f"{prefix}_x_hi")
            m.addConstr(ex >= -limit, name=f"{prefix}_x_lo")
            m.addConstr(ey <= limit, name=f"{prefix}_y_hi")
            m.addConstr(ey >= -limit, name=f"{prefix}_y_lo")
            m.addConstr(ez <= limit, name=f"{prefix}_z_hi")
            m.addConstr(ez >= -limit, name=f"{prefix}_z_lo")
        elif norm_type == "L1":
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        m.addConstr(sx * ex + sy * ey + sz * ez <= limit,
                                    name=f"{prefix}_{int(sx)}{int(sy)}{int(sz)}")
        elif norm_type == "L2":
            m.addQConstr(ex * ex + ey * ey + ez * ez <= limit * limit,
                          name=f"{prefix}_l2")
        else:
            raise ValueError(f"Unknown dynamic_constraint_type: {norm_type}")

    # ------------------------------------------------------------------
    def _fill_goal_setpoints(self) -> None:
        """Sample the solved trajectory at every dc seconds to fill
        self._goal_setpoints. Mirrors fillGoalSetPoints in C++.
        """
        if self._x_double is None:
            self._goal_setpoints = []
            return
        n_samples = max(2, int(math.ceil(self.total_traj_time / self.dc)))
        out: List[RobotState] = []
        for i in range(n_samples):
            t_in_traj = (i + 1) * self.dc
            t_in_traj = min(t_in_traj, self.total_traj_time - 1e-6)
            seg_idx, tau = self._interval_idx_and_dt(t_in_traj)
            s = RobotState()
            s.t = self.t0 + t_in_traj
            for axis in range(3):
                a = self._x_double[axis, 4 * seg_idx + 0]
                b = self._x_double[axis, 4 * seg_idx + 1]
                c = self._x_double[axis, 4 * seg_idx + 2]
                d = self._x_double[axis, 4 * seg_idx + 3]
                s.pos[axis] = a * tau ** 3 + b * tau ** 2 + c * tau + d
                s.vel[axis] = 3.0 * a * tau ** 2 + 2.0 * b * tau + c
                s.accel[axis] = 6.0 * a * tau + 2.0 * b
                s.jerk[axis] = 6.0 * a
            out.append(s)
        self._goal_setpoints = out

    def _interval_idx_and_dt(self, t_in_traj: float) -> Tuple[int, float]:
        acc = 0.0
        for i, T in enumerate(self.dt):
            if t_in_traj < acc + T:
                return i, t_in_traj - acc
            acc += T
        return len(self.dt) - 1, self.dt[-1]

    # ------------------------------------------------------------------
    # Sequential factor sweep mimicking generateNewTrajectorySequentialFactors
    # ------------------------------------------------------------------
    def generate_new_trajectory_sequential_factors(self) -> Tuple[bool, float, float]:
        f = self.par.factor_initial
        f_end = self.par.factor_final
        step = max(1e-6, self.par.factor_constant_step_size)
        best_compute = 0.0
        while f <= f_end + 1e-9:
            ok, compute = self.generate_new_trajectory(f)
            best_compute += compute
            if ok:
                return True, best_compute, f
            f += step
        return False, best_compute, 0.0

    # ------------------------------------------------------------------
    def get_piecewise_pol(self) -> PieceWisePol:
        pwp = PieceWisePol()
        if self._x_double is None:
            return pwp
        N = self.N
        # times: [t0, t0 + dt[0], t0 + dt[0] + dt[1], ...]
        pwp.times = [self.t0]
        acc = self.t0
        for T in self.dt:
            acc += T
            pwp.times.append(acc)
        for t in range(N):
            pwp.coeff_x.append(np.array(self._x_double[0, 4 * t:4 * t + 4], dtype=np.float64))
            pwp.coeff_y.append(np.array(self._x_double[1, 4 * t:4 * t + 4], dtype=np.float64))
            pwp.coeff_z.append(np.array(self._x_double[2, 4 * t:4 * t + 4], dtype=np.float64))
        return pwp

    def get_goal_setpoints(self) -> List[RobotState]:
        return list(self._goal_setpoints)

    def get_control_points_bezier(self) -> List[np.ndarray]:
        """Return (3,4) Bezier position control points per segment."""
        if self._x_double is None:
            return []
        out = []
        for t in range(self.N):
            T = self.dt[t]
            CPs = np.zeros((3, 4))
            for axis in range(3):
                a = self._x_double[axis, 4 * t + 0]
                b = self._x_double[axis, 4 * t + 1]
                c = self._x_double[axis, 4 * t + 2]
                d = self._x_double[axis, 4 * t + 3]
                CPs[axis, 0] = d
                CPs[axis, 1] = d + c * T / 3.0
                CPs[axis, 2] = d + 2.0 * c * T / 3.0 + b * T ** 2 / 3.0
                CPs[axis, 3] = d + c * T + b * T ** 2 + a * T ** 3
            out.append(CPs)
        return out
