"""v2 local trajectory optimizer — Gurobi QP.

Mirrors the C++ ``SolverGurobi`` (``include/sando/gurobi_solver.hpp`` +
``src/sando/gurobi_solver.cpp``) at the formulation level. We do not
attempt a line-by-line port of the 5kLOC C++ implementation; the result
is a clean Python rewrite that produces the same kind of trajectory:

* cubic polynomial per polynomial-segment (matches ``PieceWisePol``),
* anchored at the start to the agent's current ``RobotState`` — pos,
  vel, **and** acceleration are all matched (the v1 cubic-Hermite
  fitter dropped acceleration),
* anchored at the end to the terminal goal with zero vel and zero accel,
* C0 / C1 / C2 continuity at every internal junction,
* per-axis ``v_max`` / ``a_max`` / ``j_max`` bounds on the same Bezier
  derivative control points used by the C++ solver,
* corridor membership (``A x <= b``) via the **Bezier convex-hull
  property** — 4 control points per segment per polytope row, no
  clipping possible between samples. Matches the C++
  ``SolverGurobi::setPolyConsts`` enforcement (the C++ also computes
  MINVO CPs but uses them only for *post-solve verification*, not for
  the Gurobi inequality constraints),
* C++ jerk-smoothness objective:
  ``jerk_smooth_weight * sum_seg ||6a_seg||²``.

The "field-for-field" promise in ``solver/README.md`` is aspirational —
this implementation produces the same kind of output as the C++ but with
a different (simpler) variable layout. The C++ solver also features
variable elimination, MINVO basis control points for tighter corridor
membership, and a dynamic time-allocation factor; those are follow-ups.

Segment Count And Time
----------------------
The C++ uses exactly ``num_N`` local polynomial segments. The global
path is sampled only to seed the model; safe-corridor membership is
handled by MIQP indicators. Segment duration is uniform:
``factor * max(getInitialDt(), 2 * dc)``.

Failure modes
-------------
If Gurobi reports infeasible / unbounded / no solution, the call
returns ``SolverInfo(success=False)`` and the caller holds the previous
trajectory — same contract as v1. We never raise into the replan loop.
"""

from __future__ import annotations

import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from sando_py.core import Parameters, PieceWisePol, Polytope, RobotState

from .elimination import LinearCoeff, SegmentExpressions
from .time_alloc import TimeAlloc
from .types import SolveTimingBreakdown, SolverInfo


# Default num_N when the YAML / Parameters object hasn't set one. Picked
# to match the C++ ``int N_ = 6;`` default.
_DEFAULT_NUM_N = 6

# Floor on per-segment duration so very short edges don't generate
# numerically unstable cubic coefficients.
_T_MIN = 0.05

# Cruise = ``_CRUISE_ALPHA * v_max`` for the per-segment time alloc
# heuristic. Same value the v1 solver uses.
_CRUISE_ALPHA = 0.5

# Cap on parallel-search worker threads. The C++ fires one Gurobi worker
# per factor; we mirror that but cap at this number to avoid creating
# enough threads to wedge the Gurobi env.
_MAX_PARALLEL_WORKERS = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GurobiSolver:
    """Stateless solver — stash params, call ``solve`` per replan tick.

    Construction does *not* touch Gurobi; the model is built per-call.
    Pre-creating an environment / model and reusing it (as the C++ does
    via ``setDynamicConstraintsFaster_``) is a perf follow-up.
    """

    # Shared Gurobi environment, reused across calls. Each ``Env()``
    # creation triggers a WLS license check (~10 s); reusing it makes
    # successive solves cheap.
    _shared_env = None

    def __init__(self, params: Parameters) -> None:
        self.params = params
        # Lazy import so the module can be loaded even without gurobipy
        # available — tests that don't exercise the solver should still pass.
        self._gp = None
        self._GRB = None
        # Dynamic time-allocation adapter — yields a single 1.0 factor
        # when ``use_dynamic_factor`` is off, otherwise iterates a
        # window of candidates centred on the most recent success.
        self.time_alloc = TimeAlloc(params)
        # Persistent model + variables (mirrors the C++
        # ``setDynamicConstraintsFaster_`` pattern). Keyed by ``n_seg``
        # because variable count depends on it; if a replan tick lands
        # with a different ``n_seg`` we rebuild. Constraints are cleared
        # and rebuilt every call — they depend on the time allocation,
        # which the adapter changes each retry.
        self._cached_model = None
        self._cached_coeffs = None
        self._cached_n_seg = -1

    @classmethod
    def _get_env(cls):
        if cls._shared_env is None:
            import gurobipy as gp
            cls._shared_env = gp.Env()
        return cls._shared_env

    def _acquire_model(self, gp, GRB, n_seg: int):
        """Return ``(model, coeffs)`` for the requested segment count,
        creating a fresh model + variables only when ``n_seg`` differs
        from the cached one. Always returns a model with **no
        constraints attached** — callers add the per-call constraints.
        """
        if self._cached_n_seg != n_seg or self._cached_model is None:
            env = self._get_env()
            model = gp.Model(f"sando_v2_n{n_seg}", env=env)
            model.setParam("OutputFlag", 0)
            if self.params.max_gurobi_comp_time_sec > 0:
                model.setParam(
                    "TimeLimit", float(self.params.max_gurobi_comp_time_sec)
                )
            # Variables: coeffs[ax][seg][i] for i = 0..3 -> (a, b, c, d)
            coeffs = [
                [
                    [
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"a_ax{ax}_seg{seg}"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"b_ax{ax}_seg{seg}"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"c_ax{ax}_seg{seg}"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"d_ax{ax}_seg{seg}"),
                    ]
                    for seg in range(n_seg)
                ]
                for ax in range(3)
            ]
            self._cached_model = model
            self._cached_coeffs = coeffs
            self._cached_n_seg = n_seg
        else:
            model = self._cached_model
            coeffs = self._cached_coeffs
            # Clear all linear constraints from the previous call so we
            # can add fresh ones tied to the new time allocation /
            # corridor set. Variables persist; the previous solution
            # remains in their ``.X`` so the warm-start path still helps.
            # Batched remove is much faster than per-constraint remove on
            # models with many constraints.
            constrs = model.getConstrs()
            if constrs:
                model.remove(constrs)
            qconstrs = model.getQConstrs()
            if qconstrs:
                model.remove(qconstrs)
        return model, coeffs

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(
        self,
        A: RobotState,
        path: List[np.ndarray],
        polys,  # Sequence[Polytope] OR Sequence[Sequence[Polytope]] (time-layered)
    ) -> Tuple[PieceWisePol, SolverInfo]:
        t0 = time.perf_counter()

        if path is None or len(path) < 2:
            return PieceWisePol(), SolverInfo(success=False,
                                              wall_time_s=time.perf_counter() - t0)

        # Import Gurobi lazily. On ImportError, log once at WARNING level
        # so a missing ``gurobipy`` install doesn't silently fail every
        # replan tick — the previous behaviour returned ``success=False``
        # without any indication that the solver was never even invoked.
        try:
            if self._gp is None:
                import gurobipy as gp
                from gurobipy import GRB
                self._gp = gp
                self._GRB = GRB
        except ImportError as _e:
            if not getattr(self, "_warned_no_gurobi", False):
                import warnings
                warnings.warn(
                    f"GurobiSolver: gurobipy not importable ({_e}). "
                    "Every solve() will return failure. Install gurobipy "
                    "into the runtime Python interpreter "
                    "(e.g. `pip install gurobipy`).",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned_no_gurobi = True
            return PieceWisePol(), SolverInfo(success=False,
                                              wall_time_s=time.perf_counter() - t0)

        gp = self._gp
        GRB = self._GRB

        if polys is None:
            polys = []
        pts = [np.asarray(p, dtype=float).reshape(3) for p in path]

        # C++ uses exactly ``num_N`` local polynomial segments. Internal
        # points are only a warm-start scaffold; the QP itself is
        # constrained by boundary states, dynamics, map bounds, and SFC.
        sub_pts, parent_of_seg = _subdivide(pts, self.params)
        n_seg = len(sub_pts) - 1

        Ts_base = _allocate_times(A, sub_pts[-1], n_seg, self.params)
        if any(T <= 0 for T in Ts_base):
            return PieceWisePol(), SolverInfo(success=False,
                                              wall_time_s=time.perf_counter() - t0)

        factors = list(self.time_alloc.factors_to_try())

        # Pick the kernel: variable-elimination scheme (default, ~10x
        # fewer free vars in Gurobi) vs the naive scheme (one Gurobi var
        # per polynomial coefficient). C++ default is elimination on
        # (sando_type.hpp:173).
        use_elim = (
            bool(getattr(self.params, "using_variable_elimination", True))
            and n_seg >= 4
        )
        solve_fn = self._solve_one_elim if use_elim else self._solve_one_fresh

        # Single-factor path: keep the cached-model fast path. This is
        # the hot path when ``use_dynamic_factor`` is off, and also when
        # the adapter has converged to a 1-element window.
        if len(factors) <= 1:
            factor = factors[0] if factors else 1.0
            Ts = [T * factor for T in Ts_base]
            pwp, info = solve_fn(
                gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
            )
            info.successful_factor = float(factor)
            if info.success:
                self.time_alloc.on_success(factor)
            else:
                self.time_alloc.on_failure()
            return pwp, info

        # Multi-factor path: parallel search, mirrors the C++
        # ``std::async`` pattern in ``sando.cpp::tryToCompleteOpt``.
        # Each worker builds its own Gurobi model (no cache sharing —
        # cache is single-thread only), all run concurrently, we pick
        # the lowest-factor success.
        return self._solve_parallel(
            gp, GRB, A, sub_pts, parent_of_seg, polys,
            Ts_base, factors, n_seg, t0,
            use_elim=use_elim,
        )

    def _solve_parallel(
        self, gp, GRB, A, sub_pts, parent_of_seg, polys,
        Ts_base, factors, n_seg, t0, *, use_elim: bool = True,
    ) -> Tuple[PieceWisePol, SolverInfo]:
        """Run one Gurobi solve per factor in worker threads, return the
        result with the lowest successful factor. Mirrors the C++
        parallel-factor pattern."""
        n_workers = min(_MAX_PARALLEL_WORKERS, len(factors))
        results: List[Tuple[float, PieceWisePol, SolverInfo]] = []
        results_lock = threading.Lock()

        worker_fn = (
            self._solve_one_elim_fresh if use_elim else self._solve_one_fresh
        )

        def _worker(factor: float):
            Ts = [T * factor for T in Ts_base]
            pwp, info = worker_fn(
                gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
            )
            with results_lock:
                results.append((factor, pwp, info))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_worker, f) for f in factors]
            # Block until all complete; failures or successes both come
            # back via the results list.
            for fut in as_completed(futures):
                # Re-raise any unexpected worker exceptions
                fut.result()

        # Among successful results, return the one with the lowest
        # factor. The C++ uses the same "lowest successful" rule so the
        # trajectory ends up as tight as we can make it.
        successful = sorted(
            ((f, pwp, info) for f, pwp, info in results if info.success),
            key=lambda triple: triple[0],
        )
        if successful:
            best_factor, best_pwp, best_info = successful[0]
            best_info.successful_factor = float(best_factor)
            self.time_alloc.on_success(best_factor)
            return best_pwp, best_info

        self.time_alloc.on_failure()
        # Report the highest-cost failure we got so the caller has a
        # cost to log; ``wall_time_s`` reflects the parallel wall-clock.
        return PieceWisePol(), SolverInfo(
            success=False,
            wall_time_s=time.perf_counter() - t0,
            cost=max((info.cost for _, _, info in results), default=0.0),
        )

    def _solve_one(
        self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
    ) -> Tuple[PieceWisePol, SolverInfo]:
        """Single Gurobi solve, single-threaded fast path. Reuses the
        cached model + variables when ``n_seg`` hasn't changed since the
        last call."""
        try:
            model, coeffs = self._acquire_model(gp, GRB, n_seg)

            # Provide initial guess from a v1 cubic Hermite (helps warm-start)
            _set_initial_guess(coeffs, sub_pts, Ts, A, self.params)

            _add_boundary_constraints(model, coeffs, Ts, A, sub_pts[-1])
            _add_continuity_constraints(model, coeffs, Ts)
            _add_dynamic_constraints(model, coeffs, Ts, self.params)
            _add_map_size_constraints(model, coeffs, Ts, self.params)
            _add_corridor_constraints(
                gp, GRB, model, coeffs, Ts, parent_of_seg, polys, n_seg
            )

            obj = gp.QuadExpr()
            jerk_w = max(0.0, float(self.params.jerk_smooth_weight or 1.0))
            for ax in range(3):
                for seg in range(n_seg):
                    a = coeffs[ax][seg][0]
                    obj += jerk_w * 36.0 * a * a
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            if not _has_usable_solution(model, GRB):
                return PieceWisePol(), SolverInfo(
                    success=False, wall_time_s=time.perf_counter() - t0,
                )

            pwp = _extract_pwp(coeffs, Ts, A.t)
            return pwp, SolverInfo(
                success=True,
                cost=float(model.ObjVal),
                wall_time_s=time.perf_counter() - t0,
            )
        except gp.GurobiError:
            return PieceWisePol(), SolverInfo(
                success=False, wall_time_s=time.perf_counter() - t0,
            )

    def _solve_one_fresh(
        self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
    ) -> Tuple[PieceWisePol, SolverInfo]:
        """Single Gurobi solve with a **freshly built** model (no
        caching). Used by :meth:`_solve_parallel` so each thread has its
        own model — the cached model is not thread-safe.

        The shared ``Env`` is fine to use from multiple threads (Gurobi
        serialises operations internally), but the model + variables
        must be per-thread.
        """
        try:
            env = self._get_env()
            model = gp.Model(f"sando_v2_par_n{n_seg}", env=env)
            model.setParam("OutputFlag", 0)
            if self.params.max_gurobi_comp_time_sec > 0:
                model.setParam(
                    "TimeLimit", float(self.params.max_gurobi_comp_time_sec),
                )
            coeffs = [
                [
                    [
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"a_ax{ax}_seg{seg}"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"b_ax{ax}_seg{seg}"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"c_ax{ax}_seg{seg}"),
                        model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                     name=f"d_ax{ax}_seg{seg}"),
                    ]
                    for seg in range(n_seg)
                ]
                for ax in range(3)
            ]

            _set_initial_guess(coeffs, sub_pts, Ts, A, self.params)
            _add_boundary_constraints(model, coeffs, Ts, A, sub_pts[-1])
            _add_continuity_constraints(model, coeffs, Ts)
            _add_dynamic_constraints(model, coeffs, Ts, self.params)
            _add_map_size_constraints(model, coeffs, Ts, self.params)
            _add_corridor_constraints(
                gp, GRB, model, coeffs, Ts, parent_of_seg, polys, n_seg
            )

            obj = gp.QuadExpr()
            jerk_w = max(0.0, float(self.params.jerk_smooth_weight or 1.0))
            for ax in range(3):
                for seg in range(n_seg):
                    a = coeffs[ax][seg][0]
                    obj += jerk_w * 36.0 * a * a
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            if not _has_usable_solution(model, GRB):
                return PieceWisePol(), SolverInfo(
                    success=False, wall_time_s=time.perf_counter() - t0,
                )

            pwp = _extract_pwp(coeffs, Ts, A.t)
            return pwp, SolverInfo(
                success=True,
                cost=float(model.ObjVal),
                wall_time_s=time.perf_counter() - t0,
            )
        except gp.GurobiError:
            return PieceWisePol(), SolverInfo(
                success=False, wall_time_s=time.perf_counter() - t0,
            )


def solve_qp(
    A: RobotState,
    path: List[np.ndarray],
    polys: Sequence[Polytope],
    params: Parameters,
) -> Tuple[PieceWisePol, SolverInfo]:
    """Functional shim that matches the README contract."""
    return GurobiSolver(params).solve(A, path, polys)


# ---------------------------------------------------------------------------
# Variable-elimination kernel (mirrors C++ ``computeDependentCoefficientsN*``)
# ---------------------------------------------------------------------------

def _solve_elim_core(
    self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0, *,
    fresh: bool,
):
    """Core of both ``_solve_one_elim`` and ``_solve_one_elim_fresh``.
    Builds a Gurobi QP with only ``(n_seg - 3)`` free position variables
    per axis (vs ``4 * n_seg`` in the naive scheme) by encoding boundary
    + continuity constraints symbolically. Mirrors the C++
    ``createVars`` / ``computeDependentCoefficientsN*`` pipeline."""
    timing = SolveTimingBreakdown()
    try:
        # --- findDT: time allocation already done outside (Ts is fixed),
        # but match the C++ stage name. The Python time_alloc call is
        # cheap (microseconds) so we report it as 0 unless the caller
        # routes through here.
        t_findDT = time.perf_counter()
        # Boundary state for each axis: [P0, V0, A0, Pf, Vf, Af]
        A_pos = np.asarray(A.pos, dtype=float).reshape(3)
        A_vel = np.asarray(A.vel, dtype=float).reshape(3)
        A_acc = np.asarray(A.accel, dtype=float).reshape(3)
        Pf = np.asarray(sub_pts[-1], dtype=float).reshape(3)
        boundary_per_axis = [
            np.array([A_pos[ax], A_vel[ax], A_acc[ax],
                      Pf[ax], 0.0, 0.0], dtype=float)
            for ax in range(3)
        ]
        timing.findDT_ms = (time.perf_counter() - t_findDT) * 1000.0

        # --- setX: symbolic coefficient construction + variable creation
        # + continuity encoding (continuity is baked in by elimination,
        # so the time mostly goes to SegmentExpressions.build).
        t_setX = time.perf_counter()
        dt = np.asarray(Ts, dtype=float)
        seg_exprs_per_axis = [
            SegmentExpressions.build(n_seg, dt, boundary_per_axis[ax])
            for ax in range(3)
        ]
        num_free = seg_exprs_per_axis[0].num_free
        if num_free <= 0:
            # Degenerate: fully constrained, no DOF. Fall back to the
            # naive path so we don't have a 0-variable Gurobi model.
            return self._solve_one_fresh(
                gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
            )

        env = self._get_env()
        model = gp.Model(
            f"sando_v2_elim_n{n_seg}" + ("_par" if fresh else ""), env=env,
        )
        model.setParam("OutputFlag", 0)
        if self.params.max_gurobi_comp_time_sec > 0:
            model.setParam(
                "TimeLimit", float(self.params.max_gurobi_comp_time_sec),
            )

        # Free variables: ``num_free`` per axis (e.g. 2 per axis for N=5
        # → 6 total, vs 60 in the naive scheme).
        free_vars = [
            [
                model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                             name=f"df_ax{ax}_j{j}")
                for j in range(num_free)
            ]
            for ax in range(3)
        ]
        timing.setX_ms = (time.perf_counter() - t_setX) * 1000.0

        # Helper: LinearCoeff → Gurobi LinExpr in the per-axis free vars.
        def _le(lc: LinearCoeff, ax: int):
            expr = gp.LinExpr(lc.const)
            for j, ci in enumerate(lc.coef):
                if ci != 0.0:
                    expr.add(free_vars[ax][j], float(ci))
            return expr

        # ---- Corridor constraints ----
        # Per-segment Bezier CPs as LinExpr for all 3 axes, computed once.
        # Each cubic lies in conv({P0, P1, P2, P3}) — constraining every
        # CP inside the polytope is a *sufficient* condition for the
        # whole segment.
        t_polytopes = time.perf_counter()
        cp_le_per_seg = []  # cp_le_per_seg[seg][ax][i] = LinExpr
        for seg in range(n_seg):
            T = Ts[seg]
            T2, T3 = T * T, T * T * T
            cps_ax = []
            for ax in range(3):
                a_lc = seg_exprs_per_axis[ax].a(seg)
                b_lc = seg_exprs_per_axis[ax].b(seg)
                c_lc = seg_exprs_per_axis[ax].c(seg)
                d_lc = seg_exprs_per_axis[ax].d(seg)
                cps_lc = [
                    d_lc,
                    d_lc + c_lc * (T / 3.0),
                    d_lc + c_lc * (2.0 * T / 3.0) + b_lc * (T2 / 3.0),
                    d_lc + c_lc * T + b_lc * T2 + a_lc * T3,
                ]
                cps_ax.append([_le(cp, ax) for cp in cps_lc])
            cp_le_per_seg.append(cps_ax)

        # Detect time-layered polytopes: ``polys`` is ``List[List[Polytope]]``
        # when caller passed `sfc_time_layered` output, else `List[Polytope]`.
        is_time_layered = (
            isinstance(polys, (list, tuple))
            and len(polys) > 0
            and isinstance(polys[0], (list, tuple))
        )
        if is_time_layered or len(polys) > 0:
            if is_time_layered:
                N_time = len(polys)
                P = len(polys[0])

                def _poly_at(t_sub: int, p: int):
                    t_layer = t_sub if n_seg <= N_time else int(t_sub * N_time / n_seg)
                    t_layer = min(max(0, t_layer), N_time - 1)
                    return polys[t_layer][p]
            else:
                P = len(polys)

                def _poly_at(t_sub: int, p: int):
                    return polys[p]

            b_vars = [
                [
                    model.addVar(vtype=GRB.BINARY, name=f"b_t{t}_p{p}")
                    for p in range(P)
                ]
                for t in range(n_seg)
            ]
            for t in range(n_seg):
                # at_least_1_pol: sum_p b[t][p] >= 1
                model.addConstr(
                    gp.quicksum(b_vars[t][p] for p in range(P)) >= 1,
                    name=f"at_least_1_pol_t{t}",
                )
                for p in range(P):
                    poly = _poly_at(t, p)
                    if poly.A.shape[0] == 0:
                        # Empty polytope → disable this binary so the
                        # solver can't assign to a non-existent corridor.
                        model.addConstr(b_vars[t][p] == 0,
                                        name=f"empty_t{t}_p{p}")
                        continue
                    for i in range(4):
                        cp_le = cp_le_per_seg[t]
                        for row in range(poly.A.shape[0]):
                            a0 = float(poly.A[row, 0])
                            a1 = float(poly.A[row, 1])
                            a2 = float(poly.A[row, 2])
                            lhs = (a0 * cp_le[0][i]
                                   + a1 * cp_le[1][i]
                                   + a2 * cp_le[2][i])
                            # Gurobi's indicator constraint: b==1 → lhs <= rhs
                            model.addGenConstrIndicator(
                                b_vars[t][p], 1, lhs, GRB.LESS_EQUAL,
                                float(poly.b[row, 0]),
                                f"miqp_t{t}_p{p}_face{row}_cp{i}",
                            )
        timing.polytopes_ms = (time.perf_counter() - t_polytopes) * 1000.0

        # ---- Bezier dynamic constraints (v, a, j), matching C++ getVelCP/getAccelCP/getJerkCP ----
        t_dynamic = time.perf_counter()
        # Three norm types — match the C++ ``dynamic_constraint_type``:
        #   "Linf" → per-axis box  (default, smallest constraint count)
        #   "L1"   → 8-face octahedron over (vx, vy, vz)  (tighter)
        #   "L2"   → quadratic ball  (tightest, introduces QC)
        v_max = max(0.0, float(self.params.v_max))
        a_max = max(0.0, float(self.params.a_max))
        j_max = max(0.0, float(self.params.j_max))
        norm_type = (self.params.dynamic_constraint_type or "Linf").strip()

        # Precompute the 8 octahedron-face sign vectors for L1.
        _L1_SIGNS = np.array([
            (sx, sy, sz)
            for sx in (+1, -1) for sy in (+1, -1) for sz in (+1, -1)
        ], dtype=float)

        def _add_dyn_constraints(per_axis_cps, bound: float, label: str = "dyn", seg_idx: int = -1):
            """``per_axis_cps`` is a list of length 3 (one per axis),
            each holding a list of Bezier derivative CPs (LinearCoeff). Add the
            requested-norm constraints across all CPs."""
            if bound <= 0:
                return
            n_cps = len(per_axis_cps[0])
            if norm_type == "Linf":
                for ax in range(3):
                    for k, cp in enumerate(per_axis_cps[ax]):
                        e = _le(cp, ax)
                        model.addConstr(e <= bound, name=f"{label}_s{seg_idx}_a{ax}_k{k}_hi")
                        model.addConstr(e >= -bound, name=f"{label}_s{seg_idx}_a{ax}_k{k}_lo")
            elif norm_type == "L1":
                # |vx|+|vy|+|vz| <= v_max  ⇔  sx*vx + sy*vy + sz*vz <= v_max
                # for every sign tuple (8 faces).
                for k in range(n_cps):
                    expr_axis = [
                        _le(per_axis_cps[0][k], 0),
                        _le(per_axis_cps[1][k], 1),
                        _le(per_axis_cps[2][k], 2),
                    ]
                    for sgn in _L1_SIGNS:
                        rhs = bound
                        lhs = (
                            float(sgn[0]) * expr_axis[0]
                            + float(sgn[1]) * expr_axis[1]
                            + float(sgn[2]) * expr_axis[2]
                        )
                        model.addConstr(lhs <= rhs)
            elif norm_type == "L2":
                # vx^2 + vy^2 + vz^2 <= v_max^2 (second-order cone).
                # Squaring LinExprs directly produces bilinear constraints
                # that Gurobi treats as nonconvex. Indirect via auxiliary
                # equalities so the QCP is recognised as convex SOC.
                for k in range(n_cps):
                    aux = [
                        model.addVar(lb=-bound, ub=bound,
                                     name=f"l2aux_{norm_type}_k{k}_ax{ax}")
                        for ax in range(3)
                    ]
                    for ax in range(3):
                        cp_le = _le(per_axis_cps[ax][k], ax)
                        model.addConstr(aux[ax] == cp_le)
                    model.addQConstr(
                        aux[0] * aux[0] + aux[1] * aux[1] + aux[2] * aux[2]
                        <= bound * bound
                    )
            else:
                # Unknown — fall back to Linf to avoid silent failure.
                for ax in range(3):
                    for cp in per_axis_cps[ax]:
                        e = _le(cp, ax)
                        model.addConstr(e <= bound)
                        model.addConstr(e >= -bound)

        for seg in range(n_seg):
            T = Ts[seg]
            # Gather Bezier derivative CPs per axis (each axis → list of CPs).
            vel_cps = [None] * 3
            accel_cps = [None] * 3
            jerk_cps = [None] * 3
            for ax in range(3):
                a_lc = seg_exprs_per_axis[ax].a(seg)
                b_lc = seg_exprs_per_axis[ax].b(seg)
                c_lc = seg_exprs_per_axis[ax].c(seg)
                vel_cps[ax] = [
                    c_lc,
                    c_lc + b_lc * T,
                    c_lc + b_lc * (2.0 * T) + a_lc * (3.0 * T * T),
                ]
                accel_cps[ax] = [
                    b_lc * 2.0,
                    b_lc * 2.0 + a_lc * (6.0 * T),
                ]
                jerk_cps[ax] = [a_lc * 6.0]

            _add_dyn_constraints(vel_cps, v_max, label="vmax", seg_idx=seg)
            _add_dyn_constraints(accel_cps, a_max, label="amax", seg_idx=seg)
            _add_dyn_constraints(jerk_cps, j_max, label="jmax", seg_idx=seg)
        timing.dynamic_ms = (time.perf_counter() - t_dynamic) * 1000.0

        # ---- Map-size constraints on all Bezier position CPs ----
        t_map = time.perf_counter()
        bounds = [
            (float(self.params.x_min), float(self.params.x_max)),
            (float(self.params.y_min), float(self.params.y_max)),
            (float(self.params.z_min), float(self.params.z_max)),
        ]
        for seg in range(n_seg):
            cp_le = cp_le_per_seg[seg]
            for ax, (lo, hi) in enumerate(bounds):
                if hi > lo:
                    for i in range(4):
                        model.addConstr(cp_le[ax][i] <= hi, name=f"map_s{seg}_a{ax}_cp{i}_hi")
                        model.addConstr(cp_le[ax][i] >= lo, name=f"map_s{seg}_a{ax}_cp{i}_lo")
        timing.mapsize_ms = (time.perf_counter() - t_map) * 1000.0

        # ---- Objective: C++ jerk smoothness ----
        # C++ minimizes jerk_smooth_weight * sum_t ||getJerk(t,0)||^2.
        # There is no segment-duration multiplier.
        t_objective = time.perf_counter()
        obj = gp.QuadExpr()
        jerk_w = max(0.0, float(self.params.jerk_smooth_weight or 1.0))
        for seg in range(n_seg):
            w = 36.0 * jerk_w
            for ax in range(3):
                a_lc = seg_exprs_per_axis[ax].a(seg)
                # a²: sum_i sum_j coef[i]*coef[j]*free[i]*free[j]
                #     + 2*const*sum_i coef[i]*free[i] + const²
                for i, ci in enumerate(a_lc.coef):
                    if ci == 0.0:
                        continue
                    for j, cj in enumerate(a_lc.coef):
                        if cj == 0.0:
                            continue
                        obj.add(
                            free_vars[ax][i] * free_vars[ax][j],
                            w * float(ci * cj),
                        )
                    if a_lc.const != 0.0:
                        obj.add(free_vars[ax][i],
                                w * 2.0 * float(ci) * a_lc.const)
                obj.addConstant(w * a_lc.const * a_lc.const)

        model.setObjective(obj, GRB.MINIMIZE)
        timing.objective_ms = (time.perf_counter() - t_objective) * 1000.0

        t_call = time.perf_counter()
        model.optimize()
        timing.callOptimizer_ms = (time.perf_counter() - t_call) * 1000.0

        if not _has_usable_solution(model, GRB):
            _maybe_debug_infeasibility(
                model, GRB, n_seg, Ts, polys, A, sub_pts,
            )
            return PieceWisePol(), SolverInfo(
                success=False, wall_time_s=time.perf_counter() - t0,
                timing=timing,
            )

        # ---- Extract solution: free vars → coefficients → PieceWisePol ----
        t_post = time.perf_counter()
        free_vals = [
            np.array([free_vars[ax][j].X for j in range(num_free)], dtype=float)
            for ax in range(3)
        ]
        pwp = PieceWisePol()
        t_cum = float(A.t)
        pwp.times.append(t_cum)
        for seg in range(n_seg):
            cx = np.zeros(4); cy = np.zeros(4); cz = np.zeros(4)
            for k in range(4):
                cx[k] = float(np.dot(
                    seg_exprs_per_axis[0].coeffs[seg][k].coef, free_vals[0]
                )) + seg_exprs_per_axis[0].coeffs[seg][k].const
                cy[k] = float(np.dot(
                    seg_exprs_per_axis[1].coeffs[seg][k].coef, free_vals[1]
                )) + seg_exprs_per_axis[1].coeffs[seg][k].const
                cz[k] = float(np.dot(
                    seg_exprs_per_axis[2].coeffs[seg][k].coef, free_vals[2]
                )) + seg_exprs_per_axis[2].coeffs[seg][k].const
            pwp.coeff_x.append(cx)
            pwp.coeff_y.append(cy)
            pwp.coeff_z.append(cz)
            t_cum += Ts[seg]
            pwp.times.append(t_cum)
        timing.postsolve_ms = (time.perf_counter() - t_post) * 1000.0

        return pwp, SolverInfo(
            success=True,
            cost=float(model.ObjVal),
            wall_time_s=time.perf_counter() - t0,
            timing=timing,
        )
    except gp.GurobiError:
        return PieceWisePol(), SolverInfo(
            success=False, wall_time_s=time.perf_counter() - t0,
            timing=timing,
        )


# Bind as methods on GurobiSolver so the call sites stay clean.

def _solve_one_elim(self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0):
    return _solve_elim_core(
        self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
        fresh=False,
    )


def _solve_one_elim_fresh(self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0):
    return _solve_elim_core(
        self, gp, GRB, A, sub_pts, parent_of_seg, polys, Ts, n_seg, t0,
        fresh=True,
    )


GurobiSolver._solve_one_elim = _solve_one_elim
GurobiSolver._solve_one_elim_fresh = _solve_one_elim_fresh


# ---------------------------------------------------------------------------
# Internals — subdivision / time allocation
# ---------------------------------------------------------------------------

def _subdivide(
    pts: List[np.ndarray], params: Parameters,
) -> Tuple[List[np.ndarray], List[int]]:
    """Return exactly ``num_N + 1`` warm-start points along the path.

    C++ fixes the local optimizer segment count to ``par.num_N``. The
    global path is used for corridor generation/selection, not as a set
    of hard interpolation knots. Python still needs per-segment points
    for warm starts and for the optional non-MIQP fallback, so we sample
    the polyline at equal arclength.
    """
    P = len(pts) - 1
    if P <= 0:
        return list(pts), []

    n_target = max(1, int(params.num_N or _DEFAULT_NUM_N))
    lengths = np.array(
        [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(P)],
        dtype=float,
    )
    total = float(lengths.sum())
    if total <= 1e-9:
        return [pts[0].copy() for _ in range(n_target + 1)], [0] * n_target

    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    new_pts: List[np.ndarray] = []
    parent_of_seg: List[int] = []
    for k in range(n_target + 1):
        s = total * k / n_target
        edge = int(np.searchsorted(cum, s, side="right") - 1)
        edge = min(max(edge, 0), P - 1)
        denom = max(lengths[edge], 1e-9)
        alpha = (s - cum[edge]) / denom
        new_pts.append((1.0 - alpha) * pts[edge] + alpha * pts[edge + 1])
        if k > 0:
            mid_s = total * (k - 0.5) / n_target
            parent = int(np.searchsorted(cum, mid_s, side="right") - 1)
            parent_of_seg.append(min(max(parent, 0), P - 1))
    return new_pts, parent_of_seg


def estimate_initial_dt(A: RobotState, goal: np.ndarray, params: Parameters,
                        n_seg: int | None = None) -> float:
    """C++ ``SolverGurobi::getInitialDt`` port.

    It estimates total minimum time from per-axis velocity, acceleration,
    and jerk roots, then divides by ``N``. A missing positive root
    contributes 0, matching ``MinPositiveElement``.
    """
    N = max(1, int(n_seg if n_seg is not None else (params.num_N or _DEFAULT_NUM_N)))
    x0 = np.asarray(A.pos, dtype=float).reshape(3)
    v0 = np.asarray(A.vel, dtype=float).reshape(3)
    a0 = np.asarray(A.accel, dtype=float).reshape(3)
    xf = np.asarray(goal, dtype=float).reshape(3)
    v_max = float(params.v_max)
    a_max = float(params.a_max)
    j_max = float(params.j_max)

    candidates: List[float] = []
    for ax in range(3):
        delta = float(xf[ax] - x0[ax])
        if v_max != 0.0:
            candidates.append(abs(delta) / abs(v_max))

        sign = math.copysign(1.0, delta)
        jerk = sign * j_max
        accel = sign * a_max

        # jerk-limited: (jerk/6)t^3 + (a0/2)t^2 + v0*t + (x0-xf) = 0
        candidates.append(_min_positive_root([
            jerk / 6.0,
            float(a0[ax]) / 2.0,
            float(v0[ax]),
            float(x0[ax] - xf[ax]),
        ]))
        # accel-limited: (accel/2)t^2 + v0*t + (x0-xf) = 0
        candidates.append(_min_positive_root([
            accel / 2.0,
            float(v0[ax]),
            float(x0[ax] - xf[ax]),
        ]))

    initial_dt = max(candidates, default=0.0) / float(N)
    if initial_dt > 10000.0:
        return 0.0
    return float(initial_dt)


def _allocate_times(A: RobotState, goal: np.ndarray, n_seg: int,
                    params: Parameters) -> List[float]:
    """C++ ``findDT`` base durations before applying the factor."""
    initial_dt = estimate_initial_dt(A, goal, params, n_seg)
    base_dt = max(initial_dt, 2.0 * float(params.dc or 0.0))
    return [base_dt for _ in range(n_seg)]


def _min_positive_root(coeff_desc: Sequence[float]) -> float:
    coeff = [float(c) for c in coeff_desc]
    while coeff and abs(coeff[0]) < 1e-12:
        coeff.pop(0)
    if not coeff:
        return 0.0
    roots = np.roots(coeff)
    real_roots = sorted(
        float(r.real)
        for r in roots
        if abs(float(r.imag)) < 1e-7 and float(r.real) > 0.0
    )
    return real_roots[0] if real_roots else 0.0


# ---------------------------------------------------------------------------
# Internals — constraints
# ---------------------------------------------------------------------------

def _add_boundary_constraints(model, coeffs, Ts, A, goal) -> None:
    """Anchor start (A) and end (goal) to fix all 3 derivatives per axis."""
    A_pos = np.asarray(A.pos, dtype=float).reshape(3)
    A_vel = np.asarray(A.vel, dtype=float).reshape(3)
    A_acc = np.asarray(A.accel, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)

    for ax in range(3):
        a0, b0, c0, d0 = coeffs[ax][0]
        # Start: p(0) = A.pos, p'(0) = A.vel, p''(0) = A.accel
        model.addConstr(d0 == float(A_pos[ax]), name=f"start_pos_{ax}")
        model.addConstr(c0 == float(A_vel[ax]), name=f"start_vel_{ax}")
        model.addConstr(2.0 * b0 == float(A_acc[ax]), name=f"start_acc_{ax}")

        # End: p(T) = goal, p'(T) = 0, p''(T) = 0  on the last segment
        n_seg = len(coeffs[ax])
        a, b, c, d = coeffs[ax][n_seg - 1]
        T = Ts[-1]
        model.addConstr(a * T**3 + b * T**2 + c * T + d == float(goal[ax]),
                        name=f"end_pos_{ax}")
        model.addConstr(3 * a * T**2 + 2 * b * T + c == 0.0,
                        name=f"end_vel_{ax}")
        model.addConstr(6 * a * T + 2 * b == 0.0,
                        name=f"end_acc_{ax}")


def _add_continuity_constraints(model, coeffs, Ts) -> None:
    """C0 / C1 / C2 continuity at every internal junction."""
    n_seg = len(coeffs[0])
    for seg in range(n_seg - 1):
        T = Ts[seg]
        for ax in range(3):
            a, b, c, d = coeffs[ax][seg]
            a2, b2, c2, d2 = coeffs[ax][seg + 1]
            # Position
            model.addConstr(a * T**3 + b * T**2 + c * T + d == d2,
                            name=f"C0_seg{seg}_ax{ax}")
            # Velocity
            model.addConstr(3 * a * T**2 + 2 * b * T + c == c2,
                            name=f"C1_seg{seg}_ax{ax}")
            # Acceleration
            model.addConstr(6 * a * T + 2 * b == 2 * b2,
                            name=f"C2_seg{seg}_ax{ax}")


def _add_dynamic_constraints(model, coeffs, Ts, params) -> None:
    """C++ ``setDynamicConstraints`` port for the naive coefficient model."""
    v_max = max(0.0, float(params.v_max))
    a_max = max(0.0, float(params.a_max))
    j_max = max(0.0, float(params.j_max))
    norm_type = (params.dynamic_constraint_type or "Linf").strip()
    l1_signs = [
        (sx, sy, sz)
        for sx in (+1, -1) for sy in (+1, -1) for sz in (+1, -1)
    ]

    def _add(per_axis_cps, bound: float, prefix: str):
        if bound <= 0.0:
            return
        n_cps = len(per_axis_cps[0])
        if norm_type == "L1":
            for k in range(n_cps):
                xs = [per_axis_cps[ax][k] for ax in range(3)]
                for s in l1_signs:
                    model.addConstr(
                        float(s[0]) * xs[0]
                        + float(s[1]) * xs[1]
                        + float(s[2]) * xs[2]
                        <= bound,
                        name=f"{prefix}_L1_cp{k}",
                    )
        elif norm_type == "L2":
            for k in range(n_cps):
                xs = [per_axis_cps[ax][k] for ax in range(3)]
                model.addQConstr(
                    xs[0] * xs[0] + xs[1] * xs[1] + xs[2] * xs[2]
                    <= bound * bound,
                    name=f"{prefix}_L2_cp{k}",
                )
        else:
            for ax in range(3):
                for k, cp in enumerate(per_axis_cps[ax]):
                    model.addConstr(cp <= bound, name=f"{prefix}_max_ax{ax}_cp{k}")
                    model.addConstr(cp >= -bound, name=f"{prefix}_min_ax{ax}_cp{k}")

    n_seg = len(coeffs[0])
    for seg in range(n_seg):
        T = Ts[seg]
        vel_cps = []
        acc_cps = []
        jerk_cps = []
        for ax in range(3):
            a, b, c, _ = coeffs[ax][seg]
            vel_cps.append([
                c,
                c + b * T,
                c + 2.0 * b * T + 3.0 * a * T * T,
            ])
            acc_cps.append([
                2.0 * b,
                2.0 * b + 6.0 * a * T,
            ])
            jerk_cps.append([6.0 * a])
        _add(vel_cps, v_max, f"vel_seg{seg}")
        _add(acc_cps, a_max, f"acc_seg{seg}")
        _add(jerk_cps, j_max, f"jerk_seg{seg}")


def _add_map_size_constraints(model, coeffs, Ts, params) -> None:
    """C++ ``setMapSizeConstraints`` port for Bezier position CPs."""
    bounds = [
        (float(params.x_min), float(params.x_max)),
        (float(params.y_min), float(params.y_max)),
        (float(params.z_min), float(params.z_max)),
    ]
    n_seg = len(coeffs[0])
    for seg in range(n_seg):
        cps = _bezier_pos_cps(coeffs, Ts[seg], seg)
        for ax, (lo, hi) in enumerate(bounds):
            if hi <= lo:
                continue
            for i in range(4):
                model.addConstr(cps[ax][i] <= hi,
                                name=f"map_cp{i}_ax{ax}_max_t{seg}")
                model.addConstr(cps[ax][i] >= lo,
                                name=f"map_cp{i}_ax{ax}_min_t{seg}")


def _bezier_pos_cps(coeffs, T: float, seg: int):
    T2 = T * T
    T3 = T2 * T
    cps = [[None] * 4 for _ in range(3)]
    for ax in range(3):
        a, b, c, d = coeffs[ax][seg]
        cps[ax][0] = d
        cps[ax][1] = d + c * T / 3.0
        cps[ax][2] = d + 2.0 * c * T / 3.0 + b * T2 / 3.0
        cps[ax][3] = d + c * T + b * T2 + a * T3
    return cps


def _add_corridor_constraints(
    gp, GRB, model, coeffs, Ts, parent_of_seg, polys, n_seg,
) -> None:
    """C++ MIQP safe-corridor constraints with Bezier position CPs.

    Any cubic ``p(u) = a u³ + b u² + c u + d`` over ``u in [0, T]`` is
    equivalent to a Bezier curve with control points

        P_0 = d
        P_1 = d + cT/3
        P_2 = d + 2cT/3 + bT²/3
        P_3 = d + cT + bT² + aT³

    The Bezier basis is a partition of unity with non-negative basis
    functions, so the curve lies in ``conv({P_0, P_1, P_2, P_3})``.
    Constraining every CP to lie in the corridor polytope is therefore a
    *sufficient* condition for the entire segment to stay inside —
    strictly tighter than sampling and with fewer constraints.

    Note on basis choice: the C++ also stores a MINVO basis matrix
    (``BasisConverter.A_pos_mv_rest``), but uses it only for **post-solve
    verification** (``checkCollisionViolation`` → MINVO CPs). The
    Gurobi MIQP itself enforces corridor membership through the Bezier
    CPs in ``getCP0..3``. We follow the same split: Bezier for
    enforcement here; MINVO is available in core for verifiers.

    Every CP is constrained under the selected polytope indicator,
    including start/end CPs, matching the C++ implementation.
    """
    if not polys:
        return
    is_time_layered = (
        isinstance(polys, (list, tuple))
        and len(polys) > 0
        and isinstance(polys[0], (list, tuple))
    )

    if is_time_layered:
        N_time = len(polys)
        P = len(polys[0]) if N_time else 0

        def _poly_at(t_sub: int, p: int):
            t_layer = t_sub if n_seg <= N_time else int(t_sub * N_time / n_seg)
            t_layer = min(max(0, t_layer), N_time - 1)
            return polys[t_layer][p]
    else:
        P = len(polys)

        def _poly_at(t_sub: int, p: int):
            return polys[p]

    if P <= 0:
        return

    b_vars = [
        [
            model.addVar(vtype=GRB.BINARY, name=f"b_t{t}_p{p}")
            for p in range(P)
        ]
        for t in range(n_seg)
    ]
    for seg in range(n_seg):
        model.addConstr(
            gp.quicksum(b_vars[seg][p] for p in range(P)) >= 1,
            name=f"at_least_1_pol_t{seg}",
        )
        cps = _bezier_pos_cps(coeffs, Ts[seg], seg)
        for p in range(P):
            poly = _poly_at(seg, p)
            if poly.A.shape[0] == 0:
                model.addConstr(b_vars[seg][p] == 0,
                                name=f"empty_t{seg}_p{p}")
                continue
            for i in range(4):
                for row in range(poly.A.shape[0]):
                    expr = (
                        float(poly.A[row, 0]) * cps[0][i]
                        + float(poly.A[row, 1]) * cps[1][i]
                        + float(poly.A[row, 2]) * cps[2][i]
                    )
                    model.addGenConstrIndicator(
                        b_vars[seg][p], 1, expr, GRB.LESS_EQUAL,
                        float(poly.b[row, 0]),
                        f"sfc_miqp_seg{seg}_p{p}_cp{i}_row{row}",
                    )


# ---------------------------------------------------------------------------
# Internals — warm start, extraction, status check
# ---------------------------------------------------------------------------

def _set_initial_guess(coeffs, sub_pts, Ts, A, params) -> None:
    """Seed the QP with a v1 cubic-Hermite trajectory. Faster convergence
    on warm starts; the QP is convex so it doesn't change the optimum."""
    v_max = max(0.1, float(params.v_max))
    n = len(sub_pts)
    # Junction velocities — same heuristic as v1
    vs = [np.zeros(3) for _ in range(n)]
    vs[0] = np.asarray(A.vel, dtype=float).reshape(3).copy()
    for i in range(1, n - 1):
        dt = Ts[i - 1] + Ts[i]
        if dt > 1e-9:
            v = (sub_pts[i + 1] - sub_pts[i - 1]) / dt
            speed = float(np.linalg.norm(v))
            if speed > v_max:
                v = v * (v_max / speed)
            vs[i] = v

    for i in range(len(Ts)):
        T = Ts[i]
        p0, p1 = sub_pts[i], sub_pts[i + 1]
        v0, v1 = vs[i], vs[i + 1]
        for ax in range(3):
            dp = float(p1[ax] - p0[ax])
            a = (-2.0 * dp + (v0[ax] + v1[ax]) * T) / T**3
            b = (3.0 * dp - (2.0 * v0[ax] + v1[ax]) * T) / T**2
            c = float(v0[ax])
            d = float(p0[ax])
            coeffs[ax][i][0].Start = a
            coeffs[ax][i][1].Start = b
            coeffs[ax][i][2].Start = c
            coeffs[ax][i][3].Start = d


def _has_usable_solution(model, GRB) -> bool:
    """OPTIMAL / SUBOPTIMAL / TIME_LIMIT-with-a-feasible-solution all
    count as success — same lenient policy as the C++ wrapper."""
    if model.Status == GRB.OPTIMAL:
        return True
    if model.Status == GRB.SUBOPTIMAL:
        return True
    if model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        return True
    return False


def _maybe_debug_infeasibility(model, GRB, n_seg, Ts, polys, A, sub_pts):
    """When ``SANDO_SOLVER_DEBUG=1``, dump Gurobi status + run IIS so we
    can see which constraint is forcing infeasibility. No-op otherwise.
    Temporary diagnostic — remove when the dynamic-mode bug is fixed."""
    import os, sys
    if os.environ.get("SANDO_SOLVER_DEBUG") != "1":
        return
    status = model.Status
    sol_count = int(getattr(model, "SolCount", 0))
    is_time_layered = (
        isinstance(polys, (list, tuple)) and len(polys) > 0
        and isinstance(polys[0], (list, tuple))
    )
    if is_time_layered:
        N_time = len(polys); P = len(polys[0]) if N_time else 0
    else:
        N_time = 1; P = len(polys)
    print(
        f"[solver-debug] Status={status} SolCount={sol_count} "
        f"n_seg={n_seg} N_time={N_time} P={P} "
        f"Ts_sum={sum(Ts):.4f}",
        file=sys.stderr,
    )
    if status == GRB.INF_OR_UNBD:
        model.setParam("DualReductions", 0)
        model.optimize()
        status = model.Status
        print(f"[solver-debug] after DualReductions=0, Status={status}",
              file=sys.stderr)
    if status == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            cons_in_iis = [c for c in model.getConstrs() if c.IISConstr]
            gencons_in_iis = [c for c in model.getGenConstrs() if c.IISGenConstr]
            qcs_in_iis = [c for c in model.getQConstrs() if c.IISQConstr] \
                if hasattr(model.getQConstrs()[0] if model.NumQConstrs > 0 else type("x", (), {}), "IISQConstr") else []
            print(
                f"[solver-debug] IIS: {len(cons_in_iis)} lin + "
                f"{len(gencons_in_iis)} indicators + {len(qcs_in_iis)} QC",
                file=sys.stderr,
            )
            for c in cons_in_iis[:20]:
                print(f"  IIS-lin {c.ConstrName}", file=sys.stderr)
            for c in gencons_in_iis[:20]:
                print(f"  IIS-ind {c.GenConstrName}", file=sys.stderr)
        except Exception as e:
            print(f"[solver-debug] IIS failed: {e}", file=sys.stderr)


def _extract_pwp(coeffs, Ts, t_start: float) -> PieceWisePol:
    pwp = PieceWisePol()
    t_cum = float(t_start)
    pwp.times.append(t_cum)
    for seg in range(len(Ts)):
        cx = np.array([coeffs[0][seg][i].X for i in range(4)], dtype=float)
        cy = np.array([coeffs[1][seg][i].X for i in range(4)], dtype=float)
        cz = np.array([coeffs[2][seg][i].X for i in range(4)], dtype=float)
        pwp.coeff_x.append(cx)
        pwp.coeff_y.append(cy)
        pwp.coeff_z.append(cz)
        t_cum += Ts[seg]
        pwp.times.append(t_cum)
    return pwp
