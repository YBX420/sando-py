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
* per-axis ``v_max`` / ``a_max`` / ``j_max`` bounds sampled along each
  segment (``j_max`` is exact — jerk of a cubic is constant per segment
  so one constraint per axis suffices),
* corridor membership (``A x <= b``) via the **Bezier convex-hull
  property** — 4 control points per segment per polytope row, no
  clipping possible between samples. Matches the C++
  ``SolverGurobi::setPolyConsts`` enforcement (the C++ also computes
  MINVO CPs but uses them only for *post-solve verification*, not for
  the Gurobi inequality constraints),
* minimum integrated jerk² objective: ``sum_seg sum_axis 36·a²·T``.

The "field-for-field" promise in ``solver/README.md`` is aspirational —
this implementation produces the same kind of output as the C++ but with
a different (simpler) variable layout. The C++ solver also features
variable elimination, MINVO basis control points for tighter corridor
membership, and a dynamic time-allocation factor; those are follow-ups.

Sub-segmentation
----------------
The C++ uses ``num_N`` polynomial segments distributed over ``num_P``
corridors (default 5 segments, 3 corridors). With only ``len(path) - 1``
polynomial segments, the equality constraints (start + end + C0/C1/C2)
leave no DOF for jerk minimization, so we subdivide each path edge into
``K`` equal sub-segments where ``K`` is chosen to hit a per-axis DOF
target of ``num_N - 3``.

Failure modes
-------------
If Gurobi reports infeasible / unbounded / no solution, the call
returns ``SolverInfo(success=False)`` and the caller holds the previous
trajectory — same contract as v1. We never raise into the replan loop.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from sando_py.core import Parameters, PieceWisePol, Polytope, RobotState

from .elimination import LinearCoeff, SegmentExpressions
from .minvo import accel_minvo_cps, jerk_value, pos_minvo_cps, vel_minvo_cps
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

# Sample points per segment for dynamic (v / a) constraints. Jerk is
# exact (cubic ⇒ constant jerk per segment, one constraint suffices) and
# corridor membership uses the Bezier convex-hull property — no
# sampling, no clipping between samples.
_SAMPLES_PER_SEG = 5

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
        polys: Sequence[Polytope],
    ) -> Tuple[PieceWisePol, SolverInfo]:
        t0 = time.perf_counter()

        if path is None or len(path) < 2:
            return PieceWisePol(), SolverInfo(success=False,
                                              wall_time_s=time.perf_counter() - t0)

        # Import Gurobi lazily; on ImportError, fail cleanly.
        try:
            if self._gp is None:
                import gurobipy as gp
                from gurobipy import GRB
                self._gp = gp
                self._GRB = GRB
        except ImportError:
            return PieceWisePol(), SolverInfo(success=False,
                                              wall_time_s=time.perf_counter() - t0)

        gp = self._gp
        GRB = self._GRB

        pts = [np.asarray(p, dtype=float).reshape(3) for p in path]

        # Subdivide path edges so we have enough polynomial segments.
        sub_pts, parent_of_seg = _subdivide(pts, self.params)
        n_seg = len(sub_pts) - 1

        Ts_base = _allocate_times(sub_pts, self.params)
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
        solve_fn = self._solve_one_elim if use_elim else self._solve_one

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
            _add_corridor_constraints(model, coeffs, Ts, parent_of_seg, polys, n_seg)

            # Objective: integrated jerk². Jerk of a cubic = 6 * a (constant),
            # so integral over [0, T] of jerk² = 36 * a² * T.
            obj = gp.QuadExpr()
            for ax in range(3):
                for seg in range(n_seg):
                    a = coeffs[ax][seg][0]
                    obj += 36.0 * Ts[seg] * a * a
            # Light regularization on b (avoids drift when DOFs are slack)
            jerk_w = max(0.0, float(self.params.jerk_smooth_weight))
            if jerk_w > 0:
                for ax in range(3):
                    for seg in range(n_seg):
                        b_var = coeffs[ax][seg][1]
                        obj += jerk_w * b_var * b_var
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
            _add_corridor_constraints(model, coeffs, Ts, parent_of_seg, polys, n_seg)

            obj = gp.QuadExpr()
            for ax in range(3):
                for seg in range(n_seg):
                    a = coeffs[ax][seg][0]
                    obj += 36.0 * Ts[seg] * a * a
            jerk_w = max(0.0, float(self.params.jerk_smooth_weight))
            if jerk_w > 0:
                for ax in range(3):
                    for seg in range(n_seg):
                        b_var = coeffs[ax][seg][1]
                        obj += jerk_w * b_var * b_var
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

        use_miqp = bool(getattr(self.params, "use_miqp_corridor", False))
        if use_miqp and len(polys) > 1:
            # MIQP path: each segment selects one of the P polytopes via
            # binary vars b[t][p], with "at least one active" enforced.
            # Mirrors C++ ``createSafeCorridorConstraintsForPolytopeAtleastOne``
            # (gurobi_solver.cpp:4034-4089).
            P = len(polys)
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
                    poly = polys[p]
                    if poly.A.shape[0] == 0:
                        # Empty polytope → disable this binary so the
                        # solver can't assign to a non-existent corridor.
                        model.addConstr(b_vars[t][p] == 0,
                                        name=f"empty_t{t}_p{p}")
                        continue
                    for i in range(4):
                        if t == 0 and i == 0:
                            continue
                        if t == n_seg - 1 and i == 3:
                            continue
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
        else:
            # QP path: segment t is hard-pinned to polytope
            # ``parent_of_seg[t]`` (path-edge → corridor identity).
            for seg in range(n_seg):
                parent = parent_of_seg[seg] if seg < len(parent_of_seg) else seg
                if parent >= len(polys):
                    continue
                poly = polys[parent]
                if poly.A.shape[0] == 0:
                    continue
                for i in range(4):
                    if seg == 0 and i == 0:
                        continue
                    if seg == n_seg - 1 and i == 3:
                        continue
                    cp_le = cp_le_per_seg[seg]
                    for row in range(poly.A.shape[0]):
                        a0 = float(poly.A[row, 0])
                        a1 = float(poly.A[row, 1])
                        a2 = float(poly.A[row, 2])
                        lhs = (a0 * cp_le[0][i]
                               + a1 * cp_le[1][i]
                               + a2 * cp_le[2][i])
                        model.addConstr(lhs <= float(poly.b[row, 0]))
        timing.polytopes_ms = (time.perf_counter() - t_polytopes) * 1000.0

        # ---- MINVO dynamic constraints (v, a, j) ----
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

        def _add_dyn_constraints(per_axis_cps, bound: float):
            """``per_axis_cps`` is a list of length 3 (one per axis),
            each holding a list of MINVO CPs (LinearCoeff). Add the
            requested-norm constraints across all CPs."""
            if bound <= 0:
                return
            n_cps = len(per_axis_cps[0])
            if norm_type == "Linf":
                for ax in range(3):
                    for cp in per_axis_cps[ax]:
                        e = _le(cp, ax)
                        model.addConstr(e <= bound)
                        model.addConstr(e >= -bound)
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
            # Gather MINVO CPs per axis (each axis → list of CPs).
            vel_cps = [None] * 3
            accel_cps = [None] * 3
            jerk_cps = [None] * 3
            for ax in range(3):
                a_lc = seg_exprs_per_axis[ax].a(seg)
                b_lc = seg_exprs_per_axis[ax].b(seg)
                c_lc = seg_exprs_per_axis[ax].c(seg)
                vel_cps[ax] = vel_minvo_cps(a_lc, b_lc, c_lc, T)
                accel_cps[ax] = accel_minvo_cps(a_lc, b_lc, T)
                jerk_cps[ax] = [jerk_value(a_lc)]  # single jerk CP per segment

            _add_dyn_constraints(vel_cps, v_max)
            _add_dyn_constraints(accel_cps, a_max)
            _add_dyn_constraints(jerk_cps, j_max)
        timing.dynamic_ms = (time.perf_counter() - t_dynamic) * 1000.0

        # ---- Objective: integrated jerk² ----
        # jerk(u) = 6 a (constant per segment), integral over [0, T] of
        # jerk² is 36 * a² * T. ``a`` is a LinearCoeff so a² is a
        # quadratic form in the free vars.
        t_objective = time.perf_counter()
        obj = gp.QuadExpr()
        for seg in range(n_seg):
            T = Ts[seg]
            w = 36.0 * T
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

        # mapsize_ms: Python doesn't emit explicit map-bbox constraints
        # at this layer (handled by HGP + ellipsoid local_bbox), so 0.
        timing.mapsize_ms = 0.0

        t_call = time.perf_counter()
        model.optimize()
        timing.callOptimizer_ms = (time.perf_counter() - t_call) * 1000.0

        if not _has_usable_solution(model, GRB):
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
    """Subdivide each path edge into ``K`` equal sub-segments so the QP
    has enough DOF for jerk minimization.

    Returns the new waypoint list and, for each sub-segment, the index
    of the parent path edge (used to look up the corresponding corridor
    polytope).
    """
    P = len(pts) - 1
    if P <= 0:
        return list(pts), []

    n_target = max(_DEFAULT_NUM_N, int(params.num_N or 0))
    # Per-axis DOF = n_target - 3 (3 start + 3 end equality constraints,
    # 3*(n-1) continuity, 4n vars). We aim for at least 3 DOFs so the
    # optimizer has something meaningful to do, which means n >= 6.
    K = max(1, (n_target + P - 1) // P)

    new_pts: List[np.ndarray] = [pts[0].copy()]
    parent_of_seg: List[int] = []
    for i in range(P):
        p0, p1 = pts[i], pts[i + 1]
        for k in range(1, K + 1):
            alpha = k / K
            new_pts.append((1.0 - alpha) * p0 + alpha * p1)
            parent_of_seg.append(i)
    return new_pts, parent_of_seg


def _allocate_times(pts: List[np.ndarray], params: Parameters) -> List[float]:
    v_max = max(0.1, float(params.v_max))
    cruise = max(_T_MIN * v_max, _CRUISE_ALPHA * v_max)
    Ts = []
    for i in range(len(pts) - 1):
        L = float(np.linalg.norm(pts[i + 1] - pts[i]))
        Ts.append(max(_T_MIN, L / cruise))
    return Ts


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
    """Per-axis ``v_max`` / ``a_max`` / ``j_max`` (Linf) at sampled points
    along each segment.

    Jerk of a cubic is constant per segment (= 6 * a) so one constraint
    per segment per axis is exact, not a sample.
    """
    v_max = max(0.0, float(params.v_max))
    a_max = max(0.0, float(params.a_max))
    j_max = max(0.0, float(params.j_max))

    n_seg = len(coeffs[0])
    K = max(1, _SAMPLES_PER_SEG)
    for seg in range(n_seg):
        T = Ts[seg]
        for ax in range(3):
            a, b, c, _ = coeffs[ax][seg]
            # Jerk: exact (constant) — one pair of constraints per segment
            if j_max > 0:
                model.addConstr(6.0 * a <= j_max,
                                name=f"jmax_seg{seg}_ax{ax}")
                model.addConstr(6.0 * a >= -j_max,
                                name=f"jmin_seg{seg}_ax{ax}")
            # Sample v / a at K+1 points per segment
            for k in range(K + 1):
                u = T * k / K
                if v_max > 0:
                    vel = 3.0 * a * u * u + 2.0 * b * u + c
                    model.addConstr(vel <= v_max,
                                    name=f"vmax_seg{seg}_ax{ax}_k{k}")
                    model.addConstr(vel >= -v_max,
                                    name=f"vmin_seg{seg}_ax{ax}_k{k}")
                if a_max > 0:
                    acc = 6.0 * a * u + 2.0 * b
                    model.addConstr(acc <= a_max,
                                    name=f"amax_seg{seg}_ax{ax}_k{k}")
                    model.addConstr(acc >= -a_max,
                                    name=f"amin_seg{seg}_ax{ax}_k{k}")


def _add_corridor_constraints(
    model, coeffs, Ts, parent_of_seg, polys, n_seg,
) -> None:
    """Constrain each cubic segment to lie in its parent path-edge's
    polytope using **Bezier convex-hull control points** (matches the
    C++ ``SolverGurobi::setPolyConsts`` / ``getCP0..3``).

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

    The boundary CPs ``P_0`` of segment 0 and ``P_3`` of the last
    segment equal the fixed start / end positions, so we skip them — if
    ``A.pos`` or ``goal`` happens to sit on a corridor boundary we
    don't want a spurious infeasibility from numerical noise.
    """
    if not polys:
        return
    for seg in range(n_seg):
        parent = parent_of_seg[seg] if seg < len(parent_of_seg) else seg
        if parent >= len(polys):
            continue
        poly = polys[parent]
        if poly.A.shape[0] == 0:
            continue
        T = Ts[seg]
        T2 = T * T
        T3 = T2 * T
        # Per-axis Bezier control points (each is a linear expression
        # in the cubic coefficient variables).
        cps = [[None] * 4 for _ in range(3)]
        for ax in range(3):
            a, b, c, d = coeffs[ax][seg]
            cps[ax][0] = d
            cps[ax][1] = d + c * T / 3.0
            cps[ax][2] = d + 2.0 * c * T / 3.0 + b * T2 / 3.0
            cps[ax][3] = d + c * T + b * T2 + a * T3

        for i in range(4):
            if seg == 0 and i == 0:
                continue
            if seg == n_seg - 1 and i == 3:
                continue
            for row in range(poly.A.shape[0]):
                expr = (
                    float(poly.A[row, 0]) * cps[0][i]
                    + float(poly.A[row, 1]) * cps[1][i]
                    + float(poly.A[row, 2]) * cps[2][i]
                )
                model.addConstr(expr <= float(poly.b[row, 0]),
                                name=f"sfc_bez_seg{seg}_cp{i}_row{row}")


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
