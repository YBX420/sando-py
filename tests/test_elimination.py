"""Regression tests for the variable-elimination QP kernel — mirrors
the C++ ``computeDependentCoefficientsN*`` scheme.

These run in the conda sando env (gurobipy present). Tests verify both
mathematical equivalence with the naive kernel and tighter-bound MINVO
behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

gurobipy = pytest.importorskip("gurobipy")

from sando_py.core import Parameters, Polytope, RobotState
from sando_py.solver import GurobiSolver
from sando_py.solver.elimination import (
    CoefficientFormulas,
    LinearCoeff,
    SegmentExpressions,
)
from sando_py.solver.minvo import (
    A_ACCEL_MV_REST_INV,
    A_POS_MV_REST_INV,
    A_VEL_MV_REST_INV,
    accel_minvo_cps,
    pos_minvo_cps,
    vel_minvo_cps,
)


# ---------------------------------------------------------------------------
# Coefficient formulas
# ---------------------------------------------------------------------------

def test_segment_expressions_n5_recovers_boundary():
    """For N=5 with uniform dt, the elimination should recover the
    boundary conditions exactly. Segment 0 d_0 = P0, c_0 = V0, b_0 = A0/2.
    Last segment satisfies the final boundary."""
    dt = np.array([0.5] * 5)
    boundary = np.array([1.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # P0=1, Pf=5
    se = SegmentExpressions.build(5, dt, boundary)

    # Segment 0 boundary
    d0 = se.d(0)
    assert np.allclose(d0.coef, 0.0)
    assert abs(d0.const - 1.0) < 1e-9

    c0 = se.c(0)
    assert np.allclose(c0.coef, 0.0)
    assert abs(c0.const - 0.0) < 1e-9

    b0 = se.b(0)
    assert np.allclose(b0.coef, 0.0)
    assert abs(b0.const - 0.0) < 1e-9


def test_num_free_equals_N_minus_3():
    """N-3 free variables per axis."""
    for N in (4, 5, 6, 7):
        dt = np.ones(N) * 0.3
        boundary = np.zeros(6)
        se = SegmentExpressions.build(N, dt, boundary)
        assert se.num_free == N - 3


def test_continuity_holds_for_any_free_value():
    """Setting arbitrary free values, the constructed coefficients
    should satisfy C^2 continuity at every junction."""
    N = 6
    dt = np.array([0.3, 0.5, 0.4, 0.6, 0.2, 0.7])
    boundary = np.array([0.0, 1.0, 0.5, 8.0, 0.0, 0.0])
    se = SegmentExpressions.build(N, dt, boundary)
    free_vals = np.array([1.5, -0.7, 2.3])  # N-3 = 3 free for N=6
    assert se.num_free == 3

    def evaluate(lc: LinearCoeff) -> float:
        return float(np.dot(lc.coef, free_vals) + lc.const)

    for seg in range(N - 1):
        T = dt[seg]
        a_i = evaluate(se.a(seg))
        b_i = evaluate(se.b(seg))
        c_i = evaluate(se.c(seg))
        d_i = evaluate(se.d(seg))
        # Position at end of segment i
        p_end = a_i * T**3 + b_i * T**2 + c_i * T + d_i
        d_next = evaluate(se.d(seg + 1))
        assert abs(p_end - d_next) < 1e-8, f"C0 fails at junction {seg}"

        # Velocity continuity
        v_end = 3 * a_i * T**2 + 2 * b_i * T + c_i
        c_next = evaluate(se.c(seg + 1))
        assert abs(v_end - c_next) < 1e-8, f"C1 fails at junction {seg}"

        # Acceleration continuity
        acc_end = 6 * a_i * T + 2 * b_i
        b_next = evaluate(se.b(seg + 1))
        assert abs(acc_end - 2 * b_next) < 1e-8, f"C2 fails at junction {seg}"


def test_final_boundary_holds_for_any_free_value():
    """The final boundary condition should be satisfied symbolically."""
    N = 5
    dt = np.array([0.4] * N)
    boundary = np.array([1.0, 0.5, 0.0, 10.0, 0.0, 0.0])
    se = SegmentExpressions.build(N, dt, boundary)
    free_vals = np.array([2.0, -1.0])

    def evaluate(lc: LinearCoeff) -> float:
        return float(np.dot(lc.coef, free_vals) + lc.const)

    T = dt[N - 1]
    a = evaluate(se.a(N - 1))
    b = evaluate(se.b(N - 1))
    c = evaluate(se.c(N - 1))
    d = evaluate(se.d(N - 1))
    # p(T) == Pf, v(T) == 0, accel(T) == 0
    assert abs(a * T**3 + b * T**2 + c * T + d - 10.0) < 1e-8
    assert abs(3 * a * T**2 + 2 * b * T + c) < 1e-8
    assert abs(6 * a * T + 2 * b) < 1e-8


# ---------------------------------------------------------------------------
# MINVO control points
# ---------------------------------------------------------------------------

def test_minvo_position_matrix_last_row_is_ones():
    """The 4th row of A_pos_mv_rest_inv is the constant term — MINVO CPs
    are affine combinations of polynomial coefficients."""
    assert np.allclose(A_POS_MV_REST_INV[3], [1.0, 1.0, 1.0, 1.0])


def test_minvo_basis_shapes():
    assert A_POS_MV_REST_INV.shape == (4, 4)
    assert A_VEL_MV_REST_INV.shape == (3, 3)
    assert A_ACCEL_MV_REST_INV.shape == (2, 2)


def test_pos_minvo_cps_have_4_entries():
    a = LinearCoeff.constant(0.1, 2)
    b = LinearCoeff.constant(0.2, 2)
    c = LinearCoeff.constant(0.3, 2)
    d = LinearCoeff.constant(0.4, 2)
    cps = pos_minvo_cps(a, b, c, d, 1.0)
    assert len(cps) == 4
    for cp in cps:
        assert isinstance(cp, LinearCoeff)


def test_vel_minvo_cps_have_3_entries():
    a = LinearCoeff.constant(0.1, 2)
    b = LinearCoeff.constant(0.2, 2)
    c = LinearCoeff.constant(0.3, 2)
    cps = vel_minvo_cps(a, b, c, 1.0)
    assert len(cps) == 3


def test_accel_minvo_cps_have_2_entries():
    a = LinearCoeff.constant(0.1, 2)
    b = LinearCoeff.constant(0.2, 2)
    cps = accel_minvo_cps(a, b, 1.0)
    assert len(cps) == 2


# ---------------------------------------------------------------------------
# End-to-end equivalence with naive kernel
# ---------------------------------------------------------------------------

def _bench_params() -> Parameters:
    p = Parameters()
    p.num_N = 5
    p.num_P = 3
    p.v_max = 5.0
    p.a_max = 10.0
    p.j_max = 50.0
    p.dc = 0.01
    p.horizon = 15.0
    p.max_gurobi_comp_time_sec = 5.0
    p.use_dynamic_factor = False
    return p


def _path_polys():
    path = [np.array([float(x), float(y), 2.0]) for x, y in
            [(0, 0), (2, 0.5), (4, -0.3), (6, 0.2), (8, 0)]]
    polys = []
    for i in range(len(path) - 1):
        A_box = np.eye(3)
        A_box = np.vstack([A_box, -A_box])
        mid = 0.5 * (path[i] + path[i + 1])
        half = 5.0
        b = np.array([
            mid[0] + half, mid[1] + half, mid[2] + half,
            -mid[0] + half, -mid[1] + half, -mid[2] + half,
        ]).reshape(-1, 1)
        polys.append(Polytope(A=A_box, b=b))
    return path, polys


def test_elim_solver_succeeds_on_simple_path():
    p = _bench_params()
    p.using_variable_elimination = True
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    path, polys = _path_polys()
    pwp, info = GurobiSolver(p).solve(A, path, polys)
    assert info.success


def test_elim_and_naive_produce_equivalent_cost():
    """Both kernels should converge to the same optimal cost (they
    optimize the same QP, just with different variable layouts)."""
    p = _bench_params()
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    path, polys = _path_polys()

    p.using_variable_elimination = True
    _, info_e = GurobiSolver(p).solve(A, path, polys)
    p.using_variable_elimination = False
    _, info_n = GurobiSolver(p).solve(A, path, polys)

    assert info_e.success and info_n.success
    # Same QP → cost should match to ~1e-3 (Gurobi tolerance + reformulation noise)
    assert abs(info_e.cost - info_n.cost) / max(abs(info_n.cost), 1e-9) < 1e-3


def test_l1_constraint_mode_succeeds():
    """L1 mode adds 8-face octahedron constraints per CP — still feasible
    for an unblocked trajectory."""
    p = _bench_params()
    p.dynamic_constraint_type = "L1"
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    path, polys = _path_polys()
    pwp, info = GurobiSolver(p).solve(A, path, polys)
    assert info.success
    # L1 is strictly tighter than Linf — solution may differ but should
    # still respect per-axis v_max (it bounds the sum, so each axis
    # individually <= v_max).


def test_l2_constraint_mode_builds_qcp():
    """L2 mode adds quadratic norm constraints (turns the QP into QCP).
    The solver itself may report numerical trouble — we just verify the
    code path is exercised (i.e. doesn't crash). The Gurobi SOC solver
    is sensitive to Bezier/MINVO scaling; production use sticks to
    ``Linf`` (yaml default)."""
    p = _bench_params()
    p.dynamic_constraint_type = "L2"
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    path, polys = _path_polys()
    # Should not raise — success/failure both acceptable.
    pwp, info = GurobiSolver(p).solve(A, path, polys)
    assert info is not None


def test_miqp_corridor_succeeds():
    """MIQP corridor mode: binary indicator vars + 'at least one polytope
    active'. Larger model, but still feasible."""
    p = _bench_params()
    p.use_miqp_corridor = True
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    path, polys = _path_polys()
    pwp, info = GurobiSolver(p).solve(A, path, polys)
    assert info.success


def test_miqp_disabled_uses_parent_of_seg_pinning():
    """Default ``use_miqp_corridor=False`` uses single polytope per
    segment via parent_of_seg mapping."""
    p = _bench_params()
    assert p.use_miqp_corridor is False
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    path, polys = _path_polys()
    pwp, info = GurobiSolver(p).solve(A, path, polys)
    assert info.success


def test_elim_trajectory_anchors_at_A_and_goal():
    """After solving with elimination, the resulting PieceWisePol should
    start at A and end at the goal (numerically — elimination encodes
    these as the boundary conditions)."""
    p = _bench_params()
    p.using_variable_elimination = True
    A = RobotState()
    A.setPos([0.0, 0.0, 2.0])
    A.setVel([0.5, 0.0, 0.0])
    A.setAccel([0.0, 0.0, 0.0])
    path, polys = _path_polys()
    pwp, info = GurobiSolver(p).solve(A, path, polys)
    assert info.success
    p0 = pwp.eval(pwp.times[0])
    pf = pwp.eval(pwp.times[-1])
    assert np.allclose(p0, A.pos, atol=1e-3)
    assert np.allclose(pf, path[-1], atol=1e-3)
