"""Tests for sando_py.solver.GurobiSolver — v2 QP optimizer."""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole file when gurobipy isn't importable. The Gurobi license
# check is implicit: if Gurobi can't get a license, ``model.optimize()``
# fails inside the solver and ``info.success`` comes back False — those
# tests would fail loudly, which is the right behaviour (CI without a
# license should NOT silently pass them).
gp = pytest.importorskip("gurobipy")

from sando_py.core import Parameters, PieceWisePol, Polytope, RobotState
from sando_py.solver import GurobiSolver, SolverInfo, solve_qp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _params(**overrides) -> Parameters:
    p = Parameters()
    p.v_max = 3.0
    p.a_max = 8.0
    p.j_max = 50.0
    p.dc = 0.01
    p.num_N = 6
    p.max_gurobi_comp_time_sec = 0.0  # no per-solve time limit in tests
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _state(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), accel=(0.0, 0.0, 0.0),
           t: float = 0.0) -> RobotState:
    s = RobotState()
    s.setPos(*pos)
    s.setVel(*vel)
    s.setAccel(*accel)
    s.setTimeStamp(t)
    return s


def _box_polytope(center, half=(2.0, 2.0, 2.0)) -> Polytope:
    """Axis-aligned box centered at ``center``."""
    c = np.asarray(center, dtype=float).reshape(3)
    h = np.asarray(half, dtype=float).reshape(3)
    A = np.zeros((6, 3))
    b = np.zeros((6, 1))
    for i in range(3):
        A[i, i] = 1.0
        b[i, 0] = c[i] + h[i]
        A[3 + i, i] = -1.0
        b[3 + i, 0] = -(c[i] - h[i])
    return Polytope(A=A, b=b)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_solve_qp_empty_path_returns_failure():
    pwp, info = solve_qp(_state(), [], [], _params())
    assert isinstance(info, SolverInfo)
    assert info.success is False


def test_solve_qp_singleton_path_returns_failure():
    pwp, info = solve_qp(_state(), [np.zeros(3)], [], _params())
    assert info.success is False


# ---------------------------------------------------------------------------
# Single-edge problem — boundary conditions
# ---------------------------------------------------------------------------

def test_solve_qp_matches_start_pos_vel_accel():
    """v2's headline win over v1: start acceleration is preserved."""
    A = _state(pos=(0.0, 0.0, 2.0), vel=(0.5, 0.0, 0.0), accel=(0.3, 0.0, 0.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0])]
    pwp, info = solve_qp(A, path, [], _params())
    assert info.success is True

    t0 = pwp.times[0]
    assert np.allclose(pwp.eval(t0), A.pos, atol=1e-6)
    assert np.allclose(pwp.velocity(t0), A.vel, atol=1e-6)
    assert np.allclose(pwp.acceleration(t0), A.accel, atol=1e-6)


def test_solve_qp_matches_end_state_at_zero_vel_accel():
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0])]
    pwp, info = solve_qp(A, path, [], _params())
    assert info.success is True
    t_end = pwp.times[-1]
    assert np.allclose(pwp.eval(t_end), path[-1], atol=1e-6)
    assert np.allclose(pwp.velocity(t_end), 0.0, atol=1e-5)
    assert np.allclose(pwp.acceleration(t_end), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------

def test_solve_qp_is_C2_continuous_at_junctions():
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [
        np.array([0.0, 0.0, 2.0]),
        np.array([3.0, 1.0, 2.0]),
        np.array([6.0, 0.0, 2.0]),
    ]
    pwp, info = solve_qp(A, path, [], _params())
    assert info.success is True

    # Eval pos / vel / accel just before and after each interior knot
    for t_knot in pwp.times[1:-1]:
        eps = 1e-6
        assert np.allclose(pwp.eval(t_knot - eps), pwp.eval(t_knot + eps), atol=1e-3)
        assert np.allclose(pwp.velocity(t_knot - eps),
                           pwp.velocity(t_knot + eps), atol=1e-3)
        assert np.allclose(pwp.acceleration(t_knot - eps),
                           pwp.acceleration(t_knot + eps), atol=1e-3)


# ---------------------------------------------------------------------------
# Dynamic limits
# ---------------------------------------------------------------------------

def test_solve_qp_respects_v_max():
    """With a tight v_max, the resulting velocities should be bounded
    (with a small tolerance for sample-spacing error)."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    p = _params(v_max=1.0)
    pwp, info = solve_qp(A, p_path := path, [], p)
    assert info.success is True
    # Sample velocities across the trajectory
    t0, t1 = pwp.times[0], pwp.times[-1]
    ts = np.linspace(t0, t1, 50)
    vs = np.stack([pwp.velocity(t) for t in ts])
    # Linf bound per axis
    assert vs[:, :3].max() <= p.v_max + 0.2
    assert vs[:, :3].min() >= -p.v_max - 0.2


def test_solve_qp_respects_j_max():
    """Jerk of a cubic is constant per segment — when the bound is loose
    enough to be feasible, every segment's jerk must hold |6a| <= j_max."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([2.0, 0.5, 2.0])]
    p = _params(j_max=30.0)
    pwp, info = solve_qp(A, path, [], p)
    assert info.success is True
    # Jerk per segment per axis = 6 * coeff_x[0]
    for seg in range(len(pwp.coeff_x)):
        for c in (pwp.coeff_x, pwp.coeff_y, pwp.coeff_z):
            assert abs(6.0 * c[seg][0]) <= p.j_max + 1e-6


def test_solve_qp_fails_gracefully_when_j_max_too_tight():
    """Very tight j_max for a long rest-to-rest move is infeasible — the
    QP must report failure, not raise."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    p = _params(j_max=0.5)  # very tight
    _, info = solve_qp(A, path, [], p)
    assert info.success is False


# ---------------------------------------------------------------------------
# Corridor constraints
# ---------------------------------------------------------------------------

def test_solve_qp_keeps_trajectory_inside_corridor():
    """Tight corridor along the x-axis: y-excursions must be bounded by
    the corridor's y-half-width, not v_max-driven overshoot."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    # Tight tube of y-half-width 0.5
    poly = _box_polytope(center=(5.0, 0.0, 2.0), half=(10.0, 0.5, 5.0))
    pwp, info = solve_qp(A, path, [poly], _params())
    assert info.success is True

    ts = np.linspace(pwp.times[0], pwp.times[-1], 30)
    for t in ts:
        p = pwp.eval(t)
        assert -0.5 - 0.05 <= p[1] <= 0.5 + 0.05, f"y={p[1]} out of corridor at t={t}"


def test_bezier_corridor_no_clipping_with_dense_sampling():
    """Bezier convex-hull guarantee: corridor membership holds at every
    point on the curve, not just at the samples used to add
    constraints. Densely sample the resulting trajectory and check.

    A bent path through a tight tube is the case where a sample-based
    approach could clip between samples; the Bezier-CP constraints rule
    it out by the convex-hull property."""
    A = _state(pos=(0.0, -0.4, 2.0), vel=(2.0, 0.0, 0.0))
    path = [
        np.array([0.0, -0.4, 2.0]),
        np.array([5.0, 0.4, 2.0]),
        np.array([10.0, -0.4, 2.0]),
    ]
    # Two tight tubes that just barely contain the waypoints
    polys = [
        _box_polytope(center=(2.5, 0.0, 2.0), half=(3.0, 0.5, 5.0)),
        _box_polytope(center=(7.5, 0.0, 2.0), half=(3.0, 0.5, 5.0)),
    ]
    pwp, info = solve_qp(A, path, polys, _params(v_max=4.0))
    assert info.success is True

    # Sample the trajectory at 200 points across its duration; every one
    # must satisfy its segment's corridor (within a tiny tolerance).
    ts = np.linspace(pwp.times[0], pwp.times[-1], 200)
    for t in ts:
        p = pwp.eval(t)
        # Which path-segment polytope does t fall into?
        in_some = False
        for poly in polys:
            if (poly.A @ p <= poly.b[:, 0] + 1e-4).all():
                in_some = True
                break
        assert in_some, f"trajectory at t={t}, p={p} lies outside all corridors"


# ---------------------------------------------------------------------------
# Infeasible inputs
# ---------------------------------------------------------------------------

def test_solve_qp_returns_failure_when_infeasible():
    """A corridor that excludes the goal must mark the QP infeasible —
    not raise."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    # Corridor that doesn't include x=10
    poly = _box_polytope(center=(2.0, 0.0, 2.0), half=(1.0, 1.0, 1.0))
    pwp, info = solve_qp(A, path, [poly], _params())
    assert info.success is False


# ---------------------------------------------------------------------------
# PieceWisePol shape
# ---------------------------------------------------------------------------

def test_solve_qp_pwp_has_expected_structure():
    """times has n+1 entries, coeff_{x,y,z} have n entries of length 4 each."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0])]
    pwp, info = solve_qp(A, path, [], _params())
    assert info.success is True
    assert isinstance(pwp, PieceWisePol)
    n = len(pwp.coeff_x)
    assert n > 0
    assert len(pwp.times) == n + 1
    assert len(pwp.coeff_y) == n
    assert len(pwp.coeff_z) == n
    for c in pwp.coeff_x:
        assert c.shape == (4,)


def test_solve_qp_respects_A_t_as_start_time():
    A = _state(pos=(0.0, 0.0, 2.0), t=10.0)
    path = [np.array([0.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0])]
    pwp, info = solve_qp(A, path, [], _params())
    assert info.success is True
    assert pwp.times[0] == pytest.approx(10.0)
    assert pwp.times[-1] > 10.0


# ---------------------------------------------------------------------------
# Class form parity
# ---------------------------------------------------------------------------

def test_class_form_matches_functional_form():
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0])]
    p = _params()
    _, info_fn = solve_qp(A, path, [], p)
    _, info_cl = GurobiSolver(p).solve(A, path, [])
    assert info_fn.success == info_cl.success is True


# ---------------------------------------------------------------------------
# Time-allocation adapter integration
# ---------------------------------------------------------------------------

def test_dynamic_factor_recovers_from_too_tight_v_max():
    """At factor=1.0 the QP is infeasible (rest-to-rest 10 m in the
    factor=1 time budget exceeds v_max). With ``use_dynamic_factor`` the
    adapter should retry at higher factors and find a feasible
    trajectory."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    p = _params(
        v_max=1.0,                  # tight
        use_dynamic_factor=True,
        dynamic_factor_initial_mean=1.0,
        factor_initial=1.0,
        factor_final=4.0,
        factor_constant_step_size=0.5,
        dynamic_factor_k_radius=0.0,  # single-element window
    )
    solver = GurobiSolver(p)
    _, info = solver.solve(A, path, [])
    # Call once more if first didn't succeed — the adapter shifts the
    # window after a failed solve.
    if not info.success:
        _, info = solver.solve(A, path, [])
    assert info.success is True


def test_parallel_factor_search_picks_lowest_successful():
    """With a multi-factor window, the parallel path picks the lowest
    factor that yields a feasible solve. Mirrors the C++
    ``tryToCompleteOpt`` pattern: all candidates run concurrently and
    the lowest successful one wins.

    Setting dynamics generously so every factor in the window is
    feasible — the test then verifies the adapter recenters on the
    *lowest* one (= ``factor_initial`` in this configuration).
    """
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    p = _params(
        v_max=20.0,
        a_max=20.0,
        j_max=500.0,
        use_dynamic_factor=True,
        dynamic_factor_initial_mean=2.5,
        factor_initial=1.5,
        factor_final=3.5,
        factor_constant_step_size=0.5,
        dynamic_factor_k_radius=1.0,      # window: 1.5..3.5 (5 factors)
    )
    solver = GurobiSolver(p)
    pwp, info = solver.solve(A, path, [])
    assert info.success is True
    # The adapter recenters on the *lowest* feasible factor. We don't
    # pin the exact value — the cubic-spline feasibility floor is
    # problem-specific — but it must be strictly less than the initial
    # mean, proving the parallel search did find a feasible candidate
    # lower than where we started.
    assert solver.time_alloc.current_mean < p.dynamic_factor_initial_mean


def test_parallel_factor_search_returns_failure_when_all_factors_infeasible():
    """If every factor in the window leaves the QP infeasible, the
    parallel path returns failure (mirrors C++ behaviour) and the
    adapter shifts the window up for the next call."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    p = _params(
        v_max=10.0,
        a_max=0.05,          # absurdly tight: infeasible for any plausible time budget
        j_max=200.0,
        use_dynamic_factor=True,
        dynamic_factor_initial_mean=1.5,
        factor_initial=1.0,
        factor_final=2.0,
        factor_constant_step_size=0.5,
        dynamic_factor_k_radius=0.5,      # window: 1.0..2.0 (3 factors)
    )
    solver = GurobiSolver(p)
    mean_before = solver.time_alloc.current_mean
    _, info = solver.solve(A, path, [])
    assert info.success is False
    assert solver.time_alloc.current_mean > mean_before - 1e-6  # adapter shifted up


def test_disabled_adapter_only_tries_one_factor():
    """With ``use_dynamic_factor=False`` the solver never retries.
    Confirm it returns the expected single result on a problem where
    the dynamic adapter would otherwise have helped."""
    A = _state(pos=(0.0, 0.0, 2.0))
    path = [np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])]
    # a_max too tight for the factor=1 time budget. Dynamic adapter
    # would step up; static stays at factor=1 and fails.
    p = _params(
        v_max=10.0,
        a_max=0.3,
        j_max=200.0,
        use_dynamic_factor=False,
        dynamic_factor_initial_mean=1.0,
    )
    _, info = solve_qp(A, path, [], p)
    assert info.success is False
