"""Tests for sando_py.solver — v1 cubic-Hermite trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from sando_py.core import Parameters, PieceWisePol, RobotState
from sando_py.solver import MinJerkSolver, SolverInfo, solve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _solver_params(**overrides) -> Parameters:
    p = Parameters()
    p.v_max = 2.0
    p.a_max = 5.0
    p.j_max = 30.0
    p.dc = 0.01
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _at_origin(vel=(0.0, 0.0, 0.0), t: float = 0.0) -> RobotState:
    s = RobotState()
    s.setPos(0.0, 0.0, 0.0)
    s.setVel(*vel)
    s.setTimeStamp(t)
    return s


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_solve_empty_path_returns_failure():
    pwp, info = solve(_at_origin(), [], [], _solver_params())
    assert isinstance(info, SolverInfo)
    assert info.success is False
    assert info.wall_time_s >= 0.0
    assert pwp.times == []


def test_solve_singleton_path_returns_failure():
    pwp, info = solve(_at_origin(), [np.zeros(3)], [], _solver_params())
    assert info.success is False
    assert pwp.times == []


# ---------------------------------------------------------------------------
# Single-segment Hermite — boundary conditions
# ---------------------------------------------------------------------------

def test_solve_single_segment_matches_endpoints_and_velocities():
    """Single segment from origin to (10, 0, 2), starting at rest, ending
    at rest. Eval at boundaries must match exactly to ~1e-9."""
    A = _at_origin()
    path = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 2.0])]
    pwp, info = solve(A, path, [], _solver_params())
    assert info.success is True
    assert len(pwp.times) == 2
    assert len(pwp.coeff_x) == 1

    t0, t1 = pwp.times
    # Positions at the two endpoints
    assert np.allclose(pwp.eval(t0), [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(pwp.eval(t1), [10.0, 0.0, 2.0], atol=1e-9)
    # Boundary velocities: start = A.vel = 0, end = hover = 0
    assert np.allclose(pwp.velocity(t0), [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(pwp.velocity(t1), [0.0, 0.0, 0.0], atol=1e-9)


def test_solve_single_segment_starts_with_A_velocity():
    """Boundary velocity at t = A.t must equal the agent's current vel."""
    A = _at_origin(vel=(1.0, 0.5, 0.0))
    path = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]
    pwp, info = solve(A, path, [], _solver_params())
    assert info.success is True
    v0 = pwp.velocity(pwp.times[0])
    assert np.allclose(v0, [1.0, 0.5, 0.0], atol=1e-9)


def test_solve_respects_start_time():
    """times[0] should equal A.t, not 0."""
    A = _at_origin(t=42.5)
    path = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]
    pwp, info = solve(A, path, [], _solver_params())
    assert info.success is True
    assert pwp.times[0] == pytest.approx(42.5)
    assert pwp.times[-1] > 42.5


# ---------------------------------------------------------------------------
# Multi-segment continuity
# ---------------------------------------------------------------------------

def test_solve_multi_segment_C1_continuous_at_junctions():
    """Position and velocity must match across segment joins (C1
    continuity)."""
    A = _at_origin()
    path = [
        np.array([0.0, 0.0, 1.0]),
        np.array([3.0, 1.0, 1.0]),
        np.array([6.0, 0.0, 1.0]),
        np.array([9.0, 1.0, 1.0]),
    ]
    pwp, info = solve(A, path, [], _solver_params())
    assert info.success is True
    assert len(pwp.times) == len(path)  # n+1 knots

    # At each interior knot, eval from the left side (the previous
    # segment evaluated at its end) and from t directly. PieceWisePol
    # uses [t_i, t_i+1) intervals, so eval(t_i) is the *start* of
    # segment i. To get the end of segment i-1 we eval at t_i - epsilon.
    for i in range(1, len(pwp.times) - 1):
        t = pwp.times[i]
        # Position continuity = the waypoint
        assert np.allclose(pwp.eval(t), path[i], atol=1e-9)
        # Velocity continuity: left limit ≈ right limit
        eps = 1e-7
        v_left = pwp.velocity(t - eps)
        v_right = pwp.velocity(t + eps)
        assert np.allclose(v_left, v_right, atol=1e-4)


def test_solve_multi_segment_waypoint_pass():
    """Every waypoint must be hit exactly at its segment's start time."""
    A = _at_origin()
    path = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
    ]
    pwp, info = solve(A, path, [], _solver_params())
    assert info.success is True
    for t_knot, wp in zip(pwp.times, path):
        assert np.allclose(pwp.eval(t_knot), wp, atol=1e-9)


# ---------------------------------------------------------------------------
# Time allocation
# ---------------------------------------------------------------------------

def test_solve_longer_segment_gets_more_time():
    A = _at_origin()
    path = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([6.0, 0.0, 0.0]),
    ]
    pwp, info = solve(A, path, [], _solver_params())
    assert info.success is True
    T0 = pwp.times[1] - pwp.times[0]
    T1 = pwp.times[2] - pwp.times[1]
    assert T1 > T0
    # And both should respect the floor (no zero-duration segments)
    assert T0 > 0.0 and T1 > 0.0


def test_solve_v_max_scales_total_duration():
    """Doubling v_max should roughly halve total time."""
    path = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
    ]
    A = _at_origin()
    _, info_slow = solve(A, path, [], _solver_params(v_max=1.0))
    _, info_fast = solve(A, path, [], _solver_params(v_max=4.0))
    # Re-solve to get the actual times
    pwp_slow, _ = solve(A, path, [], _solver_params(v_max=1.0))
    pwp_fast, _ = solve(A, path, [], _solver_params(v_max=4.0))
    Tslow = pwp_slow.times[-1] - pwp_slow.times[0]
    Tfast = pwp_fast.times[-1] - pwp_fast.times[0]
    assert Tfast < Tslow
    # 4x velocity should give roughly 4x faster, within the cubic-headroom
    # cruise factor — allow a generous tolerance.
    assert Tfast == pytest.approx(Tslow / 4.0, rel=0.5)


# ---------------------------------------------------------------------------
# Info fields
# ---------------------------------------------------------------------------

def test_solve_reports_nonnegative_cost_and_wall_time():
    A = _at_origin()
    path = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
    _, info = solve(A, path, [], _solver_params())
    assert info.cost >= 0.0
    assert info.wall_time_s >= 0.0


# ---------------------------------------------------------------------------
# Class form
# ---------------------------------------------------------------------------

def test_class_form_matches_functional_form():
    A = _at_origin()
    path = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]
    pwp_fn, info_fn = solve(A, path, [], _solver_params())
    pwp_cl, info_cl = MinJerkSolver(_solver_params()).solve(A, path, [])
    assert info_fn.success == info_cl.success is True
    assert len(pwp_fn.times) == len(pwp_cl.times)
    assert np.allclose(pwp_fn.coeff_x[0], pwp_cl.coeff_x[0])


# ---------------------------------------------------------------------------
# PieceWisePol shape — make sure we produce something the eval code can
# actually drive without IndexErrors
# ---------------------------------------------------------------------------

def test_eval_outside_horizon_returns_endpoint_position():
    A = _at_origin()
    path = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]
    pwp, _ = solve(A, path, [], _solver_params())
    assert isinstance(pwp, PieceWisePol)
    # Past the end -> last waypoint
    assert np.allclose(pwp.eval(pwp.times[-1] + 5.0), [5.0, 0.0, 0.0], atol=1e-9)
    # Before the start -> first waypoint
    assert np.allclose(pwp.eval(pwp.times[0] - 5.0), [0.0, 0.0, 0.0], atol=1e-9)
