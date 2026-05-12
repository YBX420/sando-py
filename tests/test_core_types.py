"""Tests for foundational types ported from sando_type.hpp."""

import math

import numpy as np
import pytest

from sando_py.core.basis_converter import BasisConverter
from sando_py.core.dyn_traj import DynTraj, TrajMode
from sando_py.core.piecewise_poly import PieceWisePol
from sando_py.core.types import Parameters, RobotState


# --- Parameters defaults (sando_type.hpp:34) -------------------------
def test_parameters_defaults():
    p = Parameters()
    assert p.factor_initial == 1.0
    assert p.factor_final == 5.0
    assert p.factor_constant_step_size == 0.1
    assert p.inflate_unknown_boundary is True
    assert p.using_variable_elimination is True
    assert p.hover_avoidance_d_trigger == 4.0
    assert p.dynamic_constraint_type == "Linf"


# --- RobotState (sando_type.hpp:988) ---------------------------------
def test_robotstate_setters():
    s = RobotState()
    s.setPos(1.0, 2.0, 3.0)
    s.setVel([4.0, 5.0, 6.0])
    s.setAccel(np.array([7.0, 8.0, 9.0]))
    assert s.pos.tolist() == [1.0, 2.0, 3.0]
    assert s.vel.tolist() == [4.0, 5.0, 6.0]
    assert s.accel.tolist() == [7.0, 8.0, 9.0]
    s.setZero()
    assert s.pos.tolist() == [0.0, 0.0, 0.0]
    assert s.yaw == 0.0


# --- PieceWisePol (sando_type.hpp:545) -------------------------------
def test_piecewise_eval():
    pwp = PieceWisePol()
    pwp.times = [0.0, 1.0, 2.0]
    pwp.coeff_x = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 1.0, 5.0])]
    pwp.coeff_y = [np.zeros(4), np.zeros(4)]
    pwp.coeff_z = [np.zeros(4), np.zeros(4)]

    # Interval 0 at t=0.5: 1*0.125 + 2*0.25 + 3*0.5 + 4 = 6.125
    assert pwp.eval(0.5)[0] == pytest.approx(6.125)
    # velocity at t=0.5: 3*0.25 + 4*0.5 + 3 = 5.75
    assert pwp.velocity(0.5)[0] == pytest.approx(5.75)
    # acceleration at t=0.5: 6*0.5 + 4 = 7.0
    assert pwp.acceleration(0.5)[0] == pytest.approx(7.0)

    # t before first knot -> uses first segment at u=0 -> constant term = 4.0
    assert pwp.eval(-1.0)[0] == pytest.approx(4.0)
    # t past last knot -> uses last segment at u = times[-1] - times[-2] = 1.0
    # last coeffs = [0,0,1,5] -> u + 5 = 6.0
    assert pwp.eval(5.0)[0] == pytest.approx(6.0)

    assert pwp.getDuration() == pytest.approx(2.0)
    assert pwp.getEndTime() == pytest.approx(2.0)


# --- BasisConverter (sando_type.hpp:210) -----------------------------
def test_basis_converter_shapes():
    bc = BasisConverter()
    assert bc.A_pos_mv_rest.shape == (4, 4)
    assert bc.A_vel_mv_rest.shape == (3, 3)
    assert bc.A_accel_mv_rest.shape == (2, 2)
    assert bc.A_pos_be_rest.shape == (4, 4)
    assert bc.A_pos_bs_rest.shape == (4, 4)


def test_basis_converter_inverses():
    bc = BasisConverter()
    assert np.allclose(bc.A_pos_mv_rest @ bc.A_pos_mv_rest_inv, np.eye(4), atol=1e-12)
    assert np.allclose(bc.A_vel_mv_rest @ bc.A_vel_mv_rest_inv, np.eye(3), atol=1e-12)
    assert np.allclose(bc.A_accel_mv_rest @ bc.A_accel_mv_rest_inv, np.eye(2), atol=1e-12)


def test_basis_converter_segment_assembly():
    bc = BasisConverter()
    n = 8
    assert len(bc.getABSpline(n)) == n
    assert len(bc.getAMinvo(n)) == n
    assert len(bc.getABezier(n)) == n
    assert len(bc.getMinvoPosConverters(n)) == n
    assert len(bc.getBezierPosConverters(n)) == n
    assert len(bc.getMinvoVelConverters(n)) == n - 1   # 1 seg0 + (n-3) rest + 1 last = n-1
    assert len(bc.getBezierVelConverters(n)) == n - 1


def test_bezier_to_minvo_pos():
    """getMinvoPosConverterFromBezier returns A_mv_inv * A_be."""
    bc = BasisConverter()
    M = bc.getMinvoPosConverterFromBezier()
    expected = bc.A_pos_mv_rest_inv @ bc.A_pos_be_rest
    assert np.allclose(M, expected)


# --- DynTraj (sando_type.hpp:724) ------------------------------------
def test_dyntraj_analytic():
    dt = DynTraj()
    dt.mode = TrajMode.Analytic
    dt.traj_x = "sin(t)"
    dt.traj_y = "cos(t)"
    dt.traj_z = "1.5"
    assert dt.compileAnalytic()
    p = dt.eval(0.5)
    assert p[0] == pytest.approx(math.sin(0.5))
    assert p[1] == pytest.approx(math.cos(0.5))
    assert p[2] == pytest.approx(1.5)


def test_dyntraj_piecewise():
    pwp = PieceWisePol()
    pwp.times = [0.0, 1.0]
    pwp.coeff_x = [np.array([0, 0, 1, 0])]   # x(u)=u
    pwp.coeff_y = [np.array([0, 0, 2, 0])]
    pwp.coeff_z = [np.array([0, 0, 0, 1])]
    dt = DynTraj()
    dt.setPiecewise(pwp)
    p = dt.eval(0.5)
    assert p.tolist() == [0.5, 1.0, 1.0]


def test_dyntraj_poly5_horner():
    # f(tau) = 1 + 2*tau + 3*tau^2  (c = [1,2,3,0,0,0])
    c = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    assert DynTraj.poly5_abs(c, 2.0) == pytest.approx(1 + 4 + 12)         # 17
    assert DynTraj.dpoly5_abs(c, 2.0) == pytest.approx(2 + 12)            # 14
    assert DynTraj.ddpoly5_abs(c, 2.0) == pytest.approx(6)                # 6
