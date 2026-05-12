"""Tests for sando_py.core.utils — algorithmic ports from sando/utils.cpp."""

import math

import numpy as np
import pytest

from sando_py.core.types import RobotState
from sando_py.core.utils import (
    angle_wrap,
    createMoreVertexes,
    euclideanDistance,
    findVelocitiesInPath,
    getIntersectionWithPlane,
    getMinTimeDoubleIntegrator1D,
    getMinTimeDoubleIntegrator3D,
    getTravelTimes,
    projectPointToBox,
    projectPointToSphere,
    quaternion2Euler,
    saturate,
    sgn,
    transformStampedToMatrix,
)


def test_sgn():
    assert sgn(5.0) == 1
    assert sgn(-3.0) == -1
    assert sgn(0.0) == 0


def test_saturate():
    assert saturate(5.0, 0.0, 10.0) == 5.0
    assert saturate(-1.0, 0.0, 10.0) == 0.0
    assert saturate(15.0, 0.0, 10.0) == 10.0


def test_angle_wrap():
    assert angle_wrap(0.0) == pytest.approx(0.0)
    assert angle_wrap(math.pi) == pytest.approx(-math.pi)  # boundary -> -pi after wrap
    assert angle_wrap(3 * math.pi) == pytest.approx(-math.pi)
    assert angle_wrap(-3 * math.pi) == pytest.approx(-math.pi)
    assert angle_wrap(math.pi / 2) == pytest.approx(math.pi / 2)
    assert angle_wrap(-math.pi / 2) == pytest.approx(-math.pi / 2)


def test_euclidean_distance():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert euclideanDistance(p1, p2) == pytest.approx(5.0)


def test_intersection_with_plane():
    # Plane z=0 (coeffs [0,0,1,0])
    p1 = np.array([0.0, 0.0, -1.0])
    p2 = np.array([0.0, 0.0, 1.0])
    hit, inter = getIntersectionWithPlane(p1, p2, np.array([0.0, 0.0, 1.0, 0.0]))
    assert hit
    assert np.allclose(inter, [0.0, 0.0, 0.0])

    # No intersection within segment
    p1b = np.array([0.0, 0.0, 1.0])
    p2b = np.array([0.0, 0.0, 2.0])
    hit, _ = getIntersectionWithPlane(p1b, p2b, np.array([0.0, 0.0, 1.0, 0.0]))
    assert not hit


def test_project_point_to_box_inside():
    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([0.5, 0.5, 0.5])
    out = projectPointToBox(P1, P2, wdx=4.0, wdy=4.0, wdz=4.0)
    assert np.allclose(out, P2)


def test_project_point_to_box_outside_then_inside():
    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([10.0, 0.0, 0.0])
    out = projectPointToBox(P1, P2, wdx=4.0, wdy=4.0, wdz=4.0)
    # intersection with x=2 plane, then pulled back by 1.0 toward P1 -> at x=1
    assert out[0] == pytest.approx(1.0)
    assert abs(out[1]) < 1e-9
    assert abs(out[2]) < 1e-9


def test_project_point_to_sphere():
    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([10.0, 0.0, 0.0])
    out = projectPointToSphere(P1, P2, radius=3.0)
    assert np.allclose(out, [3.0, 0.0, 0.0])

    # already inside
    P3 = np.array([1.0, 0.0, 0.0])
    out = projectPointToSphere(P1, P3, radius=3.0)
    assert np.allclose(out, P3)


def test_quaternion_to_euler_identity():
    # Identity quaternion -> all zero
    roll, pitch, yaw = quaternion2Euler([0.0, 0.0, 0.0, 1.0])
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)


def test_quaternion_to_euler_yaw_90():
    # 90 deg yaw about z
    qx, qy, qz, qw = 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)
    _, _, yaw = quaternion2Euler([qx, qy, qz, qw])
    assert yaw == pytest.approx(math.pi / 2)


def test_transform_stamped_to_matrix():
    T = transformStampedToMatrix([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert T[3, 3] == 1.0


def test_create_more_vertexes():
    path = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
    createMoreVertexes(path, d=2.0)
    # Matches C++ behavior: floor(10/2)=5 insertions, the 5th lands ON the endpoint
    # so we get [P0, P2, P4, P6, P8, P10, P10] — 7 points with a duplicated endpoint.
    assert len(path) == 7
    assert path[0].tolist() == [0.0, 0.0, 0.0]
    assert np.allclose(path[1], [2.0, 0.0, 0.0])
    assert np.allclose(path[-1], [10.0, 0.0, 0.0])
    assert np.allclose(path[-2], [10.0, 0.0, 0.0])


def test_min_time_double_integrator_1d_static_start_end():
    # Symmetric maneuver: 0->10, zero start/end velocity, v_max=2, a_max=1
    t = getMinTimeDoubleIntegrator1D(0.0, 0.0, 10.0, 0.0, 2.0, 1.0)
    # Bang-bang with cruise: accel 0->2 in 2s (2m), decel in 2s (2m), cruise 6m at 2m/s = 3s -> 7s
    assert t == pytest.approx(7.0, abs=1e-6)


def test_min_time_double_integrator_3d_returns_max_axis():
    p0 = np.zeros(3); pf = np.array([10.0, 1.0, 0.0])
    v0 = np.zeros(3); vf = np.zeros(3)
    v_max = np.array([2.0, 2.0, 2.0])
    a_max = np.array([1.0, 1.0, 1.0])
    t = getMinTimeDoubleIntegrator3D(p0, v0, pf, vf, v_max, a_max)
    # x-axis is the bottleneck (10m)
    assert t == pytest.approx(7.0, abs=1e-6)


def test_find_velocities_monotonic_path():
    path = [np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0])]
    A = RobotState()
    A.vel = np.zeros(3)
    v_max = np.array([3.0, 3.0, 3.0])
    vs = findVelocitiesInPath(path, A, v_max, verbose=False)
    assert len(vs) == len(path)
    assert np.allclose(vs[0], [0.0, 0.0, 0.0])
    assert np.allclose(vs[-1], [0.0, 0.0, 0.0])
    # middle velocities should be positive on x
    assert vs[1][0] > 0
    assert vs[2][0] > 0


def test_get_travel_times():
    A = RobotState()
    A.vel = np.zeros(3)
    path = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]
    times = getTravelTimes(path, A, v_max_3d=np.array([2.0, 2.0, 2.0]),
                           a_max_3d=np.array([1.0, 1.0, 1.0]))
    assert len(times) == 1
    assert times[0] > 0
