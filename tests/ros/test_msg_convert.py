"""Round-trip tests for sando_py.sim_io.msg_convert.

Run from a ROS-sourced shell:

    scripts/with_ros_env.sh python -m pytest tests/ros/ -q

The whole suite is skipped if dynus_interfaces / geometry_msgs can't be
imported, so it stays harmless in conda-only environments.
"""

import math

import numpy as np
import pytest

# Skip the whole file in conda-only environments.
dim = pytest.importorskip("dynus_interfaces.msg")
gm = pytest.importorskip("geometry_msgs.msg")

from sando_py.core import Parameters, PieceWisePol, RobotState, TrajMode
from sando_py.sim_io import msg_convert as mc


def _params_with_drone_bbox():
    p = Parameters()
    p.drone_bbox = [0.2, 0.2, 0.2]
    p.z_min = 0.5
    p.z_max = 6.0
    p.force_goal_z = True
    p.default_goal_z = 2.0
    return p


# ---------------------------------------------------------------------------
# Vector3 / Quaternion helpers
# ---------------------------------------------------------------------------

def test_vec3_round_trip():
    v = gm.Vector3()
    v.x, v.y, v.z = 1.0, -2.5, 3.25
    a = mc.vec3_to_array(v)
    assert a.shape == (3,)
    assert a.tolist() == [1.0, -2.5, 3.25]

    v2 = mc.array_to_vec3([4.0, 5.0, 6.0])
    assert (v2.x, v2.y, v2.z) == (4.0, 5.0, 6.0)


def test_quat_to_yaw_identity_and_90deg():
    q_id = gm.Quaternion()
    q_id.x, q_id.y, q_id.z, q_id.w = 0.0, 0.0, 0.0, 1.0
    assert mc.quat_to_yaw(q_id) == pytest.approx(0.0)

    q_yaw90 = gm.Quaternion()
    q_yaw90.x, q_yaw90.y = 0.0, 0.0
    q_yaw90.z, q_yaw90.w = math.sin(math.pi / 4), math.cos(math.pi / 4)
    assert mc.quat_to_yaw(q_yaw90) == pytest.approx(math.pi / 2)


# ---------------------------------------------------------------------------
# State -> RobotState
# ---------------------------------------------------------------------------

def test_state_msg_to_robot_state():
    msg = dim.State()
    msg.pos.x, msg.pos.y, msg.pos.z = 1.0, 2.0, 3.0
    msg.vel.x, msg.vel.y, msg.vel.z = 0.5, -0.5, 0.0
    msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w = (
        0.0,
        0.0,
        math.sin(math.pi / 4),
        math.cos(math.pi / 4),
    )
    s = mc.state_msg_to_robot_state(msg, t_now=42.0)
    assert isinstance(s, RobotState)
    assert s.t == 42.0
    assert s.pos.tolist() == [1.0, 2.0, 3.0]
    assert s.vel.tolist() == [0.5, -0.5, 0.0]
    assert s.accel.tolist() == [0.0, 0.0, 0.0]
    assert s.yaw == pytest.approx(math.pi / 2)


# ---------------------------------------------------------------------------
# PoseStamped -> goal
# ---------------------------------------------------------------------------

def test_pose_stamped_to_goal_forces_z():
    p = _params_with_drone_bbox()
    msg = gm.PoseStamped()
    msg.pose.position.x = 10.0
    msg.pose.position.y = 5.0
    msg.pose.position.z = 0.0  # ignored because force_goal_z=True
    g = mc.pose_stamped_to_goal(msg, p)
    assert g.tolist() == [10.0, 5.0, 2.0]


def test_pose_stamped_to_goal_uses_msg_z():
    p = _params_with_drone_bbox()
    p.force_goal_z = False
    msg = gm.PoseStamped()
    msg.pose.position.x = 0.0
    msg.pose.position.y = 0.0
    msg.pose.position.z = 4.0
    g = mc.pose_stamped_to_goal(msg, p)
    assert g.tolist() == [0.0, 0.0, 4.0]


def test_pose_stamped_to_goal_rejects_out_of_bounds():
    p = _params_with_drone_bbox()
    p.force_goal_z = False
    msg = gm.PoseStamped()
    msg.pose.position.z = 20.0  # > z_max=6
    with pytest.raises(ValueError):
        mc.pose_stamped_to_goal(msg, p)


# ---------------------------------------------------------------------------
# RobotState -> Goal
# ---------------------------------------------------------------------------

def test_robot_state_to_goal_msg():
    s = RobotState()
    s.setPos(1.0, 2.0, 3.0)
    s.setVel(0.1, 0.2, 0.3)
    s.setAccel(-0.1, -0.2, -0.3)
    s.setJerk(0.0, 0.0, 1.0)
    s.setYaw(0.7)
    s.setDYaw(0.05)
    g = mc.robot_state_to_goal_msg(s, frame_id="map")
    assert g.header.frame_id == "map"
    assert (g.p.x, g.p.y, g.p.z) == (1.0, 2.0, 3.0)
    assert (g.v.x, g.v.y, g.v.z) == (0.1, 0.2, 0.3)
    assert (g.a.x, g.a.y, g.a.z) == (-0.1, -0.2, -0.3)
    assert (g.j.x, g.j.y, g.j.z) == (0.0, 0.0, 1.0)
    assert g.yaw == pytest.approx(0.7)
    assert g.dyaw == pytest.approx(0.05)
    assert g.power is True
    assert g.mode_xy == dim.Goal.MODE_POSITION_CONTROL
    assert g.mode_z == dim.Goal.MODE_POSITION_CONTROL


# ---------------------------------------------------------------------------
# PWPTraj <-> PieceWisePol
# ---------------------------------------------------------------------------

def _sample_pwp():
    pwp = PieceWisePol()
    pwp.times = [0.0, 1.0, 2.5]
    pwp.coeff_x = [np.array([1, 2, 3, 4], dtype=float),
                   np.array([5, 6, 7, 8], dtype=float)]
    pwp.coeff_y = [np.array([0, 0, 0, 1], dtype=float),
                   np.array([0, 0, 0, 2], dtype=float)]
    pwp.coeff_z = [np.array([0, 0, 0, 3], dtype=float),
                   np.array([0, 0, 0, 3], dtype=float)]
    return pwp


def test_pwp_round_trip():
    pwp_in = _sample_pwp()
    msg = mc.pwp_to_pwp_msg(pwp_in)

    assert list(msg.times) == [0.0, 1.0, 2.5]
    assert len(msg.coeff_x) == len(msg.coeff_y) == len(msg.coeff_z) == 2
    assert (msg.coeff_x[0].a, msg.coeff_x[0].b, msg.coeff_x[0].c, msg.coeff_x[0].d) == (1, 2, 3, 4)

    pwp_out = mc.pwp_msg_to_pwp(msg)
    assert pwp_out.times == pwp_in.times
    for c_in, c_out in zip(pwp_in.coeff_x, pwp_out.coeff_x):
        assert np.allclose(c_in, c_out)
    for c_in, c_out in zip(pwp_in.coeff_y, pwp_out.coeff_y):
        assert np.allclose(c_in, c_out)
    for c_in, c_out in zip(pwp_in.coeff_z, pwp_out.coeff_z):
        assert np.allclose(c_in, c_out)


def test_pwp_msg_length_mismatch_raises():
    msg = dim.PWPTraj()
    msg.times = [0.0, 1.0]
    msg.coeff_x = [dim.CoeffPoly3()]
    msg.coeff_y = [dim.CoeffPoly3(), dim.CoeffPoly3()]  # mismatch
    msg.coeff_z = [dim.CoeffPoly3()]
    with pytest.raises(ValueError):
        mc.pwp_msg_to_pwp(msg)


# ---------------------------------------------------------------------------
# DynTraj msg -> core.DynTraj
# ---------------------------------------------------------------------------

def _build_dyn_traj_msg(*, is_agent=False, with_analytic=True,
                       with_pwp=True, obs_id=7):
    msg = dim.DynTraj()
    msg.id = obs_id
    msg.bbox = [0.8, 0.8, 0.8]
    msg.is_agent = is_agent

    msg.pos.x, msg.pos.y, msg.pos.z = 1.0, 2.0, 1.5

    if with_pwp:
        # Two-segment trivial PWP at constant pos
        pwp = _sample_pwp()
        msg.pwp = mc.pwp_to_pwp_msg(pwp)

    if with_analytic:
        msg.function = ["sin(t)", "cos(t)", "1.5"]
        msg.velocity = ["cos(t)", "-sin(t)", "0.0"]

    if is_agent:
        msg.goal = [10.0, 0.0, 2.0]
        msg.header.stamp.sec = 100
        msg.header.stamp.nanosec = 500_000_000  # 100.5s

    return msg


def test_dyn_traj_msg_obstacle_pwp_only():
    p = _params_with_drone_bbox()
    msg = _build_dyn_traj_msg(is_agent=False, with_analytic=False, with_pwp=True)

    traj = mc.dyn_traj_msg_to_dyn_traj(msg, p, t_now=10.0)

    assert traj.id == 7
    assert traj.is_agent is False
    # bbox = obstacle_half + drone_half = 0.4 + 0.1 = 0.5
    assert np.allclose(traj.bbox, [0.5, 0.5, 0.5])
    assert traj.mode == TrajMode.Piecewise
    assert traj.time_received == 10.0
    assert np.allclose(traj.current_pos, [1.0, 2.0, 1.5])
    # No analytic compile attempted -> should not be set
    assert not traj.analytic_compiled


def test_dyn_traj_msg_obstacle_analytic_overrides():
    p = _params_with_drone_bbox()
    msg = _build_dyn_traj_msg(is_agent=False, with_analytic=True, with_pwp=True)

    traj = mc.dyn_traj_msg_to_dyn_traj(msg, p, t_now=0.0)

    # Analytic compile should succeed and override the mode
    assert traj.analytic_compiled
    assert traj.mode == TrajMode.Analytic
    # eval at t=0 -> (sin0, cos0, 1.5) = (0, 1, 1.5)
    pos0 = traj.eval(0.0)
    assert pos0[0] == pytest.approx(0.0)
    assert pos0[1] == pytest.approx(1.0)
    assert pos0[2] == pytest.approx(1.5)


def test_dyn_traj_msg_skip_future_uses_stationary():
    p = _params_with_drone_bbox()
    p.use_only_curr_pos_for_dynamic_obst = True
    msg = _build_dyn_traj_msg(is_agent=False, with_analytic=True, with_pwp=True)

    traj = mc.dyn_traj_msg_to_dyn_traj(msg, p, t_now=0.0)

    # Even with analytic strings in the msg, skip_future replaces them with
    # stationary traj at msg.pos before the analytic branch.
    # The stationary analytic is compiled first; then the analytic branch is
    # skipped because skip_future=True, so msg.function never overrides.
    assert traj.mode == TrajMode.Analytic
    pos5 = traj.eval(5.0)
    assert np.allclose(pos5, [1.0, 2.0, 1.5])  # stationary


def test_dyn_traj_msg_agent_path_sets_goal_and_delay():
    p = _params_with_drone_bbox()
    p.use_comm_delay_inflation = True
    msg = _build_dyn_traj_msg(is_agent=True)

    traj = mc.dyn_traj_msg_to_dyn_traj(msg, p, t_now=101.0)

    assert traj.is_agent is True
    assert np.allclose(traj.goal, [10.0, 0.0, 2.0])
    # delay = 101.0 - (100 + 0.5) = 0.5
    assert traj.communication_delay == pytest.approx(0.5)


def test_dyn_traj_msg_negative_delay_clamped():
    p = _params_with_drone_bbox()
    p.use_comm_delay_inflation = True
    msg = _build_dyn_traj_msg(is_agent=True)

    # t_now < header.stamp -> negative delay should be clamped to 0
    traj = mc.dyn_traj_msg_to_dyn_traj(msg, p, t_now=50.0)
    assert traj.communication_delay == 0.0


# ---------------------------------------------------------------------------
# DynTrajArray
# ---------------------------------------------------------------------------

def test_dyn_traj_array_msg_to_list():
    p = _params_with_drone_bbox()
    arr = dim.DynTrajArray()
    for i in range(3):
        arr.trajs.append(
            _build_dyn_traj_msg(is_agent=False, with_analytic=False,
                               with_pwp=True, obs_id=i)
        )
    trajs = mc.dyn_traj_array_msg_to_dyn_trajs(arr, p, t_now=0.0)
    assert len(trajs) == 3
    assert [t.id for t in trajs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# PointCloud2 -> numpy
# ---------------------------------------------------------------------------

def test_cloud_msg_to_points_roundtrip():
    sm = pytest.importorskip("sensor_msgs.msg")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    header = pytest.importorskip("std_msgs.msg").Header()

    pts = np.array([
        [1.0, 2.0, 3.0],
        [-0.5, 4.0, 0.25],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    msg = pcl2_py.create_cloud_xyz32(header, pts)
    out = mc.cloud_msg_to_points(msg)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    assert np.allclose(out, pts)


def test_cloud_msg_to_points_empty():
    sm = pytest.importorskip("sensor_msgs.msg")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    header = pytest.importorskip("std_msgs.msg").Header()
    msg = pcl2_py.create_cloud_xyz32(header, np.zeros((0, 3), dtype=np.float32))
    out = mc.cloud_msg_to_points(msg)
    assert out.shape == (0, 3)


def test_cloud_msg_to_points_skips_nans():
    sm = pytest.importorskip("sensor_msgs.msg")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    header = pytest.importorskip("std_msgs.msg").Header()

    pts = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, np.nan, np.nan],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    msg = pcl2_py.create_cloud_xyz32(header, pts)
    out = mc.cloud_msg_to_points(msg)
    assert out.shape == (2, 3)
    assert np.allclose(out, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# ---------------------------------------------------------------------------
# transform_points_to_frame
# ---------------------------------------------------------------------------

def _seed_tf(buffer, parent: str, child: str, translation, quat=(0.0, 0.0, 0.0, 1.0)):
    """Inject a single transform into a tf2 ``Buffer`` for testing.

    ``parent`` is the frame the data is being expressed in BEFORE the
    transform (= ``header.frame_id``), ``child`` is the new frame
    (the one we want to express the data in). Conventionally tf2
    stores ``parent -> child`` so reads ``child` data in `parent``
    coordinates. We follow the same convention.
    """
    builtin = pytest.importorskip("builtin_interfaces.msg")
    gm_msg = pytest.importorskip("geometry_msgs.msg")
    t = gm_msg.TransformStamped()
    t.header.stamp = builtin.Time(sec=0, nanosec=0)
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])
    t.transform.rotation.x = float(quat[0])
    t.transform.rotation.y = float(quat[1])
    t.transform.rotation.z = float(quat[2])
    t.transform.rotation.w = float(quat[3])
    buffer.set_transform(t, "test_authority")


def test_transform_points_noop_when_frames_match():
    pytest.importorskip("tf2_ros")
    from tf2_ros import Buffer
    buf = Buffer()
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    header = pytest.importorskip("std_msgs.msg").Header()
    out = mc.transform_points_to_frame(pts, "map", header.stamp, buf, "map")
    assert out is pts  # exact same object


def test_transform_points_noop_when_source_frame_empty():
    pytest.importorskip("tf2_ros")
    from tf2_ros import Buffer
    buf = Buffer()
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    header = pytest.importorskip("std_msgs.msg").Header()
    out = mc.transform_points_to_frame(pts, "", header.stamp, buf, "map")
    assert out is pts


def test_transform_points_returns_none_on_missing_tf():
    pytest.importorskip("tf2_ros")
    from tf2_ros import Buffer
    buf = Buffer()
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    header = pytest.importorskip("std_msgs.msg").Header()
    out = mc.transform_points_to_frame(
        pts, "sensor", header.stamp, buf, "map", timeout_s=0.0,
    )
    assert out is None


def test_transform_points_translation_only():
    """A pure-translation transform from ``sensor`` to ``map`` should
    add the translation to every point."""
    pytest.importorskip("tf2_ros")
    builtin = pytest.importorskip("builtin_interfaces.msg")
    from tf2_ros import Buffer

    buf = Buffer()
    # map -> sensor at (10, 0, 0), identity rotation. lookup_transform
    # (map, sensor) returns the pose of sensor in map.
    _seed_tf(buf, parent="map", child="sensor", translation=(10.0, 0.0, 0.0))

    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = mc.transform_points_to_frame(
        pts, "sensor", builtin.Time(sec=0, nanosec=0), buf, "map",
        timeout_s=0.0,
    )
    assert out is not None
    assert np.allclose(out, [[11.0, 2.0, 3.0]])


def test_transform_points_rotation_only():
    """A 90 deg yaw transform from ``sensor`` to ``map`` should rotate
    each point accordingly."""
    pytest.importorskip("tf2_ros")
    builtin = pytest.importorskip("builtin_interfaces.msg")
    from tf2_ros import Buffer

    buf = Buffer()
    # Yaw +90 deg: x_sensor -> y_map
    # quat for yaw=pi/2: (0, 0, sin(pi/4), cos(pi/4))
    sqrt2 = math.sqrt(2.0) / 2.0
    _seed_tf(buf, parent="map", child="sensor",
             translation=(0.0, 0.0, 0.0), quat=(0.0, 0.0, sqrt2, sqrt2))

    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    out = mc.transform_points_to_frame(
        pts, "sensor", builtin.Time(sec=0, nanosec=0), buf, "map",
        timeout_s=0.0,
    )
    assert out is not None
    assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)
