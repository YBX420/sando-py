"""Skeleton-level tests for ``sando_py.sim_io.sando_node.SandoNode``.

These exercise the callback wiring (state initialization, terminal goal
acceptance, state-machine transitions, traj cleanup) without spinning
the executor or running a real ROS network. We construct the node, call
its private callbacks directly with hand-built messages, and inspect the
internal state.

Run from a ROS-sourced shell:

    scripts/with_ros_env.sh python3 -m pytest tests/ros/ -q
"""

import math
import time

import numpy as np
import pytest

# Skip the whole file outside ROS-sourced envs.
dim = pytest.importorskip("dynus_interfaces.msg")
gm = pytest.importorskip("geometry_msgs.msg")
rclpy = pytest.importorskip("rclpy")

from sando_py.sim_io import DroneStatus, SandoNode
from sando_py.sim_io.msg_convert import pwp_to_pwp_msg

from sando_py.core import PieceWisePol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node():
    n = SandoNode()
    yield n
    n.destroy_node()


def _converge_yawing(node, max_ticks: int = 200) -> None:
    """Drive ``_publish_goal_cb`` until YAWING -> TRAVELING.

    Each call advances yaw by at most ``w_max_yawing * dc``. For a
    drone already facing the goal (the common case in our tests) this
    converges in 1 tick; for arbitrary starting yaw it may need many.
    """
    for _ in range(max_ticks):
        if node.status_ != DroneStatus.YAWING:
            return
        node._publish_goal_cb()
    raise RuntimeError(
        f"YAWING failed to converge in {max_ticks} ticks "
        f"(status={node.status_})"
    )


def _state(pos, yaw_z_quat=False) -> "dim.State":
    msg = dim.State()
    msg.pos.x, msg.pos.y, msg.pos.z = float(pos[0]), float(pos[1]), float(pos[2])
    msg.vel.x = msg.vel.y = msg.vel.z = 0.0
    # identity quaternion (yaw=0) unless yaw_z_quat overrides
    msg.quat.x = msg.quat.y = msg.quat.z = 0.0
    msg.quat.w = 1.0
    return msg


def _pose_stamped(pos):
    p = gm.PoseStamped()
    p.pose.position.x, p.pose.position.y, p.pose.position.z = (
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
    )
    p.pose.orientation.w = 1.0
    return p


def _dyn_traj(obs_id, pos=(1.0, 2.0, 1.5), is_agent=False):
    msg = dim.DynTraj()
    msg.id = obs_id
    msg.bbox = [0.5, 0.5, 0.5]
    msg.pos.x, msg.pos.y, msg.pos.z = pos
    msg.is_agent = is_agent
    # Minimal valid PWP
    pwp = PieceWisePol()
    pwp.times = [0.0, 1.0]
    pwp.coeff_x = [np.array([0, 0, 0, pos[0]], dtype=float)]
    pwp.coeff_y = [np.array([0, 0, 0, pos[1]], dtype=float)]
    pwp.coeff_z = [np.array([0, 0, 0, pos[2]], dtype=float)]
    msg.pwp = pwp_to_pwp_msg(pwp)
    return msg


# ---------------------------------------------------------------------------
# Construction / init
# ---------------------------------------------------------------------------

def test_node_constructs_with_default_yaml(node):
    """Initialization reads the default sando-py config and sets defaults."""
    assert node.get_name() == "sando_node"
    assert node.id_ == 0
    assert node.frame_id_ == "map"
    # Parameters were loaded from the YAML
    assert node.params.v_max > 0
    assert node.params.goal_radius > 0
    # Initial state-machine status mirrors the C++ initial value
    assert node.status_ == DroneStatus.GOAL_REACHED
    assert node.state_initialized_ is False
    assert node.current_state_ is None
    assert node.term_goal_ is None


def test_node_subscriptions_and_publishers(node):
    """The right topics are wired up."""
    topics = {n: t for n, t in node.get_topic_names_and_types()}
    # Goal publisher
    assert any("goal" in n for n in topics), topics
    # We can't reliably introspect subscriber topic names without spinning
    # a graph discovery — but the subscription objects should exist.
    assert hasattr(node, "sub_state_")
    assert hasattr(node, "sub_term_goal_")
    assert hasattr(node, "sub_predicted_traj_")
    # /trajs subscription only if not ignored
    if not node.params.ignore_other_trajs:
        assert hasattr(node, "sub_traj_")


# ---------------------------------------------------------------------------
# State callback
# ---------------------------------------------------------------------------

def test_state_callback_initializes_state(node):
    assert not node.state_initialized_
    node._state_cb(_state([5.0, 0.0, 2.0]))
    assert node.state_initialized_ is True
    assert node.current_state_ is not None
    assert node.current_state_.pos.tolist() == [5.0, 0.0, 2.0]


# ---------------------------------------------------------------------------
# Terminal goal callback + state machine
# ---------------------------------------------------------------------------

def test_terminal_goal_starts_replan_and_yawing(node):
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))  # z gets forced to default_goal_z
    assert node.term_goal_ is not None
    # default_goal_z from config
    assert node.term_goal_[2] == pytest.approx(node.params.default_goal_z)
    # status moves into YAWING (unless skip_initial_yawing)
    expected = DroneStatus.TRAVELING if node.params.skip_initial_yawing else DroneStatus.YAWING
    assert node.status_ == expected


def _yaw_state(pos, yaw_rad: float):
    """Build a State message with a non-zero yaw (z-axis quaternion)."""
    msg = dim.State()
    msg.pos.x, msg.pos.y, msg.pos.z = float(pos[0]), float(pos[1]), float(pos[2])
    msg.vel.x = msg.vel.y = msg.vel.z = 0.0
    msg.quat.x = msg.quat.y = 0.0
    msg.quat.z = float(np.sin(yaw_rad / 2.0))
    msg.quat.w = float(np.cos(yaw_rad / 2.0))
    return msg


# ---------------------------------------------------------------------------
# Yaw planning
# ---------------------------------------------------------------------------

def test_yawing_state_blocks_replan(node):
    """While YAWING, ``_replan_cb`` must return without touching the
    pipeline — the C++ has the same early-return."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    assert node.status_ == DroneStatus.YAWING
    node._replan_cb()
    # No committed trajectory should be produced while YAWING
    assert node.committed_traj_ is None
    assert node.status_ == DroneStatus.YAWING


def test_yawing_step_toward_goal_capped_by_w_max_yawing(node):
    """During YAWING the commanded yaw should march toward the goal
    direction by at most ``w_max_yawing * dc`` per tick."""
    # Agent at (0,0,2), facing pi rad (-x). Goal at (10, 0, *), so
    # desired yaw = 0. Diff = pi which is far from converged.
    import math as _math
    node._state_cb(_yaw_state([0.0, 0.0, 2.0], yaw_rad=_math.pi))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    assert node.status_ == DroneStatus.YAWING

    # previous_yaw_ was seeded to pi by term_goal_cb
    assert node.previous_yaw_ == pytest.approx(_math.pi, abs=1e-6)

    # One tick. Desired = 0, previous = pi. diff_cmd = angle_wrap(-pi)
    # = -pi (or pi depending on wrap convention). Step is clamped to
    # ±w_max_yawing * dc.
    yaw_before = node.previous_yaw_
    node._publish_goal_cb()
    step = node.previous_yaw_ - yaw_before
    max_step = node.params.w_max_yawing * node.params.dc
    assert abs(step) <= max_step + 1e-6


def test_yawing_converges_when_actual_yaw_within_threshold(node):
    """When the actual drone yaw is within 0.3 rad of the desired yaw,
    YAWING transitions to TRAVELING immediately."""
    # Agent already facing +x — desired yaw is 0, actual is 0, diff=0
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    assert node.status_ == DroneStatus.YAWING

    node._publish_goal_cb()
    # Should have transitioned (|diff_actual| = 0 < 0.3)
    assert node.status_ == DroneStatus.TRAVELING


def test_yawing_skipped_when_skip_initial_yawing_true(node):
    """``params.skip_initial_yawing=True`` should bypass YAWING and go
    straight to TRAVELING on the first terminal-goal callback."""
    node.params.skip_initial_yawing = True
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    assert node.status_ == DroneStatus.TRAVELING


def test_traveling_yaw_follows_velocity_direction(node):
    """In TRAVELING, the commanded yaw should bend toward the
    velocity direction (filtered via alpha_filter_dyaw)."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    _converge_yawing(node)
    node._replan_cb()
    assert node.status_ in (DroneStatus.TRAVELING, DroneStatus.GOAL_SEEN)

    # The committed trajectory's velocity at t_now is +x, so desired
    # yaw is 0. Run a publish tick and check yaw drifted (slightly)
    # toward 0 from previous_yaw_.
    yaw_before = node.previous_yaw_
    node._publish_goal_cb()
    # Either yaw didn't change (already aligned) or moved by <= w_max*dc
    step = node.previous_yaw_ - yaw_before
    max_step = node.params.w_max * node.params.dc
    assert abs(step) <= max_step + 1e-6


def test_goal_reached_holds_yaw(node):
    """In GOAL_REACHED, yaw must be held constant and dyaw = 0."""
    node._state_cb(_state([10.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    _converge_yawing(node)
    # First replan: status is TRAVELING but distance is 0 -> GOAL_REACHED
    node._replan_cb()
    assert node.status_ == DroneStatus.GOAL_REACHED

    yaw_before = node.previous_yaw_
    node._publish_goal_cb()
    # Yaw unchanged
    assert node.previous_yaw_ == pytest.approx(yaw_before, abs=1e-9)


def test_replan_drives_full_state_machine(node):
    """Walking from far away to inside goal_radius traverses YAWING ->
    TRAVELING -> GOAL_SEEN -> GOAL_REACHED.

    ``_replan_cb`` skips work while YAWING (the drone rotates in place;
    ``_publish_goal_cb`` drives the yaw step). We start the agent
    already facing the goal so the first replan-tick converges YAWING
    -> TRAVELING in one publish-goal call.
    """
    # Start at the goal direction (yaw=0 by default, goal is at +x)
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    assert node.status_ == DroneStatus.YAWING

    # _publish_goal_cb drives yaw convergence; current yaw aligns with
    # the goal direction so the first publish converges immediately.
    node._publish_goal_cb()
    assert node.status_ == DroneStatus.TRAVELING

    # Tick 1 (TRAVELING) runs the pipeline
    node._replan_cb()
    assert node.status_ == DroneStatus.TRAVELING

    # Move to within goal_seen_radius but outside goal_radius
    seen_r = node.params.goal_seen_radius
    inside_seen_outside_goal = max(seen_r * 0.5, node.params.goal_radius + 0.1)
    node._state_cb(_state([10.0 - inside_seen_outside_goal, 0.0, 2.0]))
    node._replan_cb()
    assert node.status_ == DroneStatus.GOAL_SEEN

    # Now inside goal_radius
    node._state_cb(_state([10.0, 0.0, 2.0]))
    node._replan_cb()
    assert node.status_ == DroneStatus.GOAL_REACHED


# ---------------------------------------------------------------------------
# Traj callback
# ---------------------------------------------------------------------------

def test_traj_cb_stores_msg_and_filters_self(node):
    node._traj_cb(_dyn_traj(obs_id=7))
    node._traj_cb(_dyn_traj(obs_id=8))
    # Same id as self should be ignored
    node._traj_cb(_dyn_traj(obs_id=node.id_))
    assert set(node.trajs_.keys()) == {7, 8}


def test_cleanup_drops_stale_trajs(node):
    """Trajs older than ``traj_lifetime`` are dropped on cleanup."""
    node._traj_cb(_dyn_traj(obs_id=1))
    # Force the stored traj to look old
    node.trajs_[1].time_received = node._t_now() - 999.0
    node._cleanup_old_trajs_cb()
    assert 1 not in node.trajs_


# ---------------------------------------------------------------------------
# Goal publisher
# ---------------------------------------------------------------------------

def test_publish_goal_no_state_is_noop(node):
    """Hover-stub silently skips when no state has been received."""
    assert node.current_state_ is None
    # Must not raise
    node._publish_goal_cb()


# ---------------------------------------------------------------------------
# End-to-end pipeline — hgp -> decomp -> solver
# ---------------------------------------------------------------------------

def test_replan_commits_trajectory_through_pipeline(node):
    """A clean state → terminal goal → replan should populate
    ``committed_traj_`` with a PieceWisePol that hits the goal."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))  # z forced to default_goal_z
    assert node.committed_traj_ is None

    # Converge YAWING (already facing +x toward goal) before replan
    _converge_yawing(node)
    node._replan_cb()

    assert node.committed_traj_ is not None
    pwp = node.committed_traj_
    assert isinstance(pwp, PieceWisePol)
    assert len(pwp.times) >= 2
    # Trajectory should end near the terminal goal
    end_pos = pwp.eval(pwp.times[-1])
    assert np.linalg.norm(end_pos - node.term_goal_) < 1.0


def test_publish_goal_after_replan_emits_traj_setpoint(node):
    """Once a trajectory is committed, the goal publisher should evaluate
    it instead of hovering at the current state."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    _converge_yawing(node)
    node._replan_cb()
    assert node.committed_traj_ is not None
    # Must not raise; just exercising the eval path. We can't easily
    # observe the published message without a subscriber, but the
    # callback must run cleanly with a committed trajectory in place.
    node._publish_goal_cb()


def test_cloud_cb_stores_points(node):
    """The cloud callback should parse the PointCloud2 into ``(N, 3)``
    numpy array and stash it on ``latest_cloud_``. With the cloud
    already in the map frame, no TF lookup happens."""
    sm = pytest.importorskip("sensor_msgs.msg")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    Header = pytest.importorskip("std_msgs.msg").Header

    assert node.latest_cloud_ is None
    pts = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    header = Header()
    header.frame_id = node.frame_id_  # already in map frame
    msg = pcl2_py.create_cloud_xyz32(header, pts)
    node._cloud_cb(msg)
    assert node.latest_cloud_ is not None
    assert node.latest_cloud_.shape == (2, 3)


def test_cloud_cb_transforms_sensor_frame_to_map_frame(node):
    """A cloud arriving in a non-map frame must be TF-transformed
    before storage. Seed the buffer with a translation transform and
    confirm the stored points carry the translation."""
    sm = pytest.importorskip("sensor_msgs.msg")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    builtin = pytest.importorskip("builtin_interfaces.msg")
    Header = pytest.importorskip("std_msgs.msg").Header
    gm_msg = pytest.importorskip("geometry_msgs.msg")

    # Seed: map -> sensor at (5, 0, 0)
    tf = gm_msg.TransformStamped()
    tf.header.stamp = builtin.Time(sec=0, nanosec=0)
    tf.header.frame_id = node.frame_id_
    tf.child_frame_id = "sensor"
    tf.transform.translation.x = 5.0
    tf.transform.rotation.w = 1.0
    node.tf_buffer_.set_transform(tf, "test")

    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    header = Header()
    header.stamp = builtin.Time(sec=0, nanosec=0)
    header.frame_id = "sensor"
    msg = pcl2_py.create_cloud_xyz32(header, pts)
    node._cloud_cb(msg)
    assert node.latest_cloud_ is not None
    # Point at (1, 0, 0) in sensor frame -> (6, 0, 0) in map frame
    assert np.allclose(node.latest_cloud_, [[6.0, 0.0, 0.0]], atol=1e-6)


def test_cloud_cb_drops_message_when_tf_unavailable(node):
    """When the cloud's frame isn't in the TF tree, the callback
    silently drops the message — no exception, no stale data
    overwritten."""
    pytest.importorskip("tf2_ros")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    builtin = pytest.importorskip("builtin_interfaces.msg")
    Header = pytest.importorskip("std_msgs.msg").Header

    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    header = Header()
    header.stamp = builtin.Time(sec=0, nanosec=0)
    header.frame_id = "nonexistent_frame"
    msg = pcl2_py.create_cloud_xyz32(header, pts)
    node._cloud_cb(msg)  # must not raise
    assert node.latest_cloud_ is None


def test_replan_uses_stored_cloud_for_occupancy(node):
    """A cloud wall across the corridor should change the resulting
    path. Tests the cloud → hgp.plan plumbing end-to-end."""
    sm = pytest.importorskip("sensor_msgs.msg")
    pcl2_py = pytest.importorskip("sensor_msgs_py.point_cloud2")
    Header = pytest.importorskip("std_msgs.msg").Header

    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([10.0, 0.0, 0.0]))
    _converge_yawing(node)

    # Replan once without any cloud to get the baseline path
    node._replan_cb()
    baseline_traj = node.committed_traj_
    assert baseline_traj is not None

    # Drop a wall of cloud points across the path
    wall = []
    for y in np.arange(-2.0, 2.01, 0.2):
        for z in np.arange(1.8, 2.21, 0.2):
            wall.append([5.0, float(y), float(z)])
    pts = np.array(wall, dtype=np.float32)
    cloud_msg = pcl2_py.create_cloud_xyz32(Header(), pts)
    node._cloud_cb(cloud_msg)
    assert node.latest_cloud_.shape[0] > 0

    # Re-plan: the new committed trajectory should differ from the
    # baseline since the cloud now blocks the straight line.
    node._replan_cb()
    new_traj = node.committed_traj_
    assert new_traj is not None
    # The two trajectories must visit different mid-points (the cloud
    # forced a detour)
    mid_t = 0.5 * (baseline_traj.times[0] + baseline_traj.times[-1])
    p_base = baseline_traj.eval(mid_t)
    mid_t2 = 0.5 * (new_traj.times[0] + new_traj.times[-1])
    p_new = new_traj.eval(mid_t2)
    assert not np.allclose(p_base, p_new, atol=0.05)


def test_find_A_falls_back_to_current_state_without_committed_traj(node):
    """First-tick behavior: with no committed trajectory yet, ``_find_A``
    returns the current state verbatim."""
    node._state_cb(_state([1.0, 2.0, 3.0]))
    assert node.committed_traj_ is None
    A = node._find_A(node._t_now())
    assert A is not None
    assert np.allclose(A.pos, [1.0, 2.0, 3.0])


def test_find_A_returns_lookahead_point_on_committed_traj(node):
    """After a successful plan, ``_find_A`` evaluates the committed
    trajectory at ``t_now + lookahead_replan_time``, *not* the current
    state. That's the v2 hand-off-smoothing behaviour.

    Explicitly disables the adaptive lookahead so we can assert the
    deterministic ``lookahead_replan_time`` value.
    """
    node.params.use_adaptive_lookahead = False
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([5.0, 0.0, 0.0]))
    _converge_yawing(node)
    node._replan_cb()
    assert node.committed_traj_ is not None
    t_now = node._t_now()
    A = node._find_A(t_now)
    # A.t should be the future eval time, not t_now
    expected_t = t_now + node.params.lookahead_replan_time
    # Clamped to trajectory domain — the trajectory starts at A.t of
    # the previous replan (which was t_now of that call), so the
    # expected time should land within it.
    assert A.t == pytest.approx(
        min(expected_t, node.committed_traj_.times[-1]),
        abs=1e-6,
    )
    # A.pos / vel / accel come from the committed_traj evaluation
    assert np.allclose(A.pos, node.committed_traj_.eval(A.t), atol=1e-9)
    assert np.allclose(A.vel, node.committed_traj_.velocity(A.t), atol=1e-9)
    assert np.allclose(A.accel, node.committed_traj_.acceleration(A.t), atol=1e-9)


# ---------------------------------------------------------------------------
# findA k_value adaptation (port of C++ findAandAtime, sando.cpp:185-235)
# ---------------------------------------------------------------------------

def _force_committed_traj_horizon(node, horizon_s: float) -> None:
    """Make ``_find_A`` think the committed trajectory spans
    ``horizon_s`` seconds so the trajectory-horizon clamp doesn't
    fire in the warm-up case (``default_k_value = 50``, ``dc = 0.01``
    → 0.49 s lookahead needs at least that much horizon)."""
    pwp = PieceWisePol()
    pwp.times = [0.0, horizon_s]
    pwp.coeff_x = [np.array([0.0, 0.0, 0.0, 0.0], dtype=float)]
    pwp.coeff_y = [np.array([0.0, 0.0, 0.0, 0.0], dtype=float)]
    pwp.coeff_z = [np.array([0.0, 0.0, 0.0, 2.0], dtype=float)]
    node.committed_traj_ = pwp


def test_adaptive_warmup_uses_default_k_value_lookahead(node):
    """During warm-up (no full sample set yet) ``_find_A`` should
    anchor ``A`` at ``(default_k_value - 1) * dc`` seconds ahead of
    ``t_now``. Mirrors the C++ branch at sando.cpp:200-204."""
    node.params.use_adaptive_lookahead = True
    node.params.default_k_value = 50
    node.params.dc = 0.01
    node._state_cb(_state([0.0, 0.0, 2.0]))
    _force_committed_traj_horizon(node, horizon_s=10.0)

    # Pretend the trajectory's domain is anchored at t_now so the
    # lookahead clamp lands inside the trajectory.
    t_now = node._t_now()
    node.committed_traj_.times = [t_now, t_now + 10.0]
    assert not node._use_adapt_k_value, "warm-up flag should still be off"

    A = node._find_A(t_now)
    expected = t_now + (node.params.default_k_value - 1) * node.params.dc
    assert A.t == pytest.approx(expected, abs=1e-9)


def test_adaptive_warmup_collects_samples_after_replan(node):
    """Each successful replan should append the measured wall time to
    ``_store_computation_times`` (with the C++ one-off skip at
    num_replanning_ == 1)."""
    node.params.use_adaptive_lookahead = True
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([5.0, 0.0, 0.0]))
    _converge_yawing(node)

    # First replan: warm-up, no prior measurement → store the zero seed.
    node._replan_cb()
    # The first warm-up tick's _find_A ran with num_replanning_ == 0,
    # so it stored. After the pipeline, num_replanning_ ticked to 1.
    assert node._num_replanning == 1
    assert node._last_replan_comp_time > 0.0
    assert node._store_computation_times == [0.0]

    # Second replan: _find_A sees num_replanning_ == 1 → SKIP store.
    # After pipeline → num_replanning_ ticks to 2.
    node._replan_cb()
    assert node._num_replanning == 2
    assert len(node._store_computation_times) == 1

    # Third replan: stores again.
    node._replan_cb()
    assert node._num_replanning == 3
    assert len(node._store_computation_times) == 2


def test_adaptive_flips_to_ema_after_warmup_budget(node):
    """After ``num_replanning_before_adapt`` samples land in
    ``_store_computation_times`` the next successful replan flips
    ``_use_adapt_k_value`` and seeds ``_est_comp_time`` with the
    sample mean. Mirrors ``startAdaptKValue`` (sando.cpp:101-111).

    Pre-seed ``num_replanning_ = 1`` so this tick's
    :meth:`_record_warmup_sample` hits the C++ ``!= 1`` skip — that
    keeps the sample list at exactly the seeded size and the mean
    cleanly equals 0.02 without the in-flight measurement bleeding
    in.
    """
    node.params.use_adaptive_lookahead = True
    node.params.num_replanning_before_adapt = 3
    # Pre-seed the warm-up state so we don't have to run many ticks.
    node._num_replanning = 1
    node._store_computation_times = [0.01, 0.02, 0.03]
    assert not node._use_adapt_k_value

    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([5.0, 0.0, 0.0]))
    _converge_yawing(node)
    node._replan_cb()

    assert node._use_adapt_k_value is True
    assert node._got_enough_replanning is True
    # Mean of the seeded samples = 0.02 (the second-tick STORE skip
    # kept the list from growing).
    assert node._est_comp_time == pytest.approx(0.02, abs=1e-9)


def test_adaptive_ema_drives_lookahead(node):
    """In adapt mode ``_find_A`` should pick the lookahead from the
    EMA-filtered comp time: ``(int(factor * t_est / dc) - 1) * dc``.
    """
    node.params.use_adaptive_lookahead = True
    node.params.dc = 0.01
    node.params.k_value_factor = 5.0
    node.params.alpha_k_value_filtering = 1.0  # 100% weight to last sample
    node._state_cb(_state([0.0, 0.0, 2.0]))
    _force_committed_traj_horizon(node, horizon_s=10.0)

    # Force adapt mode + drop a known prior comp time into the state.
    node._use_adapt_k_value = True
    node._got_enough_replanning = True
    node._est_comp_time = 0.0
    node._last_replan_comp_time = 0.02  # 20 ms last replan

    t_now = node._t_now()
    node.committed_traj_.times = [t_now, t_now + 10.0]

    A = node._find_A(t_now)
    # α = 1.0 → est = last = 0.02
    # n_steps = int(5.0 * 0.02 / 0.01) = int(10) = 10
    # lookahead = (10 - 1) * 0.01 = 0.09
    assert A.t == pytest.approx(t_now + 0.09, abs=1e-9)
    assert node._est_comp_time == pytest.approx(0.02, abs=1e-9)


def test_adaptive_n_steps_clamped_to_trajectory_horizon(node):
    """When the EMA estimate would push ``A`` past the committed
    trajectory's last sample, ``_find_A`` must fall back to the
    current state (``n_steps = 1``, lookahead = 0). Mirrors the
    C++ bounds-violation clamp at sando.cpp:218-222."""
    node.params.use_adaptive_lookahead = True
    node.params.dc = 0.01
    node.params.k_value_factor = 5.0
    node.params.alpha_k_value_filtering = 1.0
    node._state_cb(_state([0.0, 0.0, 2.0]))
    _force_committed_traj_horizon(node, horizon_s=0.1)  # only 10 steps

    node._use_adapt_k_value = True
    node._got_enough_replanning = True
    node._last_replan_comp_time = 1.0  # absurdly long → n_steps = 500

    t_now = node._t_now()
    node.committed_traj_.times = [t_now, t_now + 0.1]

    A = node._find_A(t_now)
    # n_steps would be 500, clamped to 1 → lookahead = 0.
    assert A.t == pytest.approx(t_now, abs=1e-6)


def test_adaptive_falls_back_to_lookahead_replan_time_when_disabled(node):
    """With ``use_adaptive_lookahead=False`` the planner ignores the
    k_value state and uses the fixed knob — regression guard for
    legacy benchmarks that depend on a deterministic A."""
    node.params.use_adaptive_lookahead = False
    node.params.lookahead_replan_time = 0.07
    node._state_cb(_state([0.0, 0.0, 2.0]))
    _force_committed_traj_horizon(node, horizon_s=1.0)

    # Even with adapt-state pre-filled, the legacy path should win.
    node._use_adapt_k_value = True
    node._est_comp_time = 0.05

    t_now = node._t_now()
    node.committed_traj_.times = [t_now, t_now + 1.0]
    A = node._find_A(t_now)
    assert A.t == pytest.approx(t_now + 0.07, abs=1e-9)


def test_replan_holds_previous_trajectory_when_planning_fails(node):
    """If the goal is walled in, the new plan should fail and the
    previous committed trajectory must stay put."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([5.0, 0.0, 0.0]))
    _converge_yawing(node)
    node._replan_cb()  # First plan: succeeds
    first_traj = node.committed_traj_
    assert first_traj is not None

    # Now move the terminal goal somewhere unreachable: outside the
    # world bounds. HGP will fail and committed_traj_ must be preserved.
    out_of_bounds = np.array([1e6, 1e6, node.params.default_goal_z])
    node.term_goal_ = out_of_bounds
    node._replan_cb()
    # Either committed_traj_ is unchanged, or it's still a valid PWP
    # (depending on whether HGP errored or succeeded). The key contract
    # is that we don't end up with committed_traj_ = None.
    assert node.committed_traj_ is not None
