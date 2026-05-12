"""Tests for the hover-avoidance port (sando.cpp:2322-2492).

Covers:
  * Parameter wiring (defaults + YAML).
  * ``_is_point_threatened`` with and without lookahead.
  * ``_check_hover_avoidance`` repulsion + rotation-fallback + return-to-hover.
  * State-machine integration via ``_replan_cb``: GOAL_REACHED ->
    HOVER_AVOIDING -> back-to-hover.
  * ``_compute_yaw_setpoint`` HOVER_AVOIDING branch faces ``p_hover_``.

Run from a ROS-sourced shell:

    scripts/with_ros_env.sh python3 -m pytest tests/ros/test_hover_avoidance.py -q
"""

import math

import numpy as np
import pytest

dim = pytest.importorskip("dynus_interfaces.msg")
gm = pytest.importorskip("geometry_msgs.msg")
rclpy = pytest.importorskip("rclpy")

from sando_py.core import PieceWisePol
from sando_py.core.dyn_traj import DynTraj, TrajMode
from sando_py.sim_io import DroneStatus, SandoNode


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
    n.params.hover_avoidance_enabled = True
    n.params.hover_avoidance_2d = True
    n.params.hover_avoidance_d_trigger = 4.0
    n.params.hover_avoidance_h = 3.0
    n.params.hover_avoidance_min_repulsion_norm = 0.01
    yield n
    n.destroy_node()


def _state(pos):
    msg = dim.State()
    msg.pos.x, msg.pos.y, msg.pos.z = float(pos[0]), float(pos[1]), float(pos[2])
    msg.vel.x = msg.vel.y = msg.vel.z = 0.0
    msg.quat.x = msg.quat.y = msg.quat.z = 0.0
    msg.quat.w = 1.0
    return msg


def _pose_stamped(pos):
    p = gm.PoseStamped()
    p.pose.position.x, p.pose.position.y, p.pose.position.z = (
        float(pos[0]), float(pos[1]), float(pos[2]),
    )
    p.pose.orientation.w = 1.0
    return p


def _make_static_dyn_traj(obs_id: int, pos, *, is_agent: bool = False) -> DynTraj:
    """A piecewise DynTraj that stays at ``pos`` over [0, 1e6]."""
    pwp = PieceWisePol()
    pwp.times = [0.0, 1e6]
    pwp.coeff_x = [np.array([0.0, 0.0, 0.0, float(pos[0])], dtype=float)]
    pwp.coeff_y = [np.array([0.0, 0.0, 0.0, float(pos[1])], dtype=float)]
    pwp.coeff_z = [np.array([0.0, 0.0, 0.0, float(pos[2])], dtype=float)]

    tr = DynTraj()
    tr.mode = TrajMode.Piecewise
    tr.pwp = pwp
    tr.bbox = np.array([0.5, 0.5, 0.5], dtype=float)
    tr.id = obs_id
    tr.is_agent = is_agent
    tr.current_pos = np.asarray(pos, dtype=float)
    tr.time_received = 0.0
    return tr


def _make_orbiting_dyn_traj(obs_id: int, center_xy, radius: float, period: float,
                            z: float = 2.0) -> DynTraj:
    """An analytic DynTraj orbiting in the XY plane — used to test the
    15s lookahead window for periodic obstacles."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    omega = 2 * math.pi / period
    tr = DynTraj()
    tr.mode = TrajMode.Analytic
    tr.traj_x = f"{cx} + {radius} * cos({omega} * t)"
    tr.traj_y = f"{cy} + {radius} * sin({omega} * t)"
    tr.traj_z = f"{z}"
    ok = tr.compileAnalytic()
    assert ok, "analytic compile failed"
    tr.id = obs_id
    tr.bbox = np.array([0.5, 0.5, 0.5], dtype=float)
    tr.current_pos = tr.eval(0.0)
    tr.time_received = 0.0
    return tr


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def test_parameters_carry_hover_defaults(node):
    """The Parameters dataclass exposes the C++ defaults from
    sando_type.hpp:201-205."""
    p = node.params
    # Test fixture overrides these; check the dataclass *defaults* via a
    # freshly-constructed instance.
    from sando_py.core import Parameters
    fresh = Parameters()
    assert fresh.hover_avoidance_enabled is False
    assert fresh.hover_avoidance_2d is True
    assert fresh.hover_avoidance_d_trigger == pytest.approx(4.0)
    assert fresh.hover_avoidance_h == pytest.approx(3.0)
    assert fresh.hover_avoidance_min_repulsion_norm == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# _is_point_threatened
# ---------------------------------------------------------------------------

def test_is_point_threatened_current_pos_inside_trigger(node):
    """A static obstacle at distance < d_trigger from the query point
    must be flagged as threatening regardless of lookahead."""
    node.trajs_[1] = _make_static_dyn_traj(1, [3.0, 0.0, 2.0])
    # d_trigger = 4.0; point at origin is 3 m from obstacle.
    assert node._is_point_threatened(np.array([0.0, 0.0, 2.0]),
                                     t_now=0.0, lookahead=True)
    assert node._is_point_threatened(np.array([0.0, 0.0, 2.0]),
                                     t_now=0.0, lookahead=False)


def test_is_point_threatened_far_obstacle_not_flagged(node):
    """An obstacle beyond d_trigger doesn't threaten the query point."""
    node.trajs_[1] = _make_static_dyn_traj(1, [10.0, 0.0, 2.0])
    assert not node._is_point_threatened(np.array([0.0, 0.0, 2.0]),
                                         t_now=0.0, lookahead=True)


def test_is_point_threatened_lookahead_catches_orbit(node):
    """An orbiting obstacle currently far away but predicted to pass
    near the query point inside the 15 s lookahead must be flagged
    iff lookahead is enabled — mirrors C++ sando.cpp:2347-2358."""
    # Orbit radius 5, period 5s, centered at origin. At t=0 it's at (5,0,2).
    # At t=2.5 it's at (-5, 0, 2). Across 15 s it sweeps multiple full
    # orbits and at some point comes within 4 m of (0, 0, 2) — actually
    # the orbit *itself* is 5m away always; we need an obstacle whose
    # orbit physically intersects the d_trigger ball around the query.
    # Use radius 3 so the orbit passes within 3m of origin -> always
    # threatening. Better: radius 5 and a query point at (3, 0, 2);
    # at t=0 obstacle is at (5, 0, 2) -> 2m away (inside trigger).
    # For a non-current-but-future case, start the obstacle outside
    # trigger and let it sweep into trigger.
    # Cleaner construction: obstacle orbits at radius 5 around origin,
    # query at (8, 0, 2). At t=0 obstacle at (5,0,2) -> 3 m away
    # (inside d_trigger=4). To make a "far now, near later" case, use
    # query point near the *back* of the orbit:
    #   obstacle starts at +x (5,0,2), query at -x side (-8,0,2).
    #   At t=0: distance = 13. At t=period/2 (=2.5s): obs at (-5,0,2),
    #   distance = 3 (< d_trigger=4).
    node.trajs_[1] = _make_orbiting_dyn_traj(
        1, center_xy=(0.0, 0.0), radius=5.0, period=5.0, z=2.0,
    )
    query = np.array([-8.0, 0.0, 2.0])

    # Without lookahead: only checks current_pos -> 13m away -> safe.
    assert not node._is_point_threatened(query, t_now=0.0, lookahead=False)
    # With lookahead: the obstacle sweeps to (-5,0,2) at t=2.5 inside
    # 15s window, distance 3 < trigger -> threatened.
    assert node._is_point_threatened(query, t_now=0.0, lookahead=True)


def test_is_point_threatened_pwp_clamps_to_last_time(node):
    """Piecewise trajectories with ``pwp.times.back() < t_now`` should
    only contribute their current_pos to the threat check (no stale
    extrapolation)."""
    tr = _make_static_dyn_traj(1, [10.0, 0.0, 2.0])
    tr.pwp.times = [0.0, 1.0]  # expired by t_now = 100
    tr.current_pos = np.array([10.0, 0.0, 2.0])
    node.trajs_[1] = tr
    # Query 11 m from current_pos -> safe even with lookahead.
    assert not node._is_point_threatened(np.array([-1.0, 0.0, 2.0]),
                                         t_now=100.0, lookahead=True)


# ---------------------------------------------------------------------------
# _check_hover_avoidance — core logic
# ---------------------------------------------------------------------------

def _setup_hovering(node, agent_pos=(0.0, 0.0, 2.0), goal_pos=(0.0, 0.0, 2.0)) -> None:
    """Drop the agent at the goal and put it in HOVER_AVOIDING."""
    node._state_cb(_state(agent_pos))
    node._term_goal_cb(_pose_stamped(goal_pos))
    # Skip yaw convergence — go straight into hovering.
    node._set_status(DroneStatus.HOVER_AVOIDING)
    node.p_hover_ = np.array(goal_pos, dtype=float)


def test_check_hover_avoidance_no_obstacle_in_goal_reached_returns_false(node):
    """``GOAL_REACHED`` with no obstacles: the helper returns False so
    the replan tick simply holds — there's nothing to do."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([0.0, 0.0, 2.0]))
    node._set_status(DroneStatus.GOAL_REACHED)
    assert node.trajs_ == {}
    assert not node._check_hover_avoidance(t_now=0.0)


def test_check_hover_avoidance_no_obstacle_in_hover_avoiding_returns_to_hover(node):
    """``HOVER_AVOIDING`` with the obstacle already cleared: the helper
    returns True and sets ``term_goal_`` back to ``p_hover_`` so the
    agent drifts home — mirrors C++ sando.cpp:2467-2484."""
    _setup_hovering(node)
    assert node.trajs_ == {}
    proceed = node._check_hover_avoidance(t_now=0.0)
    assert proceed
    assert np.allclose(node.term_goal_, node.p_hover_, atol=1e-9)


def test_check_hover_avoidance_obstacle_triggers_evasion(node):
    """An obstacle within d_trigger generates a non-zero repulsion
    vector and a new evasion ``term_goal_``."""
    _setup_hovering(node)
    # Put an obstacle 2 m to the +x of the agent. d_trigger=4 -> in range.
    node.trajs_[1] = _make_static_dyn_traj(1, [2.0, 0.0, 2.0])

    proceed = node._check_hover_avoidance(t_now=0.0)
    assert proceed, "should signal continue-replanning when avoidance triggers"
    assert node.status_ == DroneStatus.HOVER_AVOIDING
    # term_goal_ overwritten with an evasion point — agent is pushed
    # away from the obstacle (in -x direction).
    assert node.term_goal_[0] < 0.0
    # 2D mode -> altitude held.
    assert node.term_goal_[2] == pytest.approx(2.0)
    # Distance from agent to evasion point should equal hover_avoidance_h.
    h = node.params.hover_avoidance_h
    agent_pos = node.current_state_.pos
    assert np.linalg.norm(node.term_goal_ - agent_pos) == pytest.approx(h, abs=1e-6)


def test_check_hover_avoidance_first_entry_snapshots_p_hover(node):
    """On the first transition from GOAL_REACHED into HOVER_AVOIDING,
    ``p_hover_`` snapshots the current terminal goal."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([0.0, 0.0, 2.0]))
    # p_hover_ should already be set from _term_goal_cb (sando.cpp:1836).
    initial_p_hover = node.p_hover_.copy()
    node._set_status(DroneStatus.GOAL_REACHED)

    # Obstacle approaches -> should enter HOVER_AVOIDING and re-snapshot
    # p_hover_ from term_goal_ (which is still the user-requested goal).
    node.trajs_[1] = _make_static_dyn_traj(1, [2.0, 0.0, 2.0])
    proceed = node._check_hover_avoidance(t_now=0.0)
    assert proceed
    assert node.status_ == DroneStatus.HOVER_AVOIDING
    # p_hover_ should match the original goal, not the evasion point.
    assert np.allclose(node.p_hover_, initial_p_hover, atol=1e-9)


def test_check_hover_avoidance_2d_preserves_altitude(node):
    """In 2D mode the repulsion is restricted to XY; the evasion point
    keeps the agent's altitude."""
    node.params.hover_avoidance_2d = True
    _setup_hovering(node, agent_pos=(0.0, 0.0, 2.5))

    # Obstacle below+offset — without 2D the repulsion would push up;
    # with 2D the altitude must stay at 2.5.
    node.trajs_[1] = _make_static_dyn_traj(1, [2.0, 0.0, 1.0])
    node._check_hover_avoidance(t_now=0.0)
    assert node.term_goal_[2] == pytest.approx(2.5)


def test_check_hover_avoidance_return_to_hover_when_clear(node):
    """Once the obstacle moves away, the agent's term_goal_ flips back
    to ``p_hover_`` so it drifts home."""
    _setup_hovering(node)
    # Trigger avoidance once.
    node.trajs_[1] = _make_static_dyn_traj(1, [2.0, 0.0, 2.0])
    node._check_hover_avoidance(t_now=0.0)
    assert node.status_ == DroneStatus.HOVER_AVOIDING
    # Now remove the obstacle and re-run.
    del node.trajs_[1]
    proceed = node._check_hover_avoidance(t_now=0.0)
    assert proceed, "should keep planning to return to hover"
    assert np.allclose(node.term_goal_, node.p_hover_, atol=1e-9)


def test_check_hover_avoidance_holds_when_hover_pos_still_threatened(node):
    """When the hover position itself is unsafe (an obstacle camping on
    it), the agent must stay where it is and not re-target."""
    _setup_hovering(node, agent_pos=(5.0, 0.0, 2.0), goal_pos=(0.0, 0.0, 2.0))
    # p_hover_ is at origin. Put an obstacle right at the hover position
    # and the agent far enough away that its own neighbourhood is clear.
    node.trajs_[1] = _make_static_dyn_traj(1, [0.0, 0.0, 2.0])
    # Agent at (5, 0, 2) is 5 m from obstacle -> outside d_trigger=4.
    proceed = node._check_hover_avoidance(t_now=0.0)
    # Agent is safe at its current position, but p_hover_ (origin) is
    # 0 m from the obstacle -> threatened -> return False (wait).
    assert not proceed


def test_check_hover_avoidance_falls_back_to_rotation_when_blocked(node):
    """If the straight repulsion direction lands on a second obstacle,
    the helper must try rotation candidates and pick a safe one.

    Use a tight d_trigger so each obstacle's threat zone is a narrow
    ball; rotating by π/6 around the repulsion direction is then enough
    to clear it. Without rotation fallback the helper would give up.
    """
    node.params.hover_avoidance_d_trigger = 1.0  # tight threat radius
    node.params.hover_avoidance_h = 3.0
    _setup_hovering(node)
    # Obstacle A at (0.8, 0, 2) -> 0.8 m from agent (< d_trigger=1) ->
    # contributes to repulsion in -x.
    node.trajs_[1] = _make_static_dyn_traj(1, [0.8, 0.0, 2.0])
    # Obstacle B exactly at the unrotated evasion (-3, 0, 2) ->
    # threatened -> rotation fallback must kick in.
    node.trajs_[2] = _make_static_dyn_traj(2, [-3.0, 0.0, 2.0])

    proceed = node._check_hover_avoidance(t_now=0.0)
    assert proceed, "rotation fallback should find a safe candidate"
    # Evasion point should not be at the unrotated landing (-3, 0, 2)
    # — it should have rotated to a clear direction in Y.
    assert abs(node.term_goal_[1]) > 0.1


# ---------------------------------------------------------------------------
# _replan_cb integration — full state-machine path
# ---------------------------------------------------------------------------

def test_replan_keeps_timer_alive_with_hover_enabled(node):
    """Reaching the goal with hover_avoidance_enabled must transition
    into HOVER_AVOIDING (not GOAL_REACHED with timer cancelled)."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([0.0, 0.0, 2.0]))
    # Skip YAWING straight to TRAVELING for test brevity.
    node._set_status(DroneStatus.TRAVELING)

    node._replan_cb()
    # We're at the goal -> first replan tick should enter HOVER_AVOIDING.
    assert node.status_ == DroneStatus.HOVER_AVOIDING
    # The replan timer must still be active (no cancel).
    assert not node.timer_replan_.is_canceled()


def test_replan_cb_goes_to_goal_reached_when_hover_disabled(node):
    """Sanity: with hover_avoidance disabled, the original GOAL_REACHED
    + timer-cancel behaviour is preserved (regression guard)."""
    node.params.hover_avoidance_enabled = False
    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([0.0, 0.0, 2.0]))
    node._set_status(DroneStatus.TRAVELING)
    node._replan_cb()
    assert node.status_ == DroneStatus.GOAL_REACHED
    assert node.timer_replan_.is_canceled()


# ---------------------------------------------------------------------------
# Yaw branch
# ---------------------------------------------------------------------------

def test_yaw_branch_faces_p_hover_in_hover_avoiding(node):
    """In HOVER_AVOIDING, the commanded yaw must march toward
    ``atan2(p_hover_ - setpoint)`` — capped at ``w_max_yawing * dc``.

    We give the hover position a slight +y offset so the desired yaw
    is unambiguously in (π/2, π) rather than ±π (where the angle_wrap
    convention would otherwise hide the direction).
    """
    node._state_cb(_state([5.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([0.0, 1.0, 2.0]))
    node._set_status(DroneStatus.HOVER_AVOIDING)
    node.p_hover_ = np.array([0.0, 1.0, 2.0])
    # Setpoint at agent's current pos (5, 0, 2). p_hover_ at (0, 1, 2).
    # dx = -5, dy = 1 -> atan2(1, -5) ≈ 2.94 rad, well inside (π/2, π).
    from sando_py.core import RobotState
    setpoint = RobotState()
    setpoint.setPos([5.0, 0.0, 2.0])
    setpoint.setVel([0.0, 0.0, 0.0])
    node.previous_yaw_ = 0.0

    new_yaw, dyaw = node._compute_yaw_setpoint(setpoint)
    max_step = node.params.w_max_yawing * node.params.dc
    # Step should be the full max_step in the positive direction
    # (desired yaw ≈ 2.94 > 0 > previous_yaw_, diff_cmd > max_step
    # -> clamped to +max_step).
    assert new_yaw == pytest.approx(max_step, abs=1e-9)
    assert dyaw == pytest.approx(max_step / node.params.dc, abs=1e-9)


def test_yaw_branch_holds_when_close_to_p_hover_(node):
    """When the setpoint is within 0.3 m of p_hover_, atan2 is unstable —
    the branch returns the previous yaw and dyaw=0."""
    node._state_cb(_state([0.1, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([0.0, 0.0, 2.0]))
    node._set_status(DroneStatus.HOVER_AVOIDING)
    node.p_hover_ = np.array([0.0, 0.0, 2.0])

    from sando_py.core import RobotState
    setpoint = RobotState()
    setpoint.setPos([0.1, 0.0, 2.0])
    node.previous_yaw_ = 1.234
    new_yaw, dyaw = node._compute_yaw_setpoint(setpoint)
    assert new_yaw == pytest.approx(1.234)
    assert dyaw == 0.0
