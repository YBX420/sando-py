"""Tests for sando_py.sim_io.viz — RViz marker / polyhedron converters
plus the wiring inside ``SandoNode`` (publishers exist, fire on the
right callbacks, throttle correctly).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip outside ROS-sourced shells
dim = pytest.importorskip("dynus_interfaces.msg")
gm = pytest.importorskip("geometry_msgs.msg")
rclpy = pytest.importorskip("rclpy")
visualization_msgs = pytest.importorskip("visualization_msgs.msg")
decomp_ros_msgs = pytest.importorskip("decomp_ros_msgs.msg")
builtin_interfaces = pytest.importorskip("builtin_interfaces.msg")

from sando_py.core import PieceWisePol, Polytope
from sando_py.sim_io import SandoNode
from sando_py.sim_io.viz import (
    color_jet,
    color_rgba,
    delete_all_marker_array,
    path_to_marker_array,
    polytopes_to_polyhedron_array,
    positions_to_marker_array,
    pwp_to_colored_marker_array,
)


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


def _stamp():
    return builtin_interfaces.Time()


def _state(pos):
    msg = dim.State()
    msg.pos.x, msg.pos.y, msg.pos.z = float(pos[0]), float(pos[1]), float(pos[2])
    msg.quat.w = 1.0
    return msg


def _pose_stamped(pos):
    p = gm.PoseStamped()
    p.pose.position.x, p.pose.position.y, p.pose.position.z = (
        float(pos[0]), float(pos[1]), float(pos[2]),
    )
    p.pose.orientation.w = 1.0
    return p


def _box(center, half):
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
# Colour helpers
# ---------------------------------------------------------------------------

def test_color_jet_low_value_is_blue():
    c = color_jet(0.0, 0.0, 1.0)
    assert c.r == pytest.approx(0.0)
    assert c.g == pytest.approx(0.0)
    assert c.b == pytest.approx(1.0)
    assert c.a == pytest.approx(1.0)


def test_color_jet_high_value_is_red():
    c = color_jet(1.0, 0.0, 1.0)
    assert c.r == pytest.approx(1.0)
    assert c.g == pytest.approx(0.0, abs=1e-6)
    assert c.b == pytest.approx(0.0, abs=1e-6)


def test_color_jet_clamps_out_of_range():
    """Out-of-range values should be clamped to the endpoints, not
    spill into invalid RGB."""
    c_hi = color_jet(100.0, 0.0, 1.0)
    assert 0.0 <= c_hi.r <= 1.0
    c_lo = color_jet(-5.0, 0.0, 1.0)
    assert 0.0 <= c_lo.b <= 1.0


def test_color_jet_degenerate_range():
    """vmin == vmax must not divide-by-zero."""
    c = color_jet(0.5, 1.0, 1.0)
    assert 0.0 <= c.r <= 1.0


# ---------------------------------------------------------------------------
# path_to_marker_array
# ---------------------------------------------------------------------------

def test_path_to_marker_array_produces_line_and_dots():
    path = [np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 2.0]),
            np.array([2.0, 1.0, 2.0])]
    ma = path_to_marker_array(path, frame_id="map", stamp=_stamp())
    assert len(ma.markers) == 2
    types = {m.type for m in ma.markers}
    assert visualization_msgs.Marker.LINE_STRIP in types
    assert visualization_msgs.Marker.SPHERE_LIST in types
    for m in ma.markers:
        assert len(m.points) == len(path)
        assert m.header.frame_id == "map"


def test_path_to_marker_array_empty_input_returns_empty_array():
    ma = path_to_marker_array([], frame_id="map", stamp=_stamp())
    assert len(ma.markers) == 0


# ---------------------------------------------------------------------------
# polytopes_to_polyhedron_array
# ---------------------------------------------------------------------------

def test_polytopes_to_polyhedron_array_box_has_six_planes():
    poly = _box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    pa = polytopes_to_polyhedron_array([poly], frame_id="map", stamp=_stamp())
    assert len(pa.polyhedrons) == 1
    ph = pa.polyhedrons[0]
    assert len(ph.points) == 6
    assert len(ph.normals) == 6


def test_polytopes_to_polyhedron_array_normals_match_box_faces():
    """Each face's normal should be a unit axis; each point should lie
    on the corresponding face."""
    poly = _box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    pa = polytopes_to_polyhedron_array([poly], frame_id="map", stamp=_stamp())
    ph = pa.polyhedrons[0]
    # First three are +x, +y, +z faces; last three are -x, -y, -z
    expected_normals = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    ]
    expected_points = [
        (1, 0, 0), (0, 2, 0), (0, 0, 3),
        (-1, 0, 0), (0, -2, 0), (0, 0, -3),
    ]
    for ph_pt, ph_n, e_pt, e_n in zip(
        ph.points, ph.normals, expected_points, expected_normals
    ):
        assert (ph_n.x, ph_n.y, ph_n.z) == pytest.approx(e_n, abs=1e-6)
        assert (ph_pt.x, ph_pt.y, ph_pt.z) == pytest.approx(e_pt, abs=1e-6)


def test_polytopes_to_polyhedron_array_empty_polys_returns_empty():
    pa = polytopes_to_polyhedron_array([], frame_id="map", stamp=_stamp())
    assert len(pa.polyhedrons) == 0


# ---------------------------------------------------------------------------
# pwp_to_colored_marker_array
# ---------------------------------------------------------------------------

def test_pwp_to_colored_marker_array_produces_line_list_with_colors():
    pwp = PieceWisePol()
    pwp.times = [0.0, 1.0]
    # Linear motion in x: p(u) = 5 * u, vel = 5 (constant)
    pwp.coeff_x = [np.array([0.0, 0.0, 5.0, 0.0])]
    pwp.coeff_y = [np.array([0.0, 0.0, 0.0, 0.0])]
    pwp.coeff_z = [np.array([0.0, 0.0, 0.0, 2.0])]
    ma = pwp_to_colored_marker_array(
        pwp, frame_id="map", stamp=_stamp(), v_max=10.0, num_samples=10,
    )
    assert len(ma.markers) == 1
    m = ma.markers[0]
    assert m.type == visualization_msgs.Marker.LINE_LIST
    # LINE_LIST: each segment is 2 points, paired with 2 colors
    assert len(m.points) == len(m.colors)
    assert len(m.points) > 0


def test_pwp_to_colored_marker_array_empty_pwp_is_empty():
    ma = pwp_to_colored_marker_array(
        PieceWisePol(), frame_id="map", stamp=_stamp(), v_max=1.0,
    )
    assert len(ma.markers) == 0


# ---------------------------------------------------------------------------
# positions_to_marker_array
# ---------------------------------------------------------------------------

def test_positions_to_marker_array_line_strip():
    positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
    ma = positions_to_marker_array(positions, frame_id="map", stamp=_stamp())
    assert len(ma.markers) == 1
    m = ma.markers[0]
    assert m.type == visualization_msgs.Marker.LINE_STRIP
    assert len(m.points) == 2


def test_positions_to_marker_array_singleton_is_empty():
    ma = positions_to_marker_array(
        [np.zeros(3)], frame_id="map", stamp=_stamp(),
    )
    assert len(ma.markers) == 0


# ---------------------------------------------------------------------------
# delete_all_marker_array
# ---------------------------------------------------------------------------

def test_delete_all_marker_array():
    ma = delete_all_marker_array(frame_id="map", stamp=_stamp())
    assert len(ma.markers) == 1
    assert ma.markers[0].action == visualization_msgs.Marker.DELETEALL


# ---------------------------------------------------------------------------
# SandoNode wiring — publishers + throttled tracking
# ---------------------------------------------------------------------------

def test_sando_node_creates_all_four_viz_publishers(node):
    """Each of the four publishers documented in sim_io/README.md must
    exist on the node after construction."""
    for attr in (
        "pub_hgp_path_marker_",
        "pub_poly_safe_",
        "pub_traj_committed_colored_",
        "pub_actual_traj_",
    ):
        assert hasattr(node, attr), f"missing {attr}"


def test_state_callback_records_actual_traj_hist(node):
    """The state callback should append to ``actual_traj_hist_`` —
    throttled, so two states in a row only add one entry."""
    node._state_cb(_state([0.0, 0.0, 2.0]))
    assert len(node.actual_traj_hist_) == 1
    # Second call same tick should be throttled out
    node._state_cb(_state([0.1, 0.0, 2.0]))
    assert len(node.actual_traj_hist_) == 1


def test_replan_publishes_plan_viz_on_success(node):
    """A successful replan should publish to all three plan-viz topics."""
    # Hook each viz publisher to count calls
    counts = {"hgp": 0, "poly": 0, "traj": 0}
    original = {
        "hgp": node.pub_hgp_path_marker_.publish,
        "poly": node.pub_poly_safe_.publish,
        "traj": node.pub_traj_committed_colored_.publish,
    }

    def _wrap(key):
        def _pub(msg):
            counts[key] += 1
            return original[key](msg)
        return _pub

    node.pub_hgp_path_marker_.publish = _wrap("hgp")
    node.pub_poly_safe_.publish = _wrap("poly")
    node.pub_traj_committed_colored_.publish = _wrap("traj")

    node._state_cb(_state([0.0, 0.0, 2.0]))
    node._term_goal_cb(_pose_stamped([5.0, 0.0, 0.0]))
    # term_goal puts us in YAWING — drive yaw to convergence (agent
    # already faces +x toward the goal, so one publish-goal tick is
    # enough) so the next replan tick actually runs the pipeline.
    from sando_py.sim_io import DroneStatus
    for _ in range(200):
        if node.status_ != DroneStatus.YAWING:
            break
        node._publish_goal_cb()
    node._replan_cb()

    assert counts["hgp"] >= 1
    assert counts["poly"] >= 1
    assert counts["traj"] >= 1


def test_publish_goal_publishes_actual_traj_when_history_present(node):
    """With a non-trivial history, _publish_goal_cb should emit an
    actual_traj MarkerArray (throttled, so the first call after history
    accumulates is enough)."""
    count = {"n": 0}
    original = node.pub_actual_traj_.publish

    def _wrap(msg):
        count["n"] += 1
        return original(msg)

    node.pub_actual_traj_.publish = _wrap

    # Manually populate the history (state_cb is throttled)
    node.actual_traj_hist_ = [
        np.array([0.0, 0.0, 2.0]),
        np.array([0.1, 0.0, 2.0]),
        np.array([0.2, 0.0, 2.0]),
    ]
    # Provide a current state so the callback doesn't bail
    node._state_cb(_state([0.2, 0.0, 2.0]))
    node._publish_goal_cb()

    assert count["n"] >= 1
