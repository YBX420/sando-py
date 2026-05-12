"""Tests for sando_py.hgp — VoxelMap, A*, and HGPPlanner."""

import math

import numpy as np
import pytest

from sando_py.core import DynTraj, Parameters, RobotState, TrajMode
from sando_py.hgp import HGPPlanner, VoxelMap, astar, octile, simplify_path_los


# ---------------------------------------------------------------------------
# VoxelMap basics
# ---------------------------------------------------------------------------

def test_voxel_map_shape_and_round_trip():
    vm = VoxelMap.from_bounds_2d((0.0, 10.0), (-5.0, 5.0), res=0.5, z_plane=2.0)
    assert vm.shape == (20, 20, 1)
    # Round-trip a point through world<->idx
    p = np.array([3.25, -1.75, 2.0])
    ix = vm.world_to_idx(p)
    centered = vm.idx_to_world(ix)
    assert np.linalg.norm(centered[:2] - p[:2]) <= vm.res / math.sqrt(2.0) + 1e-9


def test_voxel_map_in_bounds_and_default_free():
    vm = VoxelMap.from_bounds_2d((0.0, 1.0), (0.0, 1.0), res=0.25, z_plane=0.0)
    assert vm.in_bounds((0, 0, 0))
    assert not vm.in_bounds((-1, 0, 0))
    assert not vm.in_bounds((4, 0, 0))  # exclusive upper bound (shape (4,4,1))
    assert vm.is_occupied((-1, 0, 0))   # out-of-bounds counts as blocked
    assert not vm.is_occupied((0, 0, 0))


def test_voxel_map_mark_and_clear_circle():
    vm = VoxelMap.from_bounds_2d((-5, 5), (-5, 5), res=0.5, z_plane=0.0)
    flipped = vm.mark_circle([0.0, 0.0, 0.0], radius=1.0)
    assert flipped > 0
    assert vm.is_occupied(vm.world_to_idx([0.0, 0.0, 0.0]))
    # A cell well outside the circle must still be free
    assert not vm.is_occupied(vm.world_to_idx([3.0, 3.0, 0.0]))
    # Clearing removes them
    vm.clear_circle([0.0, 0.0, 0.0], radius=1.0)
    assert not vm.is_occupied(vm.world_to_idx([0.0, 0.0, 0.0]))


def test_voxel_map_from_bounds_3d_creates_full_volume():
    """v2: full 3D maps allocate ``nx × ny × nz`` cells."""
    vm = VoxelMap.from_bounds((0.0, 4.0), (0.0, 4.0), (1.0, 3.0), res=0.5)
    assert vm.shape == (8, 8, 4)
    assert vm.nz == 4


def test_voxel_map_mark_points_rasterises_into_occupancy():
    """mark_points should flip every cell containing a point and leave
    everything else free. Out-of-bounds points are silently dropped."""
    vm = VoxelMap.from_bounds((-2, 2), (-2, 2), (-2, 2), res=0.5)
    points = np.array([
        [0.0, 0.0, 0.0],          # in
        [-1.5, 0.5, 1.0],         # in
        [10.0, 10.0, 10.0],       # out of bounds (silent drop)
    ], dtype=float)
    flipped = vm.mark_points(points)
    assert flipped == 2
    assert vm.is_occupied(vm.world_to_idx([0.0, 0.0, 0.0]))
    assert vm.is_occupied(vm.world_to_idx([-1.5, 0.5, 1.0]))
    assert not vm.is_occupied(vm.world_to_idx([1.5, 1.5, 1.5]))


def test_voxel_map_mark_points_inflation_widens_footprint():
    """inflation_cells=1 should flip the 3x3x3 neighbourhood of each
    point (Chebyshev-1 dilation)."""
    vm = VoxelMap.from_bounds((-2, 2), (-2, 2), (-2, 2), res=0.5)
    vm.mark_points(np.array([[0.0, 0.0, 0.0]]), inflation_cells=1)
    # All neighbours of the centre cell should be occupied
    centre = vm.world_to_idx([0.0, 0.0, 0.0])
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                nb = (centre[0] + di, centre[1] + dj, centre[2] + dk)
                if vm.in_bounds(nb):
                    assert vm.is_occupied(nb)


def test_voxel_map_mark_points_empty_returns_zero():
    vm = VoxelMap.from_bounds((-2, 2), (-2, 2), (-2, 2), res=0.5)
    assert vm.mark_points(np.zeros((0, 3))) == 0
    assert vm.mark_points(None) == 0


def test_voxel_map_mark_sphere_3d_only_inflates_inside_radius():
    """v2: sphere marker fills cells within radius across all 3 axes."""
    vm = VoxelMap.from_bounds((-2, 2), (-2, 2), (-2, 2), res=0.5)
    flipped = vm.mark_sphere([0.0, 0.0, 0.0], radius=1.0)
    assert flipped > 0
    assert vm.is_occupied(vm.world_to_idx([0.0, 0.0, 0.0]))
    # Outside the sphere stays free
    assert not vm.is_occupied(vm.world_to_idx([1.8, 1.8, 1.8]))


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

def test_octile_zero_and_axis_aligned():
    assert octile((0, 0, 0), (0, 0, 0)) == pytest.approx(0.0)
    assert octile((0, 0, 0), (3, 0, 0)) == pytest.approx(3.0)
    # Pure planar diagonal: 5 sqrt2
    assert octile((0, 0, 0), (5, 5, 0)) == pytest.approx(5 * math.sqrt(2.0))
    # 3D diagonal: 5 sqrt3
    assert octile((0, 0, 0), (5, 5, 5)) == pytest.approx(5 * math.sqrt(3.0))


def test_astar_straight_line_on_empty_map():
    vm = VoxelMap.from_bounds_2d((-5, 5), (-5, 5), res=0.5, z_plane=0.0)
    s = vm.world_to_idx([0.0, 0.0, 0.0])
    g = vm.world_to_idx([3.0, 0.0, 0.0])
    path = astar(s, g, vm)
    assert path is not None
    assert path[0] == s
    assert path[-1] == g
    # Diagonals not needed; should be all in a straight line
    assert all(p[1] == s[1] for p in path)


def test_astar_routes_around_wall():
    vm = VoxelMap.from_bounds_2d((-2, 8), (-3, 3), res=0.5, z_plane=0.0)
    # Vertical wall at x=3 from y=-2 to y=2, with gaps above and below.
    # radius=0.4 > res/sqrt(2) so each disc fully blocks its column.
    for y in np.arange(-2.0, 2.01, 0.25):
        vm.mark_circle([3.0, float(y), 0.0], radius=0.4)
    s = vm.world_to_idx([0.0, 0.0, 0.0])
    g = vm.world_to_idx([6.0, 0.0, 0.0])
    path = astar(s, g, vm)
    assert path is not None
    # Path must clear the wall — either above (y>2) or below (y<-2)
    ys = [vm.idx_to_world(ix)[1] for ix in path]
    assert max(abs(y) for y in ys) > 2.0


def test_astar_returns_none_when_goal_blocked():
    vm = VoxelMap.from_bounds_2d((-2, 8), (-3, 3), res=0.5, z_plane=0.0)
    vm.mark_circle([6.0, 0.0, 0.0], radius=0.5)
    s = vm.world_to_idx([0.0, 0.0, 0.0])
    g = vm.world_to_idx([6.0, 0.0, 0.0])
    assert astar(s, g, vm) is None


def test_astar_3d_routes_around_sphere_in_z():
    """v2: full 3D A* — a wall in the y=0 plane can be passed over or
    under in z, not just routed around in y."""
    vm = VoxelMap.from_bounds((-2, 8), (-3, 3), (-3, 3), res=0.5)
    # Vertical wall: a plate of obstacles at x=3, y∈[-2,2], z=0
    for y in np.arange(-2.0, 2.01, 0.25):
        vm.mark_sphere([3.0, float(y), 0.0], radius=0.4)
    s = vm.world_to_idx([0.0, 0.0, 0.0])
    g = vm.world_to_idx([6.0, 0.0, 0.0])
    path = astar(s, g, vm)
    assert path is not None
    # Some waypoint should deviate in z OR y to clear the wall
    deviated = False
    for ix in path:
        wp = vm.idx_to_world(ix)
        if abs(wp[1]) > 0.6 or abs(wp[2]) > 0.6:
            deviated = True
            break
    assert deviated


def test_simplify_path_los_collapses_diagonal_staircase():
    vm = VoxelMap.from_bounds_2d((0, 10), (0, 10), res=0.5, z_plane=0.0)
    # A staircase: alternating right / up (2D indices accepted)
    path = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    out = simplify_path_los(path, vm)
    # Empty map -> single straight shot (output normalised to 3D indices)
    assert out == [(0, 0, 0), (3, 3, 0)]


# ---------------------------------------------------------------------------
# HGPPlanner end-to-end
# ---------------------------------------------------------------------------

def _planner_params(**overrides) -> Parameters:
    p = Parameters()
    p.res = 0.3
    p.factor_hgp = 1.0
    p.inflation_hgp = 0.0
    p.drone_radius = 0.2
    p.x_min, p.x_max = -50.0, 50.0
    p.y_min, p.y_max = -50.0, 50.0
    p.z_min, p.z_max = 0.5, 6.0
    p.use_free_start = True
    p.free_start_factor = 2.0
    p.use_free_goal = False
    p.global_planner_heuristic_weight = 1.0
    p.max_num_expansion = 100_000
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _stationary_obstacle(obs_id: int, pos, bbox=(0.4, 0.4, 0.4)) -> DynTraj:
    """Build a DynTraj that just sits at ``pos`` for all t."""
    t = DynTraj()
    t.id = obs_id
    t.mode = TrajMode.Analytic
    t.bbox = np.asarray(bbox, dtype=float)
    t.current_pos = np.asarray(pos, dtype=float)
    t.traj_x = str(float(pos[0]))
    t.traj_y = str(float(pos[1]))
    t.traj_z = str(float(pos[2]))
    t.traj_vx = "0.0"
    t.traj_vy = "0.0"
    t.traj_vz = "0.0"
    assert t.compileAnalytic()
    return t


def _start_state(pos) -> RobotState:
    s = RobotState()
    s.setPos(*pos)
    return s


def test_planner_empty_world_returns_straight_line():
    params = _planner_params()
    pl = HGPPlanner(params)
    path = pl.plan(_start_state([0.0, 0.0, 2.0]),
                   np.array([10.0, 0.0, 2.0]),
                   obstacles_t=[],
                   t_now=0.0)
    assert path is not None
    assert len(path) >= 2
    assert np.allclose(path[0], [0.0, 0.0, 2.0])
    assert np.allclose(path[-1], [10.0, 0.0, 2.0])
    # No obstacles -> simplifier should collapse to just endpoints
    assert len(path) == 2


def test_planner_routes_around_single_obstacle():
    params = _planner_params()
    pl = HGPPlanner(params)
    obs = [_stationary_obstacle(1, [5.0, 0.0, 2.0], bbox=(2.0, 2.0, 2.0))]
    path = pl.plan(_start_state([0.0, 0.0, 2.0]),
                   np.array([10.0, 0.0, 2.0]),
                   obstacles_t=obs,
                   t_now=0.0)
    assert path is not None
    # The straight line passes through the obstacle; the planner must
    # dodge it. With v2's 3D map the dodge can be in y *or* z — either
    # direction proves the planner detoured around the inflated sphere.
    max_y = max(abs(p[1]) for p in path)
    max_z_dev = max(abs(p[2] - 2.0) for p in path)
    assert max_y > 0.5 or max_z_dev > 0.5


def test_planner_obeys_inflation_hgp():
    """Bigger inflation should produce a wider detour."""
    obs = [_stationary_obstacle(1, [5.0, 0.0, 2.0], bbox=(1.0, 1.0, 1.0))]

    p_small = _planner_params(inflation_hgp=0.0)
    p_big = _planner_params(inflation_hgp=1.5)

    path_small = HGPPlanner(p_small).plan(
        _start_state([0.0, 0.0, 2.0]),
        np.array([10.0, 0.0, 2.0]),
        obstacles_t=obs, t_now=0.0,
    )
    path_big = HGPPlanner(p_big).plan(
        _start_state([0.0, 0.0, 2.0]),
        np.array([10.0, 0.0, 2.0]),
        obstacles_t=obs, t_now=0.0,
    )
    assert path_small is not None
    assert path_big is not None
    assert max(abs(p[1]) for p in path_big) >= max(abs(p[1]) for p in path_small)


def test_planner_returns_none_when_goal_walled_in():
    """A full cordon of obstacles around the goal should fail A*."""
    params = _planner_params()
    pl = HGPPlanner(params)
    obs = [
        _stationary_obstacle(i, pos, bbox=(0.8, 0.8, 0.8))
        for i, pos in enumerate(
            [
                [9.0, 0.0, 2.0], [11.0, 0.0, 2.0],
                [10.0, 1.0, 2.0], [10.0, -1.0, 2.0],
                [9.0, 1.0, 2.0], [11.0, -1.0, 2.0],
                [9.0, -1.0, 2.0], [11.0, 1.0, 2.0],
                [10.0, 0.7, 2.0], [10.0, -0.7, 2.0],
                [9.3, 0.5, 2.0], [10.7, 0.5, 2.0],
                [9.3, -0.5, 2.0], [10.7, -0.5, 2.0],
            ]
        )
    ]
    # use_free_goal=False (default) so the goal cell stays blocked
    path = pl.plan(_start_state([0.0, 0.0, 2.0]),
                   np.array([10.0, 0.0, 2.0]),
                   obstacles_t=obs,
                   t_now=0.0)
    assert path is None


def test_planner_free_start_clears_stuck_agent():
    """Even if the agent starts inside an obstacle, use_free_start should
    let A* escape it."""
    params = _planner_params(use_free_start=True, free_start_factor=3.0,
                             drone_radius=0.3)
    pl = HGPPlanner(params)
    obs = [_stationary_obstacle(1, [0.0, 0.0, 2.0], bbox=(0.6, 0.6, 0.6))]
    path = pl.plan(_start_state([0.0, 0.0, 2.0]),
                   np.array([10.0, 0.0, 2.0]),
                   obstacles_t=obs,
                   t_now=0.0)
    # Without use_free_start this would (likely) fail; with it, A* finds a path.
    assert path is not None
    assert np.allclose(path[-1], [10.0, 0.0, 2.0])


def test_planner_routes_around_cloud_obstacle():
    """The planner should treat ``cloud_points`` like static occupancy:
    a wall of cloud points across the corridor must force a detour."""
    params = _planner_params()
    pl = HGPPlanner(params)
    # No DynTraj obstacles — only a point-cloud "wall" between start and goal
    wall = []
    for y in np.arange(-2.0, 2.01, 0.2):
        for z in np.arange(1.8, 2.21, 0.2):
            wall.append([5.0, float(y), float(z)])
    cloud = np.array(wall, dtype=float)

    path = pl.plan(
        _start_state([0.0, 0.0, 2.0]),
        np.array([10.0, 0.0, 2.0]),
        obstacles_t=[], t_now=0.0,
        cloud_points=cloud,
    )
    assert path is not None
    # The straight line passes through the wall; the planner must
    # detour by *some* visible amount in y or z. The wall is only
    # 0.4m thick in z and 4m wide in y, so the cheapest detour goes
    # in z by a couple of cells — we expect at least 0.3m of
    # deviation in one dimension.
    max_y = max(abs(p[1]) for p in path)
    max_z_dev = max(abs(p[2] - 2.0) for p in path)
    assert max_y > 0.3 or max_z_dev > 0.3


def test_planner_no_cloud_keeps_straight_line():
    """Sanity check: when no cloud is given, the planner falls back to
    the pre-v2.1 behaviour (just trajectory snapshots + heat)."""
    params = _planner_params()
    pl = HGPPlanner(params)
    path = pl.plan(
        _start_state([0.0, 0.0, 2.0]),
        np.array([10.0, 0.0, 2.0]),
        obstacles_t=[], t_now=0.0,
        cloud_points=None,
    )
    assert path is not None
    # Empty world + no cloud -> single-segment straight line
    assert len(path) == 2


def test_planner_heat_weight_biases_away_from_dynamic_obstacle():
    """v2: with heat_weight > 0 and a dynamic obstacle near the
    straight-line path, the planner should detour further than it
    would with no heat overlay."""
    # Sine-wave obstacle that will cross y=0 during the lookahead window
    t = DynTraj()
    t.id = 99
    t.mode = TrajMode.Analytic
    t.bbox = np.array([0.6, 0.6, 0.6])
    t.current_pos = np.array([5.0, 1.5, 2.0])
    t.traj_x = "5.0"
    t.traj_y = "1.5 - 1.5 * sin(t)"  # at t=pi/2 reaches y=0
    t.traj_z = "2.0"
    t.traj_vx = "0.0"
    t.traj_vy = "-1.5 * cos(t)"
    t.traj_vz = "0.0"
    assert t.compileAnalytic()

    base = _planner_params(
        heat_weight=0.0, heat_alpha0=0.0, heat_alpha1=0.0,
        horizon=math.pi,
    )
    biased = _planner_params(
        heat_weight=20.0, heat_alpha0=1.0, heat_alpha1=4.0,
        heat_p=2, heat_q=2,
        dyn_heat_tube_radius_m=1.5,
        heat_num_samples=8,
        horizon=math.pi,
    )

    path_base = HGPPlanner(base).plan(
        _start_state([0.0, 0.0, 2.0]),
        np.array([10.0, 0.0, 2.0]),
        obstacles_t=[t], t_now=0.0,
    )
    path_biased = HGPPlanner(biased).plan(
        _start_state([0.0, 0.0, 2.0]),
        np.array([10.0, 0.0, 2.0]),
        obstacles_t=[t], t_now=0.0,
    )
    assert path_base is not None and path_biased is not None
    # The heat-biased plan should stray further from y=0 than the base
    # one because the obstacle's predicted swing makes the corridor
    # near y=0 costly.
    max_dev_base = max(abs(p[1]) for p in path_base)
    max_dev_biased = max(abs(p[1]) for p in path_biased)
    assert max_dev_biased >= max_dev_base


def test_planner_dynamic_obstacle_eval_at_t():
    """Make sure obstacles are sampled at ``t_now``, not at t=0."""
    # Sine-wave obstacle that's safely off the corridor at t=0 but
    # blocking the path at t=pi/2.
    t = DynTraj()
    t.id = 99
    t.mode = TrajMode.Analytic
    t.bbox = np.array([1.0, 1.0, 1.0])
    t.current_pos = np.array([5.0, 5.0, 2.0])
    t.traj_x = "5.0"
    t.traj_y = "5.0 - 5.0 * sin(t)"  # at t=pi/2 -> y=0 i.e. on the corridor
    t.traj_z = "2.0"
    t.traj_vx = "0.0"
    t.traj_vy = "-5.0 * cos(t)"
    t.traj_vz = "0.0"
    assert t.compileAnalytic()

    params = _planner_params()
    pl = HGPPlanner(params)

    path_t0 = pl.plan(_start_state([0.0, 0.0, 2.0]),
                      np.array([10.0, 0.0, 2.0]),
                      obstacles_t=[t], t_now=0.0)
    path_tpi2 = pl.plan(_start_state([0.0, 0.0, 2.0]),
                        np.array([10.0, 0.0, 2.0]),
                        obstacles_t=[t], t_now=math.pi / 2)
    assert path_t0 is not None and path_tpi2 is not None
    # At t=0 the obstacle is at (5,5,2) — far from the y=0 corridor — so
    # the planner should not need to dodge. At t=pi/2 it's on the corridor,
    # so the planner should detour. Therefore tpi2's max |y| should exceed t0's.
    max_y_t0 = max(abs(p[1]) for p in path_t0)
    max_y_tpi2 = max(abs(p[1]) for p in path_tpi2)
    assert max_y_tpi2 > max_y_t0
