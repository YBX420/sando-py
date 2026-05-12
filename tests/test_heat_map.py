"""Tests for sando_py.hgp.heat_map — dynamic + static heat overlays."""

from __future__ import annotations

import numpy as np
import pytest

from sando_py.core import DynTraj, Parameters, TrajMode
from sando_py.hgp import VoxelMap
from sando_py.hgp.heat_map import compute_dynamic_heat, compute_static_heat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _heat_params(**overrides) -> Parameters:
    p = Parameters()
    # Dynamic heat
    p.heat_alpha0 = 1.0
    p.heat_alpha1 = 2.0
    p.heat_p = 2
    p.heat_q = 2
    p.heat_gamma = 0.0
    p.heat_Hmax = 0.0
    p.heat_tau_ratio = 1.0
    p.heat_num_samples = 5
    p.dyn_heat_tube_radius_m = 1.0
    p.obst_max_vel = 0.0
    p.horizon = 2.0
    # Static heat
    p.static_heat_enabled = False
    p.static_heat_alpha = 1.0
    p.static_heat_p = 2
    p.static_heat_rmax_m = 1.0
    p.static_heat_Hmax = 0.0
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _stationary_obstacle(pos, bbox=(0.4, 0.4, 0.4)) -> DynTraj:
    t = DynTraj()
    t.id = 1
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


def _moving_obstacle(start, vel) -> DynTraj:
    t = DynTraj()
    t.id = 1
    t.mode = TrajMode.Analytic
    t.bbox = np.array([0.3, 0.3, 0.3])
    t.current_pos = np.asarray(start, dtype=float)
    t.traj_x = f"{start[0]} + {vel[0]} * t"
    t.traj_y = f"{start[1]} + {vel[1]} * t"
    t.traj_z = f"{start[2]} + {vel[2]} * t"
    t.traj_vx = str(vel[0])
    t.traj_vy = str(vel[1])
    t.traj_vz = str(vel[2])
    assert t.compileAnalytic()
    return t


# ---------------------------------------------------------------------------
# Dynamic heat
# ---------------------------------------------------------------------------

def test_compute_dynamic_heat_creates_heat_field():
    vm = VoxelMap.from_bounds_2d((-5, 5), (-5, 5), res=0.5, z_plane=0.0)
    obs = [_stationary_obstacle([0.0, 0.0, 0.0])]
    heat = compute_dynamic_heat(vm, obs, _heat_params(), t_now=0.0)
    assert heat is vm.heat
    assert heat.shape == vm.shape
    assert heat.dtype == np.float32


def test_dynamic_heat_peaks_at_obstacle():
    """Heat is highest at the obstacle's predicted location."""
    vm = VoxelMap.from_bounds_2d((-5, 5), (-5, 5), res=0.5, z_plane=0.0)
    obs = [_stationary_obstacle([0.0, 0.0, 0.0])]
    heat = compute_dynamic_heat(vm, obs, _heat_params(), t_now=0.0)
    # Cell at the obstacle should have higher heat than a far cell
    near = vm.world_to_idx([0.0, 0.0, 0.0])
    far = vm.world_to_idx([4.5, 4.5, 0.0])
    assert vm.heat_at(near) > vm.heat_at(far)
    assert vm.heat_at(far) == pytest.approx(0.0, abs=1e-5)


def test_dynamic_heat_decays_with_distance():
    """The (1 - d/R)^p falloff means cells just outside the obstacle
    have lower heat than the obstacle's cells."""
    vm = VoxelMap.from_bounds_2d((-5, 5), (-5, 5), res=0.25, z_plane=0.0)
    p = _heat_params(dyn_heat_tube_radius_m=2.0)
    obs = [_stationary_obstacle([0.0, 0.0, 0.0])]
    compute_dynamic_heat(vm, obs, p, t_now=0.0)
    h_at_center = vm.heat_at(vm.world_to_idx([0.0, 0.0, 0.0]))
    h_at_one = vm.heat_at(vm.world_to_idx([1.0, 0.0, 0.0]))
    h_at_two = vm.heat_at(vm.world_to_idx([2.0, 0.0, 0.0]))
    assert h_at_center > h_at_one > h_at_two


def test_dynamic_heat_temporal_sampling_picks_up_predicted_path():
    """A moving obstacle should produce heat along its predicted
    trajectory, not just its current position."""
    vm = VoxelMap.from_bounds_2d((-5, 5), (-5, 5), res=0.5, z_plane=0.0)
    # Obstacle starts at (-2, 0, 0), moves +x at 1 m/s. Over a 2s
    # horizon it'll cross (0, 0) and approach (0, 0).
    obs = [_moving_obstacle([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0])]
    p = _heat_params(horizon=2.0, heat_num_samples=5)
    compute_dynamic_heat(vm, obs, p, t_now=0.0)
    # Heat should be elevated at the predicted-trajectory cells (0, 0)
    h_now = vm.heat_at(vm.world_to_idx([-2.0, 0.0, 0.0]))
    h_predicted = vm.heat_at(vm.world_to_idx([0.0, 0.0, 0.0]))
    # Both should have positive heat (current pos via base + tube,
    # future pos via tube only)
    assert h_now > 0
    assert h_predicted > 0


def test_dynamic_heat_capped_by_Hmax():
    """When heat_Hmax > 0, every cell's heat is at most that value."""
    vm = VoxelMap.from_bounds_2d((-2, 2), (-2, 2), res=0.25, z_plane=0.0)
    p = _heat_params(
        heat_alpha0=10.0, heat_alpha1=10.0, heat_Hmax=0.5,
    )
    obs = [_stationary_obstacle([0.0, 0.0, 0.0])]
    compute_dynamic_heat(vm, obs, p, t_now=0.0)
    assert float(vm.heat.max()) <= 0.5 + 1e-6


def test_dynamic_heat_empty_obstacles_returns_zeros():
    vm = VoxelMap.from_bounds_2d((-2, 2), (-2, 2), res=0.5, z_plane=0.0)
    heat = compute_dynamic_heat(vm, [], _heat_params(), t_now=0.0)
    assert float(heat.max()) == 0.0


def test_dynamic_heat_3d_localised_to_obstacle_region():
    """In a full 3D map, heat away from the obstacle stays zero."""
    vm = VoxelMap.from_bounds((-5, 5), (-5, 5), (-5, 5), res=0.5)
    obs = [_stationary_obstacle([0.0, 0.0, 0.0])]
    compute_dynamic_heat(vm, obs, _heat_params(), t_now=0.0)
    # Far cell (in 3D) is zero
    far = vm.world_to_idx([4.5, 4.5, 4.5])
    assert vm.heat_at(far) == pytest.approx(0.0, abs=1e-5)
    # Near cell is positive
    near = vm.world_to_idx([0.0, 0.0, 0.0])
    assert vm.heat_at(near) > 0.0


# ---------------------------------------------------------------------------
# Static heat
# ---------------------------------------------------------------------------

def test_static_heat_disabled_returns_zeros():
    vm = VoxelMap.from_bounds_2d((-2, 2), (-2, 2), res=0.5, z_plane=0.0)
    vm.mark_circle([0.0, 0.0, 0.0], 0.5)
    heat = compute_static_heat(vm, _heat_params())
    assert float(heat.max()) == 0.0


def test_static_heat_adds_halo_around_occupied():
    """With static heat enabled, free cells near occupied ones get a
    positive halo contribution."""
    vm = VoxelMap.from_bounds_2d((-2, 2), (-2, 2), res=0.25, z_plane=0.0)
    vm.mark_circle([0.0, 0.0, 0.0], 0.4)
    p = _heat_params(
        static_heat_enabled=True,
        static_heat_alpha=1.0,
        static_heat_p=2,
        static_heat_rmax_m=0.8,
    )
    compute_static_heat(vm, p)
    # A free cell just outside the obstacle should have heat > 0
    halo_cell = vm.world_to_idx([0.6, 0.0, 0.0])
    if not vm.is_occupied(halo_cell):
        assert vm.heat_at(halo_cell) > 0.0
    # A cell far from any obstacle should have zero halo
    far = vm.world_to_idx([1.9, 1.9, 0.0])
    assert vm.heat_at(far) == pytest.approx(0.0, abs=1e-5)
