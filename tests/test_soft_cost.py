"""Regression test for ``use_soft_cost_obstacles`` behaviour in the
heat-map overlay (mirrors C++ map_util.hpp:488 + :654 gating)."""

from __future__ import annotations

import numpy as np

from sando_py.core import Parameters
from sando_py.hgp import VoxelMap
from sando_py.hgp.heat_map import compute_static_heat


def _params() -> Parameters:
    p = Parameters()
    p.static_heat_enabled = True
    p.static_heat_alpha = 1.0
    p.static_heat_p = 2
    p.static_heat_Hmax = 5.0
    p.static_heat_default_radius_m = 0.5
    p.static_heat_rmax_m = 1.0
    return p


def _map_with_block() -> VoxelMap:
    vm = VoxelMap.from_bounds_2d((-2.0, 2.0), (-2.0, 2.0), res=0.1, z_plane=2.0)
    # Single occupied cell at origin
    ix, iy, iz = vm.world_to_idx([0.0, 0.0, 2.0])
    vm.occupied[ix, iy, iz] = True
    return vm


def test_soft_cost_off_skips_occupied_cells():
    """Without soft cost, occupied cells receive zero heat — they're
    hard barriers."""
    vm = _map_with_block()
    p = _params()
    p.use_soft_cost_obstacles = False
    heat = compute_static_heat(vm, p, boundary_only=False)
    ix, iy, iz = vm.world_to_idx([0.0, 0.0, 2.0])
    assert heat[ix, iy, iz] == 0.0


def test_soft_cost_on_writes_alpha_at_occupied_cells():
    vm = _map_with_block()
    p = _params()
    p.use_soft_cost_obstacles = True
    heat = compute_static_heat(vm, p, boundary_only=False)
    ix, iy, iz = vm.world_to_idx([0.0, 0.0, 2.0])
    assert heat[ix, iy, iz] >= float(p.static_heat_alpha) - 1e-6


def test_static_heat_radius_clamped_to_default():
    """When default_radius < rmax_m, effective halo uses default."""
    vm = VoxelMap.from_bounds_2d((-3.0, 3.0), (-3.0, 3.0), res=0.1, z_plane=2.0)
    ix, iy, iz = vm.world_to_idx([0.0, 0.0, 2.0])
    vm.occupied[ix, iy, iz] = True
    p = _params()
    p.static_heat_default_radius_m = 0.3  # smaller than rmax_m=1.0
    p.static_heat_rmax_m = 1.0
    heat = compute_static_heat(vm, p, boundary_only=False)
    # Cell at 0.4 m from the obstacle should be OUTSIDE the halo
    # (effective radius = 0.3, so 0.4 has 0 heat).
    ix2, iy2, iz2 = vm.world_to_idx([0.4, 0.0, 2.0])
    if vm.in_bounds((ix2, iy2, iz2)):
        assert heat[ix2, iy2, iz2] == 0.0
    # Cell at 0.15 m should still be inside.
    ix3, iy3, iz3 = vm.world_to_idx([0.15, 0.0, 2.0])
    if vm.in_bounds((ix3, iy3, iz3)):
        assert heat[ix3, iy3, iz3] > 0.0
