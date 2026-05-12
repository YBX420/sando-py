"""Tests for sando_py.decomp — axis-aligned-box safety corridors."""

from __future__ import annotations

import numpy as np
import pytest

from sando_py.core import Parameters, Polytope
from sando_py.decomp import grow_aabb, sfc
from sando_py.hgp import VoxelMap


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _decomp_params(**overrides) -> Parameters:
    p = Parameters()
    p.drone_radius = 0.2
    p.sfc_size = [3.0, 3.0, 3.0]
    p.x_min, p.x_max = -50.0, 50.0
    p.y_min, p.y_max = -50.0, 50.0
    p.z_min, p.z_max = 0.5, 6.0
    p.use_shrinked_box = False
    p.shrinked_box_size = 0.0
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _empty_map(*, res: float = 0.25, z: float = 2.0) -> VoxelMap:
    return VoxelMap.from_bounds_2d((-20.0, 20.0), (-20.0, 20.0), res=res, z_plane=z)


def _polytope_bounds(poly: Polytope) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(box_lo, box_hi)`` from an axis-aligned-box ``Polytope``.

    Returns ``box_hi = b[:3]`` and ``box_lo = -b[3:6]`` — relies on the
    row ordering :func:`grow_aabb` documents.
    """
    assert poly.A.shape == (6, 3)
    assert poly.b.shape == (6, 1)
    assert np.allclose(poly.A[:3], np.eye(3))
    assert np.allclose(poly.A[3:], -np.eye(3))
    return -poly.b[3:, 0], poly.b[:3, 0]


def _contains(poly: Polytope, p) -> bool:
    p = np.asarray(p, dtype=float).reshape(3)
    return bool((poly.A @ p <= poly.b[:, 0] + 1e-9).all())


# ---------------------------------------------------------------------------
# grow_aabb — empty map
# ---------------------------------------------------------------------------

def test_grow_aabb_returns_6_face_polytope():
    vm = _empty_map()
    params = _decomp_params()
    poly = grow_aabb([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], vm, params)
    assert isinstance(poly, Polytope)
    assert poly.A.shape == (6, 3)
    assert poly.b.shape == (6, 1)


def test_grow_aabb_contains_segment_endpoints():
    vm = _empty_map()
    params = _decomp_params()
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([1.5, 0.5, 2.0])
    poly = grow_aabb(p0, p1, vm, params)
    assert _contains(poly, p0)
    assert _contains(poly, p1)
    assert _contains(poly, 0.5 * (p0 + p1))


def test_grow_aabb_capped_by_sfc_size_on_empty_map():
    """On an empty world the box should grow exactly to the sfc_size cap
    around the segment midpoint."""
    vm = _empty_map(res=0.25)
    params = _decomp_params(sfc_size=[1.5, 1.5, 1.0])
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([0.4, 0.0, 2.0])
    poly = grow_aabb(p0, p1, vm, params)
    box_lo, box_hi = _polytope_bounds(poly)
    midpoint = 0.5 * (p0 + p1)

    # x / y: limited by sfc_size from the midpoint (small step error
    # because we expand in voxel-sized steps).
    assert box_hi[0] == pytest.approx(midpoint[0] + 1.5, abs=0.3)
    assert box_lo[0] == pytest.approx(midpoint[0] - 1.5, abs=0.3)
    assert box_hi[1] == pytest.approx(midpoint[1] + 1.5, abs=0.3)
    assert box_lo[1] == pytest.approx(midpoint[1] - 1.5, abs=0.3)

    # z: capped by sfc_size cap, which is tighter than [z_min, z_max] here.
    assert box_hi[2] == pytest.approx(midpoint[2] + 1.0, abs=0.3)
    assert box_lo[2] == pytest.approx(midpoint[2] - 1.0, abs=0.3)


def test_grow_aabb_z_clamped_to_world_bounds():
    """When sfc_size is generous, ±z should stop at z_min / z_max."""
    vm = _empty_map()
    params = _decomp_params(sfc_size=[10.0, 10.0, 10.0], z_min=1.5, z_max=2.5)
    poly = grow_aabb([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], vm, params)
    box_lo, box_hi = _polytope_bounds(poly)
    assert box_lo[2] == pytest.approx(1.5, abs=1e-6)
    assert box_hi[2] == pytest.approx(2.5, abs=1e-6)


# ---------------------------------------------------------------------------
# grow_aabb — obstacles in the map
# ---------------------------------------------------------------------------

def test_grow_aabb_halts_at_obstacle_in_plus_y():
    """Wall above the segment should stop +y growth short of sfc_size."""
    vm = _empty_map(res=0.25)
    # Wall along y = 1.0, from x = -5 to x = 5
    for x in np.arange(-5.0, 5.01, 0.25):
        vm.mark_circle([float(x), 1.0, 2.0], radius=0.25)
    params = _decomp_params(sfc_size=[5.0, 5.0, 5.0])
    poly = grow_aabb([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], vm, params)
    box_lo, box_hi = _polytope_bounds(poly)
    # +y face should stop well short of the wall (definitely < 1.0)
    assert box_hi[1] < 1.0
    # -y face is unblocked, so it should reach the sfc_size cap (~-5).
    assert box_lo[1] < -2.0


def test_grow_aabb_halts_at_obstacle_in_minus_x():
    """A wall behind the segment should pin the -x face."""
    vm = _empty_map(res=0.25)
    for y in np.arange(-2.0, 2.01, 0.25):
        vm.mark_circle([-1.0, float(y), 2.0], radius=0.25)
    params = _decomp_params(sfc_size=[5.0, 5.0, 5.0])
    poly = grow_aabb([0.0, 0.0, 2.0], [2.0, 0.0, 2.0], vm, params)
    box_lo, _ = _polytope_bounds(poly)
    assert box_lo[0] > -1.0
    # And it should be roughly at the wall, not way ahead of it.
    assert box_lo[0] > -1.0 and box_lo[0] < -0.5


# ---------------------------------------------------------------------------
# Top-level sfc walker
# ---------------------------------------------------------------------------

def test_sfc_returns_one_poly_per_segment():
    vm = _empty_map()
    params = _decomp_params()
    path = [
        np.array([0.0, 0.0, 2.0]),
        np.array([2.0, 0.0, 2.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([4.0, 2.0, 2.0]),
    ]
    polys = sfc(path, vm, [], params)
    assert len(polys) == len(path) - 1
    for poly, p0, p1 in zip(polys, path, path[1:]):
        assert _contains(poly, p0)
        assert _contains(poly, p1)


def test_sfc_consecutive_polys_overlap_at_join():
    """The join waypoint must be inside both adjacent corridors so the
    solver has a feasible hand-off point."""
    vm = _empty_map()
    params = _decomp_params()
    path = [
        np.array([0.0, 0.0, 2.0]),
        np.array([2.0, 0.0, 2.0]),
        np.array([2.0, 2.0, 2.0]),
    ]
    polys = sfc(path, vm, [], params)
    join = path[1]
    assert _contains(polys[0], join)
    assert _contains(polys[1], join)


def test_sfc_empty_or_singleton_path_returns_empty_list():
    vm = _empty_map()
    params = _decomp_params()
    assert sfc([], vm, [], params) == []
    assert sfc([np.zeros(3)], vm, [], params) == []


# ---------------------------------------------------------------------------
# Shrink option
# ---------------------------------------------------------------------------

def test_use_shrinked_box_shrinks_but_keeps_segment_inside():
    """With ``use_shrinked_box=True`` the final box is tighter, but the
    segment endpoints must still be in it (mirrors C++ behaviour)."""
    vm = _empty_map()
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([0.5, 0.0, 2.0])
    base = _polytope_bounds(grow_aabb(p0, p1, vm, _decomp_params()))
    shrunk = _polytope_bounds(grow_aabb(
        p0, p1, vm, _decomp_params(use_shrinked_box=True, shrinked_box_size=0.4)
    ))
    # Shrunk x-range narrower (or equal) than base
    base_extent = base[1] - base[0]
    shrunk_extent = shrunk[1] - shrunk[0]
    assert (shrunk_extent <= base_extent + 1e-6).all()
    # Endpoints still feasible
    poly_shrunk = grow_aabb(
        p0, p1, vm, _decomp_params(use_shrinked_box=True, shrinked_box_size=0.4)
    )
    assert _contains(poly_shrunk, p0)
    assert _contains(poly_shrunk, p1)
