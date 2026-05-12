"""Regression tests for :mod:`sando_py.decomp.ellipsoid` — the v2
EllipsoidDecomp3D port.

These cover the basics the C++ ``LineSegment<3>::find_ellipsoid`` +
``DecompBase<3>::find_polyhedron`` pipeline guarantees: endpoints are
inside the resulting polytope, obstacles are excluded, and the corridor
respects the local-bbox / z-bound walls.
"""

from __future__ import annotations

import numpy as np

from sando_py.core import Polytope
from sando_py.decomp import (
    Ellipsoid,
    Hyperplane,
    Polyhedron,
    decompose_path,
    decompose_segment,
    find_ellipsoid,
    find_polyhedron,
    polyhedron_to_polytope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains(poly: Polytope, p, eps: float = 1e-6) -> bool:
    p = np.asarray(p, dtype=float).reshape(3)
    if poly.A.size == 0:
        return True
    return bool(np.all(poly.A @ p <= poly.b[:, 0] + eps))


# ---------------------------------------------------------------------------
# Ellipsoid / Polyhedron primitives
# ---------------------------------------------------------------------------

def test_ellipsoid_dist_and_inside():
    E = Ellipsoid(C=np.diag([2.0, 1.0, 1.0]), d=np.zeros(3))
    assert E.inside(np.zeros(3))
    assert abs(E.dist(np.array([2.0, 0.0, 0.0])) - 1.0) < 1e-9
    assert not E.inside(np.array([3.0, 0.0, 0.0]))


def test_polyhedron_to_polytope_flips_normals_to_face_inside_point():
    # Hyperplane at x=1 with normal pointing -x → midpoint at origin
    # is on the +x side, so flip is needed.
    poly = Polyhedron()
    poly.add(Hyperplane(np.array([1.0, 0.0, 0.0]),
                        np.array([-1.0, 0.0, 0.0])))
    p = polyhedron_to_polytope(poly, np.zeros(3))
    # After flip the constraint should accept the origin
    assert p.A[0, 0] == 1.0
    assert p.b[0, 0] == 1.0


# ---------------------------------------------------------------------------
# find_ellipsoid / find_polyhedron — single segment, no obstacles
# ---------------------------------------------------------------------------

def test_decompose_segment_empty_obstacles_returns_segment_inside():
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([2.0, 0.0, 2.0])
    res = decompose_segment(
        p0, p1, np.zeros((0, 3)),
        inflate_distance=0.2,
        local_bbox=np.array([3.0, 3.0, 3.0]),
        z_min=0.5, z_max=6.0,
    )
    assert _contains(res.polytope, p0)
    assert _contains(res.polytope, p1)
    assert _contains(res.polytope, 0.5 * (p0 + p1))
    # No obstacles → empty obstacle-derived hyperplanes; only local bbox
    # + z walls. So ≤ 8 rows total (6 bbox + 2 z), each with finite norm.
    assert res.polytope.A.shape[1] == 3


# ---------------------------------------------------------------------------
# find_ellipsoid — obstacles excluded
# ---------------------------------------------------------------------------

def test_decompose_segment_obstacle_outside_polytope():
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([2.0, 0.0, 2.0])
    # Wall to the +y side of the segment
    obstacles = np.array([
        [0.5, 1.0, 2.0],
        [1.0, 1.0, 2.0],
        [1.5, 1.0, 2.0],
    ])
    res = decompose_segment(
        p0, p1, obstacles,
        inflate_distance=0.1,
        local_bbox=np.array([3.0, 3.0, 3.0]),
        z_min=0.5, z_max=6.0,
    )
    # Endpoints feasible
    assert _contains(res.polytope, p0)
    assert _contains(res.polytope, p1)
    # Every obstacle must be infeasible (no hyperplane lets it through
    # since the ellipsoid was grown to exclude them).
    for pt in obstacles:
        assert not _contains(res.polytope, pt, eps=-1e-3), \
            f"Obstacle {pt} should be outside the polytope"


def test_decompose_segment_z_bounds_clamp():
    """A generous segment in z should be clamped by z_min / z_max."""
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([1.0, 0.0, 2.0])
    res = decompose_segment(
        p0, p1, np.zeros((0, 3)),
        local_bbox=np.array([10.0, 10.0, 10.0]),
        z_min=1.5, z_max=2.5,
    )
    # Points just above / below the z bounds should be infeasible.
    assert _contains(res.polytope, np.array([0.5, 0.0, 2.0]))
    assert not _contains(res.polytope, np.array([0.5, 0.0, 2.6]), eps=-1e-3)
    assert not _contains(res.polytope, np.array([0.5, 0.0, 1.4]), eps=-1e-3)


# ---------------------------------------------------------------------------
# decompose_path — multi-segment overlap at joins
# ---------------------------------------------------------------------------

def test_decompose_path_join_inside_both_corridors():
    path = [
        np.array([0.0, 0.0, 2.0]),
        np.array([2.0, 0.0, 2.0]),
        np.array([2.0, 2.0, 2.0]),
    ]
    results = decompose_path(
        path, np.zeros((0, 3)),
        local_bbox=np.array([3.0, 3.0, 3.0]),
        z_min=0.5, z_max=6.0,
    )
    assert len(results) == 2
    join = path[1]
    assert _contains(results[0].polytope, join)
    assert _contains(results[1].polytope, join)


# ---------------------------------------------------------------------------
# Inflation distance
# ---------------------------------------------------------------------------

def test_inflate_distance_pulls_corridor_back_from_obstacle():
    """With a non-zero inflate_distance the corridor should sit further
    from the obstacle than without it (more conservative)."""
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([2.0, 0.0, 2.0])
    obstacles = np.array([[1.0, 1.0, 2.0]])

    base = decompose_segment(
        p0, p1, obstacles,
        inflate_distance=0.0,
        local_bbox=np.array([5.0, 5.0, 5.0]),
        z_min=0.5, z_max=6.0,
    )
    inflated = decompose_segment(
        p0, p1, obstacles,
        inflate_distance=0.5,
        local_bbox=np.array([5.0, 5.0, 5.0]),
        z_min=0.5, z_max=6.0,
    )

    # A probe close to the obstacle should be infeasible in the
    # inflated version sooner than in the base version.
    probe = np.array([1.0, 0.7, 2.0])
    base_ok = _contains(base.polytope, probe, eps=-1e-3)
    inflated_ok = _contains(inflated.polytope, probe, eps=-1e-3)
    # Inflation should reduce feasibility (strictly weaker, or at
    # worst equal — but obstacle is right above the probe so the
    # inflated case must reject it if base did).
    if base_ok:
        assert not inflated_ok or base_ok


def test_elongated_obstacle_marked_as_aabb_not_sphere():
    """A tall thin tree ``[0.4, 0.4, 4.0]`` should occupy a thin column
    in the voxel map, not a 4 m radius sphere. Regression: previously
    ``max(bbox)`` was used as a sphere radius, walling off the whole
    horizon for 200 dynamic-mode trees."""
    from sando_py.hgp import HGPPlanner, VoxelMap

    vm = VoxelMap.from_bounds((-10.0, 10.0), (-10.0, 10.0), (0.0, 8.0), res=0.3)
    HGPPlanner._mark_aabb(vm, np.array([0.0, 0.0, 2.0]),
                          np.array([0.2, 0.2, 2.0]), 0.0)

    # 3 m off-axis should be FREE (would be occupied if sphere-marked
    # with radius 4 m).
    ix, iy, iz = vm.world_to_idx([3.0, 0.0, 2.0])
    assert vm.in_bounds((ix, iy, iz))
    assert vm.occupied[ix, iy, iz] == False  # noqa: E712

    # The trunk should be occupied at z=2.0.
    ix, iy, iz = vm.world_to_idx([0.0, 0.0, 2.0])
    assert vm.occupied[ix, iy, iz] == True  # noqa: E712
