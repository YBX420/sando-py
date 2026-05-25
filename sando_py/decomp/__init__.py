"""Convex decomposition — turn a polyline global path into a sequence of
overlapping safety corridors (``Polytope``s, ``Ax <= b``).

The C++ project uses DecompROS2 (ellipsoid + dilation). For the Python
port we start with **axis-aligned boxes** — simpler, still correct,
slightly more conservative — and can swap in something tighter later.

See ``decomp/README.md``.

Public API
----------
sfc(path, voxel_map, obstacles_t, params) -> List[Polytope]
    Top-level: one corridor per path segment, overlapping at the
    segment joins so the solver can hand off between them.

grow_aabb(p0, p1, voxel_map, params) -> Polytope
    The per-segment primitive: a 6-face axis-aligned box that contains
    the segment and is grown outward until it hits obstacles, world
    bounds, or the ``sfc_size`` cap.
"""

from .aabb import grow_aabb
from .ellipsoid import (
    Ellipsoid,
    EllipsoidDecompResult,
    Hyperplane,
    Polyhedron,
    decompose_path,
    decompose_segment,
    find_ellipsoid,
    find_polyhedron,
    polyhedron_to_polytope,
)
from .sfc import sfc, sfc_time_layered

__all__ = [
    "grow_aabb",
    "sfc",
    "sfc_time_layered",
    "decompose_segment",
    "decompose_path",
    "find_ellipsoid",
    "find_polyhedron",
    "polyhedron_to_polytope",
    "Ellipsoid",
    "EllipsoidDecompResult",
    "Hyperplane",
    "Polyhedron",
]
