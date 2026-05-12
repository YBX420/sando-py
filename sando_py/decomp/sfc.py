"""Top-level safety-corridor generator.

``sfc(path, voxel_map, obstacles_t, params)`` walks the polyline output
of the global planner and produces one corridor per segment.

Two modes, selected by ``params.environment_assumption``:

* **Default (``"static"`` / ``"dynamic_worst_case"`` / ``"dynamic"`` / etc.
  / empty)** — uses the v2 ellipsoid+dilation decomposition
  (:mod:`sando_py.decomp.ellipsoid`), a Python port of DecompROS2's
  ``EllipsoidDecomp3D``. This is the C++-aligned path.
* **``"aabb"`` (Python-only override)** — falls back to the v1 axis-
  aligned grow-from-segment in :mod:`sando_py.decomp.aabb`. Useful for
  benchmarking the corridor upgrade and for setups without a populated
  voxel-map (the v1 doesn't need an obstacle point cloud).

Adjacent corridors share their endpoint, so they overlap at the segment
joins — the local solver uses that overlap to hand off between pieces.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from sando_py.core import DynTraj, Parameters, Polytope
from sando_py.hgp import VoxelMap

from .aabb import grow_aabb
from .ellipsoid import decompose_segment


def _voxel_map_occupied_points(vm: VoxelMap) -> np.ndarray:
    """Return ``(N, 3)`` world-coordinate cell centers of all occupied
    voxels in ``vm``. Mirrors the C++ "obstacle vector" passed to
    EllipsoidDecomp (``vec_uo_`` / ``vec_o_`` in hgp_manager.cpp).

    Cached on the VoxelMap by its ``_occ_version`` counter so repeat
    calls in the same replan tick (decomp + viz) reuse the result. The
    full ``np.nonzero`` scan is ~O(grid size); for a 100×40×20 map at
    0.3 m resolution that's 80k voxels — non-trivial when run 10 Hz.
    """
    if vm is None or vm.occupied.size == 0:
        return np.zeros((0, 3))
    # Cache hit?
    cached_pts = getattr(vm, "_cached_obs_pts", None)
    cached_ver = getattr(vm, "_cached_obs_pts_version", -1)
    if cached_pts is not None and cached_ver == vm._occ_version:
        return cached_pts
    ix, iy, iz = np.nonzero(vm.occupied)
    if ix.size == 0:
        pts = np.zeros((0, 3))
    else:
        xs = vm.x_min + (ix + 0.5) * vm.res
        ys = vm.y_min + (iy + 0.5) * vm.res
        zs = vm.z_min + (iz + 0.5) * vm.res
        pts = np.stack([xs, ys, zs], axis=-1)
    vm._cached_obs_pts = pts
    vm._cached_obs_pts_version = vm._occ_version
    return pts


def _dynamic_obstacle_points(
    obstacles: Iterable[DynTraj],
    params: Parameters,
    t_now: float,
) -> np.ndarray:
    """Sample each dynamic obstacle's *axis-aligned bbox corners*
    (inflated by ``obst_max_vel*horizon + obst_position_error``) so
    EllipsoidDecomp sees its real extent, not a sphere of radius
    ``max(bbox)``. Without this, elongated obstacles like trees
    ``[0.4, 0.4, 4.0]`` or horizontal poles ``[0.4, 8.0, 0.4]`` get
    represented as a single center point + 6 axial halos and the
    resulting corridor doesn't carve around them properly.

    The halo radius (``obst_max_vel * horizon + obst_position_error``)
    is the same conservative envelope used for the dynamic-tube heat
    overlay (hgp_manager.cpp:630)."""
    obstacles = [o for o in obstacles] if obstacles else []
    if not obstacles:
        return np.zeros((0, 3))

    horizon = max(0.0, float(params.horizon))
    halo_r = (
        float(params.obst_max_vel) * horizon
        + float(params.obst_position_error)
    )
    pts: List[np.ndarray] = []
    # 8 corner offsets ±1 along each axis
    corner_signs = np.array([
        [+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1],
        [-1, +1, +1], [-1, +1, -1], [-1, -1, +1], [-1, -1, -1],
    ], dtype=float)
    for tr in obstacles:
        c = np.asarray(tr.current_pos, dtype=float).reshape(3)
        bbox = np.asarray(tr.bbox, dtype=float).reshape(-1)
        if bbox.size < 3:
            bbox = np.array([
                float(bbox[0]) if bbox.size > 0 else 0.4,
                float(bbox[1]) if bbox.size > 1 else 0.4,
                float(bbox[2]) if bbox.size > 2 else 0.4,
            ])
        half = bbox[:3] + halo_r  # uniform-radius inflation around AABB
        pts.append(c)
        for sgn in corner_signs:
            pts.append(c + sgn * half)
    return np.stack(pts, axis=0)


def sfc(
    path: List[np.ndarray],
    voxel_map: VoxelMap,
    obstacles_t: Iterable[DynTraj],
    params: Parameters,
    *,
    t_now: float = 0.0,
) -> List[Polytope]:
    """Return one ``Polytope`` per segment in ``path``."""
    if path is None or len(path) < 2:
        return []

    mode = (params.environment_assumption or "").lower()
    if mode == "aabb":
        return [
            grow_aabb(path[i], path[i + 1], voxel_map, params)
            for i in range(len(path) - 1)
        ]

    # ----- v2 EllipsoidDecomp path -----
    obs_static = _voxel_map_occupied_points(voxel_map)
    obs_dynamic = _dynamic_obstacle_points(obstacles_t, params, t_now)
    if obs_static.size == 0 and obs_dynamic.size == 0:
        obstacles = np.zeros((0, 3))
    elif obs_static.size == 0:
        obstacles = obs_dynamic
    elif obs_dynamic.size == 0:
        obstacles = obs_static
    else:
        obstacles = np.concatenate([obs_static, obs_dynamic], axis=0)

    # ``sfc_size`` is the local bbox half-widths along (x=segment,
    # y=horizontal-perp, z=vertical-perp). Falls back to a generous
    # default so we never produce an empty corridor when the YAML omits
    # the key.
    if params.sfc_size:
        lb = list(params.sfc_size)
        while len(lb) < 3:
            lb.append(lb[-1] if lb else 10.0)
        local_bbox = np.asarray(lb[:3], dtype=float)
    else:
        local_bbox = np.array([10.0, 10.0, 10.0], dtype=float)

    z_min = float(params.z_min) if params.z_min != params.z_max else -np.inf
    z_max = float(params.z_max) if params.z_min != params.z_max else np.inf
    inflate = float(params.drone_radius)

    polys: List[Polytope] = []
    for i in range(len(path) - 1):
        res = decompose_segment(
            path[i], path[i + 1], obstacles,
            offset_x=0.0,
            inflate_distance=inflate,
            local_bbox=local_bbox,
            z_min=z_min, z_max=z_max,
        )
        polys.append(res.polytope)

    # Optional uniform shrink (mirrors C++ ``use_shrinked_box`` /
    # ``shrinked_box_size``: each face is moved inward by the same
    # amount). The shift on b is just -shrink_dist for unit normals.
    if params.use_shrinked_box and params.shrinked_box_size > 0:
        s = float(params.shrinked_box_size)
        shrunk: List[Polytope] = []
        for p in polys:
            # Re-normalize each row of A; only adjust b if the row norm
            # is finite (degenerate rows are skipped).
            A = p.A
            b = p.b.copy()
            for i in range(A.shape[0]):
                nrow = float(np.linalg.norm(A[i]))
                if nrow > 1e-9:
                    b[i, 0] -= s * nrow
            shrunk.append(Polytope(A=A, b=b))
        polys = shrunk

    return polys
