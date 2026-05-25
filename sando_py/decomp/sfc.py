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


def _voxel_map_occupied_points(vm: VoxelMap, *,
                               z_floor_thresh: float = -np.inf) -> np.ndarray:
    """Return ``(N, 3)`` world-coordinate cell centers of all occupied
    voxels in ``vm``. Mirrors the C++ "obstacle vector" passed to
    EllipsoidDecomp (``vec_uo_`` / ``vec_o_`` in hgp_manager.cpp).

    ``z_floor_thresh`` (default ``-inf`` → no filter): drops voxels with
    ``z <= z_floor_thresh``. Matches C++ sando.cpp:935-950 which strips
    ``z <= z_min + res`` from ``base_uo`` before decomposition — the
    floor is already enforced by the solver's z_min constraint and the
    ellipsoid decomp's ``add_z_min_and_max``. Including floor voxels in
    the obstacle set causes the ellipsoid to shrink in all axes
    (ellipsoidal coupling), producing unnecessarily skinny corridors.

    Cached on the VoxelMap by its ``_occ_version`` counter so repeat
    calls in the same replan tick (decomp + viz) reuse the result. The
    cache key includes ``z_floor_thresh`` so different callers (the
    SFC pass which filters vs the viz publish which doesn't) don't
    clobber each other's cache.
    """
    if vm is None or vm.occupied.size == 0:
        return np.zeros((0, 3))
    # Cache hit? Key on (version, floor threshold) so the unfiltered
    # viz call and the filtered decomp call don't fight over the cache.
    cache_key = (vm._occ_version, float(z_floor_thresh))
    cached = getattr(vm, "_cached_obs_pts", None)
    cached_key = getattr(vm, "_cached_obs_pts_key", None)
    if cached is not None and cached_key == cache_key:
        return cached
    ix, iy, iz = np.nonzero(vm.occupied)
    if ix.size == 0:
        pts = np.zeros((0, 3))
    else:
        xs = vm.x_min + (ix + 0.5) * vm.res
        ys = vm.y_min + (iy + 0.5) * vm.res
        zs = vm.z_min + (iz + 0.5) * vm.res
        pts = np.stack([xs, ys, zs], axis=-1)
        if z_floor_thresh > -np.inf:
            keep = pts[:, 2] > z_floor_thresh
            pts = pts[keep]
    vm._cached_obs_pts = pts
    vm._cached_obs_pts_key = cache_key
    return pts


def _dynamic_obstacle_points(
    obstacles: Iterable[DynTraj],
    params: Parameters,
    t_now: float,
) -> np.ndarray:
    """C++ ``HGPManager::obstacle_to_vec`` dynamic-obstacle half.

    Each dynamic obstacle contributes every lattice point inside its
    bbox inflated by ``obst_max_vel * horizon + obst_position_error``.
    The lattice resolution is ``factor_hgp * res``.
    """
    obstacles = [o for o in obstacles] if obstacles else []
    if not obstacles:
        return np.zeros((0, 3))

    horizon = max(0.0, float(params.horizon))
    halo_r = (
        float(params.obst_max_vel) * horizon
        + float(params.obst_position_error)
    )
    step = max(1e-6, float(params.factor_hgp or 1.0) * float(params.res or 0.3))

    pts_list: List[np.ndarray] = []
    for tr in obstacles:
        c = np.asarray(tr.current_pos, dtype=float).reshape(3)
        bbox = np.asarray(tr.bbox, dtype=float).reshape(-1)
        if bbox.size < 3:
            bbox = np.array([
                float(bbox[0]) if bbox.size > 0 else 0.4,
                float(bbox[1]) if bbox.size > 1 else 0.4,
                float(bbox[2]) if bbox.size > 2 else 0.4,
            ])
        half = bbox[:3] + halo_r
        pts_list.append(_sample_aabb_volume(c, half, step))
    if not pts_list:
        return np.zeros((0, 3))
    return np.concatenate(pts_list, axis=0)


def _sample_aabb_volume(center: np.ndarray, half: np.ndarray,
                        step: float) -> np.ndarray:
    """Return lattice points filling the inflated AABB volume."""
    center = np.asarray(center, dtype=float).reshape(3)
    half = np.asarray(half, dtype=float).reshape(3)
    step = max(1e-6, float(step))

    def _axis(extent: float) -> np.ndarray:
        m = int(np.ceil(float(extent) / step))
        return np.arange(-m, m + 1, dtype=float) * step

    gx, gy, gz = _axis(half[0]), _axis(half[1]), _axis(half[2])
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    offsets = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    keep = (
        (np.abs(offsets[:, 0]) <= half[0] + 1e-9)
        & (np.abs(offsets[:, 1]) <= half[1] + 1e-9)
        & (np.abs(offsets[:, 2]) <= half[2] + 1e-9)
    )
    return offsets[keep] + center


def _dynamic_obstacle_points_with_halo(
    obstacles: Iterable[DynTraj], halo_r: float, step: float,
) -> np.ndarray:
    """Same as :func:`_dynamic_obstacle_points` but with an explicit
    ``halo_r`` (instead of pulling it from params). Used by
    :func:`sfc_time_layered` so each time layer can inflate by a
    different radius without rebuilding params."""
    obstacles = [o for o in obstacles] if obstacles else []
    if not obstacles:
        return np.zeros((0, 3))
    pts_list: List[np.ndarray] = []
    for tr in obstacles:
        c = np.asarray(tr.current_pos, dtype=float).reshape(3)
        bbox = np.asarray(tr.bbox, dtype=float).reshape(-1)
        if bbox.size < 3:
            bbox = np.array([
                float(bbox[0]) if bbox.size > 0 else 0.4,
                float(bbox[1]) if bbox.size > 1 else 0.4,
                float(bbox[2]) if bbox.size > 2 else 0.4,
            ])
        half = bbox[:3] + halo_r
        pts_list.append(_sample_aabb_volume(c, half, step))
    return np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 3))


def sfc_time_layered(
    path: List[np.ndarray],
    voxel_map: VoxelMap,
    obstacles_t: Iterable[DynTraj],
    params: Parameters,
    *,
    num_time_layers: int,
    dt_layer: float,
    t_now: float = 0.0,
) -> List[List[Polytope]]:
    """Time-layered convex decomposition — port of C++
    ``HGPManager::cvxEllipsoidDecompTimeLayered``
    (hgp_manager.cpp:464-558) + ``SANDO::generateLocalTrajectory``
    timing setup (sando.cpp:1334-1376).

    For each time layer ``n ∈ [0, num_time_layers)``, the dynamic
    obstacles are inflated by ``r_n = obst_max_vel * t_n + obst_position_error``
    where ``t_n = (n + 1) * dt_layer`` is the end time of solver segment
    ``n``. The static obstacle set (voxel-map occupied points, with the
    floor filter) stays the same across layers. For each path segment
    ``p ∈ [0, len(path) - 1)``, an ellipsoid is dilated against the
    layer-``n`` obstacle set and converted to a polytope.

    Returns a list-of-lists ``polys[n][p]`` so the downstream solver
    can index ``polyAt_(t, p) = polys[t_layer(t)][p]`` (matches the C++
    ``polytopes_time_layered_`` flat-vector indexing rule).
    """
    if path is None or len(path) < 2 or num_time_layers <= 0:
        return []

    # Static base obstacle set (same for every layer)
    z_floor = -np.inf
    if params.z_min < params.z_max:
        z_floor = float(params.z_min) + float(params.res)
    obs_static = _voxel_map_occupied_points(voxel_map, z_floor_thresh=z_floor)

    # Local bbox + z bounds + drone inflation are layer-independent
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
    step = max(1e-6, float(params.factor_hgp or 1.0) * float(params.res or 0.3))

    out: List[List[Polytope]] = []
    obstacles_list = list(obstacles_t) if obstacles_t else []
    for n in range(int(num_time_layers)):
        t_n = (n + 1) * float(dt_layer)
        halo_r = float(params.obst_max_vel) * t_n + float(params.obst_position_error)
        obs_dynamic_n = _dynamic_obstacle_points_with_halo(
            obstacles_list, halo_r, step,
        )
        if obs_static.size == 0 and obs_dynamic_n.size == 0:
            obs_n = np.zeros((0, 3))
        elif obs_static.size == 0:
            obs_n = obs_dynamic_n
        elif obs_dynamic_n.size == 0:
            obs_n = obs_static
        else:
            obs_n = np.concatenate([obs_static, obs_dynamic_n], axis=0)

        layer_polys: List[Polytope] = []
        for i in range(len(path) - 1):
            res = decompose_segment(
                path[i], path[i + 1], obs_n,
                offset_x=0.0,
                inflate_distance=inflate,
                local_bbox=local_bbox,
                z_min=z_min, z_max=z_max,
            )
            layer_polys.append(res.polytope)

        # Optional shrink (mirrors aabb / single-layer behaviour)
        if params.use_shrinked_box and params.shrinked_box_size > 0:
            s = float(params.shrinked_box_size)
            shrunk: List[Polytope] = []
            for p in layer_polys:
                A = p.A
                b = p.b.copy()
                for i in range(A.shape[0]):
                    nrow = float(np.linalg.norm(A[i]))
                    if nrow > 1e-9:
                        b[i, 0] -= s * nrow
                shrunk.append(Polytope(A=A, b=b))
            layer_polys = shrunk
        out.append(layer_polys)
    return out


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
    # Floor voxel filter (C++ sando.cpp:935-950): drop occupied voxels
    # whose centers fall at ``z <= z_min + res``. Including the floor
    # would couple the ellipsoid's z-shrink to its x/y-shrink and force
    # unnecessarily skinny corridors at low altitude.
    z_floor = -np.inf
    if params.z_min < params.z_max:
        z_floor = float(params.z_min) + float(params.res)
    obs_static = _voxel_map_occupied_points(voxel_map, z_floor_thresh=z_floor)
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
