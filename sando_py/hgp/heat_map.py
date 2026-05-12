"""Dynamic + static heat map overlay for the :class:`VoxelMap`.

Mirrors the C++ ``map_util.hpp`` ``updateHeat`` flow at the level we
need for v2 trajectory bias:

* For each dynamic obstacle, sample its trajectory at
  ``heat_num_samples`` times in ``[t_now, t_now + horizon]``.
* For each sample, compute box-distance from every voxel in the
  obstacle's reachable AABB to the obstacle's predicted bounding box.
* Heat contribution = ``heat_alpha1 * exp(-t/tau) * (1 - d/R)^heat_q``
  inside a growing tube ``R = dyn_heat_tube_radius_m + heat_gamma * t``.
* Base term (reachable-radius halo at ``t_now``):
  ``heat_alpha0 * (1 - d/R_reach)^heat_p``.
* Per cell, heat is the *max* across obstacles, capped at
  ``heat_Hmax``.

Static heat (halo around occupied cells) is included as an additional
``static_heat_alpha`` term over the reachable distance from any
occupied cell, with radial falloff power ``static_heat_p``.

All work is vectorised per obstacle / per time-sample using numpy
broadcasting over the local AABB — never the full grid — so the cost
stays linear in (number of obstacles × samples × cells-near-each-obstacle).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from sando_py.core import DynTraj, Parameters

from .voxel_map import VoxelMap


def compute_dynamic_heat(
    vm: VoxelMap,
    obstacles: Iterable[DynTraj],
    params: Parameters,
    t_now: float,
) -> np.ndarray:
    """Populate ``vm.heat`` (creating it if absent) with the dynamic
    heat field for the given obstacles at ``t_now`` and return it.

    Returns the heat array so callers can introspect / reuse it; the
    same array is stored on ``vm``.
    """
    heat = vm.ensure_heat()
    # Always start from a clean overlay — heat depends on the current
    # obstacle snapshot and the v_max-derived reachable horizon.
    heat[...] = 0.0

    obstacles = list(obstacles)
    if not obstacles:
        return heat

    alpha0 = float(params.heat_alpha0 or 0.0)
    alpha1 = float(params.heat_alpha1 or 0.0)
    heat_p = int(params.heat_p or 2)
    heat_q = int(params.heat_q or 2)
    heat_gamma = float(params.heat_gamma or 0.0)
    heat_Hmax = float(params.heat_Hmax or 0.0)
    tau_w = max(1e-6, float(params.heat_tau_ratio or 0.5) * float(params.horizon or 1.0))
    R0 = max(0.0, float(params.dyn_heat_tube_radius_m or 0.0))
    num_samples = max(1, int(params.heat_num_samples or 5))
    obst_max_vel = max(0.0, float(params.obst_max_vel or 0.0))
    horizon = max(1e-3, float(params.horizon or 1.0))

    # Time samples in [0, horizon]
    t_samples = np.linspace(0.0, horizon, num_samples)

    # Pre-compute world coords once per axis (used for every obstacle)
    xs = vm.x_min + (np.arange(vm.nx) + 0.5) * vm.res
    ys = vm.y_min + (np.arange(vm.ny) + 0.5) * vm.res
    zs = vm.z_min + (np.arange(vm.nz) + 0.5) * vm.res

    for obs in obstacles:
        bbox = np.asarray(obs.bbox, dtype=float).reshape(-1)
        if bbox.size < 3:
            bbox = np.array([0.4, 0.4, 0.4])
        hx, hy, hz = float(bbox[0]), float(bbox[1]), float(bbox[2])
        max_extent = max(hx, hy, hz)
        R_reach = max_extent + obst_max_vel * horizon

        # Predicted centres at each time sample
        centres = np.empty((num_samples, 3), dtype=float)
        for j, t_off in enumerate(t_samples):
            try:
                centres[j] = obs.eval(t_now + float(t_off))
            except Exception:  # noqa: BLE001
                centres[j] = obs.current_pos
        ck = centres[0]  # base centre (at t_now)

        # Tube radii / time-decay weights
        Rj = R0 + heat_gamma * t_samples
        Wj = np.exp(-t_samples / tau_w)

        # Compute the local AABB that any contribution could reach
        # (max growing radius + abs-pos extent across samples + bbox).
        rmax = float(max(R_reach, np.max(Rj) if Rj.size else 0.0))
        margin = rmax + max_extent
        lo = np.min(centres, axis=0) - margin
        hi = np.max(centres, axis=0) + margin
        ix_lo = max(0, int(np.floor((lo[0] - vm.x_min) / vm.res)))
        ix_hi = min(vm.nx, int(np.ceil((hi[0] - vm.x_min) / vm.res)) + 1)
        iy_lo = max(0, int(np.floor((lo[1] - vm.y_min) / vm.res)))
        iy_hi = min(vm.ny, int(np.ceil((hi[1] - vm.y_min) / vm.res)) + 1)
        iz_lo = max(0, int(np.floor((lo[2] - vm.z_min) / vm.res)))
        iz_hi = min(vm.nz, int(np.ceil((hi[2] - vm.z_min) / vm.res)) + 1)
        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            continue

        sub_x = xs[ix_lo:ix_hi]
        sub_y = ys[iy_lo:iy_hi]
        sub_z = zs[iz_lo:iz_hi]
        # Box-distance to current centre (reachable-radius base term)
        dx_box = np.maximum(0.0, np.abs(sub_x - ck[0]) - hx)
        dy_box = np.maximum(0.0, np.abs(sub_y - ck[1]) - hy)
        dz_box = np.maximum(0.0, np.abs(sub_z - ck[2]) - hz)
        d2_base = (
            dx_box[:, None, None] ** 2
            + dy_box[None, :, None] ** 2
            + dz_box[None, None, :] ** 2
        )
        Hbase = np.zeros_like(d2_base, dtype=np.float32)
        if alpha0 > 0 and R_reach > 1e-6:
            mask = d2_base <= R_reach * R_reach
            d = np.sqrt(np.maximum(0.0, d2_base, where=mask, out=np.zeros_like(d2_base)))
            u = np.clip(d / R_reach, 0.0, 1.0)
            contrib = alpha0 * (1.0 - u) ** heat_p
            Hbase = np.where(mask, contrib, 0.0).astype(np.float32)

        # Tube contributions across samples; per-cell max over j.
        # Vectorise the sample loop: stack (num_samples, sub_x, sub_y, sub_z)
        # in one broadcast so the inner loop runs in numpy, not Python.
        tube_max = np.zeros_like(d2_base, dtype=np.float32)
        if alpha1 > 0:
            valid = Rj > 1e-6
            if valid.any():
                # ax_dx shape (J, nx); etc.
                cx_j = centres[:, 0][:, None]  # (J, 1)
                cy_j = centres[:, 1][:, None]
                cz_j = centres[:, 2][:, None]
                dx_j = np.maximum(0.0, np.abs(sub_x[None, :] - cx_j) - hx)  # (J, nx)
                dy_j = np.maximum(0.0, np.abs(sub_y[None, :] - cy_j) - hy)  # (J, ny)
                dz_j = np.maximum(0.0, np.abs(sub_z[None, :] - cz_j) - hz)  # (J, nz)
                # Squared distances: (J, nx, ny, nz)
                d2_j = (
                    dx_j[:, :, None, None] ** 2
                    + dy_j[:, None, :, None] ** 2
                    + dz_j[:, None, None, :] ** 2
                )
                # Per-sample R² broadcast
                R2 = (Rj * Rj)[:, None, None, None]
                R_b = Rj[:, None, None, None]
                inside = d2_j <= R2
                # Distance ratio u = d/R clipped to [0, 1]
                # Where inside is False, contrib = 0 regardless of u
                with np.errstate(invalid="ignore", divide="ignore"):
                    u_j = np.clip(np.sqrt(np.maximum(0.0, d2_j)) / R_b, 0.0, 1.0)
                g_j = (1.0 - u_j) ** heat_q
                W_b = Wj[:, None, None, None]
                contrib_j = (W_b * g_j).astype(np.float32)
                contrib_j = np.where(inside & valid[:, None, None, None],
                                     contrib_j, 0.0)
                # Reduce across samples with max
                tube_max = contrib_j.max(axis=0)

        H_k = Hbase + alpha1 * tube_max
        if heat_Hmax > 0:
            H_k = np.minimum(H_k, heat_Hmax)

        # Soft-cost gate: in hard-obstacle mode (default), occupied
        # cells aren't traversable so writing heat there is wasted —
        # zero them out. With ``use_soft_cost_obstacles=True`` heat is
        # written through occupied cells so HGP can route over them at
        # a high but finite cost. Mirrors map_util.hpp:488.
        if not bool(params.use_soft_cost_obstacles):
            sub_occ = vm.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
            H_k = np.where(sub_occ, np.float32(0.0), H_k)

        sub = heat[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        np.maximum(sub, H_k, out=sub)

    return heat


def compute_static_heat(
    vm: VoxelMap,
    params: Parameters,
    *,
    boundary_only: bool = True,
) -> np.ndarray:
    """Add a halo of soft cost around static occupied cells. Mirrors the
    C++ ``static_heat_*`` parameters; ``boundary_only=True`` is the
    default-ish C++ behaviour (only the surface of occupied regions
    seeds heat — cheaper than every interior cell).
    """
    heat = vm.ensure_heat()
    if not bool(params.static_heat_enabled):
        return heat

    alpha = float(params.static_heat_alpha or 0.0)
    # C++ ``setStaticHeatRadiusFn`` evaluates a lambda per obstacle (default
    # ``[default_r](pt){ return default_r; }``) and clamps the result to
    # ``static_heat_rmax_m_``. With a const lambda the effective halo
    # radius is ``min(default_radius_m, rmax_m)`` — mirror that here so
    # the YAML's ``static_heat_default_radius_m`` actually does
    # something (it was ignored in v1, which baked ``rmax_m`` in
    # directly). map_util.hpp:755-756, hgp_manager.cpp:72-79.
    default_r = float(params.static_heat_default_radius_m or 0.0)
    cap_r = float(params.static_heat_rmax_m or 0.0)
    if cap_r > 0.0 and default_r > 0.0:
        rmax = min(default_r, cap_r)
    elif cap_r > 0.0:
        rmax = cap_r
    else:
        rmax = default_r
    static_p = int(params.static_heat_p or 2)
    Hmax_static = float(params.static_heat_Hmax or 0.0)
    if alpha <= 0 or rmax <= 1e-6:
        return heat

    occ = vm.occupied
    if not occ.any():
        return heat

    # Pick seeds: every occupied cell, or just those with a free neighbour
    if boundary_only:
        seeds = _boundary_cells(occ)
    else:
        seeds = occ
    seed_idx = np.argwhere(seeds)
    if seed_idx.size == 0:
        return heat

    r_cells = int(np.ceil(rmax / vm.res))
    # Pre-compute the falloff kernel (centred ball)
    di, dj, dk = np.mgrid[-r_cells:r_cells + 1,
                          -r_cells:r_cells + 1,
                          -r_cells:r_cells + 1]
    d_world = vm.res * np.sqrt(di ** 2 + dj ** 2 + dk ** 2)
    u = np.clip(d_world / rmax, 0.0, 1.0)
    kernel = (alpha * (1.0 - u) ** static_p).astype(np.float32)
    # Don't write heat to the seed cell itself — only its neighbourhood
    # (matches the C++ "free cells only" intent).
    center = r_cells

    # use_soft_cost_obstacles flips occupied cells from "no heat written"
    # (hard barrier) to "receive full alpha heat" (soft cost). Matches
    # map_util.hpp:654-660: occupied cells become carriers of alpha-valued
    # heat under soft-cost mode, while free cells still get the radial
    # distance-decay kernel.
    soft_cost = bool(params.use_soft_cost_obstacles)

    nx, ny, nz = vm.shape
    for si, sj, sk in seed_idx:
        ix_lo = max(0, si - r_cells); ix_hi = min(nx, si + r_cells + 1)
        iy_lo = max(0, sj - r_cells); iy_hi = min(ny, sj + r_cells + 1)
        iz_lo = max(0, sk - r_cells); iz_hi = min(nz, sk + r_cells + 1)

        kx_lo = ix_lo - (si - r_cells); kx_hi = kx_lo + (ix_hi - ix_lo)
        ky_lo = iy_lo - (sj - r_cells); ky_hi = ky_lo + (iy_hi - iy_lo)
        kz_lo = iz_lo - (sk - r_cells); kz_hi = kz_lo + (iz_hi - iz_lo)

        block = heat[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        kblock = kernel[kx_lo:kx_hi, ky_lo:ky_hi, kz_lo:kz_hi]
        occ_block = occ[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        free_mask = ~occ_block
        if soft_cost:
            # Occupied cells get the full alpha (they ARE the obstacle).
            contrib = np.where(free_mask, kblock, np.float32(alpha))
        else:
            contrib = np.where(free_mask, kblock, np.float32(0.0))
        np.maximum(block, contrib, out=block)

    if Hmax_static > 0:
        np.minimum(heat, Hmax_static, out=heat)
    return heat


def _boundary_cells(occ: np.ndarray) -> np.ndarray:
    """Boolean mask of occupied cells that have at least one free
    6-neighbour. Cheaper than all-occupied for halo computation."""
    pad = np.pad(occ, 1, mode="constant", constant_values=True)
    neighbours_occupied = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1]
        & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    # Boundary = occupied AND not all neighbours occupied
    return occ & ~neighbours_occupied
