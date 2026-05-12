"""Axis-aligned bounding-box safety corridor for a single path segment.

Grows a 3D AABB around the segment ``[p0, p1]`` one voxel step at a time,
expanding each of the 6 faces outward until it hits

* an occupied cell in the 2D voxel map (only x / y faces — the v1 voxel
  map is 2D so the ±z faces only halt on the ``z_min`` / ``z_max`` world
  bounds), or
* the ``params.sfc_size[axis]`` cap measured from the segment midpoint,
  whichever is hit first per face.

This is the v1 stand-in for the C++ ellipsoid-dilation SFC
(``HGPManager::cvxEllipsoidDecomp`` in ``hgp_manager.cpp``). Slightly
more conservative, much simpler, and good enough to give the local
solver a starting set of corridors. ``decomp/README.md`` covers the
trade-off.
"""

from __future__ import annotations

import numpy as np

from sando_py.core import Parameters, Polytope
from sando_py.hgp import VoxelMap


# Right-hand-rule order so callers can rely on a stable layout:
# rows 0..2 are +x / +y / +z, rows 3..5 are -x / -y / -z.
_AXES = (0, 1, 2)
_SIGNS = (+1, -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grow_aabb(
    p0,
    p1,
    voxel_map: VoxelMap,
    params: Parameters,
) -> Polytope:
    """Return an AABB ``Polytope`` enclosing the segment ``[p0, p1]``.

    The returned polytope has exactly 6 rows in ``A`` (axis-aligned, half-
    spaces in this order: +x, +y, +z, -x, -y, -z) and 6 entries in ``b``
    of shape ``(6, 1)``. ``A @ x <= b`` defines the corridor.

    Both endpoints are always strictly inside the box modulo a numerical
    margin: the seed box bounding them is grown, never shrunk past them.
    """
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)

    seed_lo = np.minimum(p0, p1)
    seed_hi = np.maximum(p0, p1)
    midpoint = 0.5 * (p0 + p1)

    # Seed half-thickness: drone radius is the natural choice. Force at
    # least half a voxel either side so a degenerate (p0 == p1) segment
    # still gives a 1-cell-wide box.
    r = max(float(params.drone_radius), 0.5 * float(voxel_map.res))
    box_lo = seed_lo - r
    box_hi = seed_hi + r

    # World bounds — sfc must lie inside the voxel map (for x / y) and
    # inside [z_min, z_max] (for z). The voxel-map's z is just a planner
    # altitude, not a bound, so we take z from params.
    vm_x_max = voxel_map.x_min + voxel_map.nx * voxel_map.res
    vm_y_max = voxel_map.y_min + voxel_map.ny * voxel_map.res

    world_lo = np.array([voxel_map.x_min, voxel_map.y_min, float(params.z_min)])
    world_hi = np.array([vm_x_max, vm_y_max, float(params.z_max)])

    # If z_min/z_max aren't meaningfully set (z_min >= z_max), just keep
    # the seed z range so the cap below is non-empty.
    if world_hi[2] <= world_lo[2]:
        world_lo[2] = box_lo[2] - 1e-3
        world_hi[2] = box_hi[2] + 1e-3

    # Clamp the seed into the world to start from something legal.
    box_lo = np.maximum(box_lo, world_lo)
    box_hi = np.minimum(box_hi, world_hi)
    # If clamping inverted the box (start outside the world), give it
    # zero thickness in that axis rather than negative.
    box_hi = np.maximum(box_hi, box_lo)

    # sfc_size cap (half-extents from the segment midpoint). If the
    # parameter is unset / empty, fall back to a generous default so we
    # don't accidentally produce an empty corridor.
    sfc_size = list(params.sfc_size) if params.sfc_size else [10.0, 10.0, 10.0]
    if len(sfc_size) < 3:
        sfc_size = list(sfc_size) + [sfc_size[-1]] * (3 - len(sfc_size))
    cap_lo = midpoint - np.asarray(sfc_size[:3], dtype=float)
    cap_hi = midpoint + np.asarray(sfc_size[:3], dtype=float)

    # Final per-face stop targets: tightest of (sfc_size cap, world bound).
    stop_hi = np.minimum(cap_hi, world_hi)
    stop_lo = np.maximum(cap_lo, world_lo)

    step = voxel_map.res

    halted = [False, False, False, False, False, False]  # +x +y +z -x -y -z

    # Round-robin expansion — keeps the box's shape roughly balanced
    # rather than greedily blowing out one axis first.
    while not all(halted):
        progressed = False
        for fi, (axis, sign) in enumerate(((0, +1), (1, +1), (2, +1),
                                           (0, -1), (1, -1), (2, -1))):
            if halted[fi]:
                continue
            if sign == +1:
                curr = box_hi[axis]
                target = stop_hi[axis]
                if curr >= target - 1e-12:
                    halted[fi] = True
                    continue
                new_val = min(curr + step, target)
            else:
                curr = box_lo[axis]
                target = stop_lo[axis]
                if curr <= target + 1e-12:
                    halted[fi] = True
                    continue
                new_val = max(curr - step, target)

            if axis == 2 or _slab_is_free(voxel_map, box_lo, box_hi, axis, sign,
                                          curr, new_val):
                if sign == +1:
                    box_hi[axis] = new_val
                else:
                    box_lo[axis] = new_val
                progressed = True
            else:
                halted[fi] = True

        if not progressed:
            # All remaining faces are blocked — fixed point.
            break

    # Optional uniform shrink (mirrors C++ ``use_shrinked_box`` /
    # ``shrinked_box_size``). Never shrink past the segment endpoints.
    if params.use_shrinked_box and params.shrinked_box_size > 0:
        s = float(params.shrinked_box_size)
        box_lo = np.minimum(box_lo + s, seed_lo)
        box_hi = np.maximum(box_hi - s, seed_hi)

    return _aabb_to_polytope(box_lo, box_hi)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _slab_is_free(
    vm: VoxelMap,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    axis: int,
    sign: int,
    curr: float,
    new_val: float,
) -> bool:
    """True iff no cell inside the expansion slab on ``axis`` is occupied.

    The slab is the 2D rectangle the new face would sweep through:
    ``[curr, new_val]`` (sorted) along ``axis``, full extent on the other
    in-plane axis. ``axis`` is 0 or 1 here — z is handled by the caller.
    """
    # Other planar axis (only one in the 2D voxel map)
    other = 1 if axis == 0 else 0

    if axis == 0:
        x_lo, x_hi = (curr, new_val) if sign == +1 else (new_val, curr)
        y_lo, y_hi = box_lo[other], box_hi[other]
    else:
        y_lo, y_hi = (curr, new_val) if sign == +1 else (new_val, curr)
        x_lo, x_hi = box_lo[other], box_hi[other]

    # Voxel grid range covering the slab. Use a half-cell inset so we
    # check cell *centers* rather than slab edges — otherwise a face
    # sliding flush against a row of obstacles flips occupancy on a knife
    # edge.
    res = vm.res
    eps = 0.5 * res

    ix_lo = max(0, int(np.floor((x_lo + eps - vm.x_min) / res)))
    ix_hi = min(vm.nx, int(np.ceil((x_hi - eps - vm.x_min) / res)) + 1)
    iy_lo = max(0, int(np.floor((y_lo + eps - vm.y_min) / res)))
    iy_hi = min(vm.ny, int(np.ceil((y_hi - eps - vm.y_min) / res)) + 1)

    if ix_lo >= ix_hi or iy_lo >= iy_hi:
        return True
    # Slab covers the AABB's full z range. For a 2D map (``nz == 1``)
    # this collapses to a single z layer; for a 3D map we check every
    # layer that intersects the box.
    iz_lo = max(0, int(np.floor((box_lo[2] + eps - vm.z_min) / res)))
    iz_hi = min(vm.nz, int(np.ceil((box_hi[2] - eps - vm.z_min) / res)) + 1)
    if iz_lo >= iz_hi:
        return True
    return not bool(
        vm.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi].any()
    )


def _aabb_to_polytope(box_lo: np.ndarray, box_hi: np.ndarray) -> Polytope:
    """Build the 6-row ``A x <= b`` representation of an AABB.

    Row order matches ``_AXES`` × ``_SIGNS`` flattened with positive
    signs first: +x, +y, +z, -x, -y, -z.
    """
    A = np.zeros((6, 3), dtype=float)
    b = np.zeros((6, 1), dtype=float)
    for i in range(3):
        A[i, i] = 1.0
        b[i, 0] = float(box_hi[i])
        A[3 + i, i] = -1.0
        b[3 + i, 0] = -float(box_lo[i])
    return Polytope(A=A, b=b)
