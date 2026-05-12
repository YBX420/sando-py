"""3D 26-connected A* over a :class:`VoxelMap`.

The grid is 3D-canonical — a 2D scenario is the degenerate case ``nz =
1``, in which case the 26-connected neighbourhood collapses to the
8-connected 2D one for free.

Edge cost = geometric step + ``heat_weight * vm.heat[neighbor]``. When
no heat overlay is set, the cost reduces to the pure geometric
shortest path (= the v1 behaviour).

Returns the path as a list of ``(ix, iy, iz)`` index tuples or
``None`` if no path was found within ``max_expansions``.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Optional, Tuple

from .voxel_map import Index, VoxelMap


_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)


def _neighbors_3d() -> Tuple[Tuple[int, int, int, float], ...]:
    """All 26 neighbours in a 3D grid: 6 axis-aligned (cost 1), 12
    face-diagonal (cost √2), 8 corner-diagonal (cost √3)."""
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                step = math.sqrt(dx * dx + dy * dy + dz * dz)
                out.append((dx, dy, dz, step))
    return tuple(out)


_NEIGHBORS_3D: Tuple[Tuple[int, int, int, float], ...] = _neighbors_3d()


def octile(a: Index, b: Index) -> float:
    """3D octile distance (Chebyshev with √2 / √3 diagonals).

    Reduces to the classic 2D octile when ``a[2] == b[2]``.
    """
    if len(a) == 2:
        a = (a[0], a[1], 0)
    if len(b) == 2:
        b = (b[0], b[1], 0)
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    # Sort dx, dy, dz so dmin <= dmid <= dmax
    d = sorted((dx, dy, dz))
    dmin, dmid, dmax = d[0], d[1], d[2]
    return (_SQRT3 - _SQRT2) * dmin + (_SQRT2 - 1.0) * dmid + dmax


def astar(
    start: Index,
    goal: Index,
    vm: VoxelMap,
    *,
    heuristic_weight: float = 1.0,
    max_expansions: int = 100_000,
    heat_weight: float = 0.0,
    timeout_s: float = 0.0,
) -> Optional[List[Index]]:
    """Run A* in 3D (or 2D-as-degenerate-3D). Returns indices
    ``[start, ..., goal]`` or ``None`` if no path was found.

    ``heat_weight`` is the multiplier on per-cell heat for the edge
    cost. ``0`` (default) reduces to pure geometric A*.

    ``timeout_s`` (default ``0`` = unbounded) is the wall-clock budget
    in seconds; matches the C++ ``hgp_timeout_duration_ms`` plumbing.
    """
    # Promote 2-tuples to 3-tuples for backwards-compat with callers
    # that still use the 2D-style index format.
    start = _to_3d(start)
    goal = _to_3d(goal)

    if start == goal:
        return [start]
    if not vm.in_bounds(start) or not vm.in_bounds(goal):
        return None
    if vm.is_occupied(goal):
        return None

    heat = vm.heat  # may be None
    use_heat = heat is not None and heat_weight > 0.0

    occ = vm.occupied
    nx, ny, nz = vm.nx, vm.ny, vm.nz
    nynz = ny * nz

    def _enc(ix, iy, iz):
        return ix * nynz + iy * nz + iz

    def _dec(idx):
        ix, rem = divmod(idx, nynz)
        iy, iz = divmod(rem, nz)
        return (ix, iy, iz)

    start_idx = _enc(*start)
    goal_idx = _enc(*goal)
    gx, gy, gz = goal

    open_heap: List[Tuple[float, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start_idx))
    # int → int dict is ~2-3× faster than tuple → tuple. Combined with
    # the flat-index encoding the per-expansion hash cost drops a lot.
    came_from: dict = {start_idx: -1}
    g_score: dict = {start_idx: 0.0}
    expansions = 0

    t_start = time.perf_counter()
    inf = math.inf

    while open_heap:
        if timeout_s > 0 and (time.perf_counter() - t_start) > timeout_s:
            return None

        _, _, current = heapq.heappop(open_heap)

        if current == goal_idx:
            return _reconstruct_flat(came_from, current, nynz, nz)

        expansions += 1
        if expansions > max_expansions:
            return None

        cx, cy, cz = _dec(current)
        g_cur = g_score[current]
        for dx, dy, dz, step in _NEIGHBORS_3D:
            nxn, nyn, nzn = cx + dx, cy + dy, cz + dz
            if nxn < 0 or nxn >= nx or nyn < 0 or nyn >= ny or nzn < 0 or nzn >= nz:
                continue
            if occ[nxn, nyn, nzn]:
                continue
            nb = nxn * nynz + nyn * nz + nzn
            cost = step
            if use_heat:
                cost += heat_weight * float(heat[nxn, nyn, nzn])
            tentative = g_cur + cost
            if tentative < g_score.get(nb, inf):
                came_from[nb] = current
                g_score[nb] = tentative
                # Inline octile heuristic for hot loop.
                ddx = abs(nxn - gx); ddy = abs(nyn - gy); ddz = abs(nzn - gz)
                dmin = ddx if ddx < ddy else ddy
                dmin = dmin if dmin < ddz else ddz
                dmax = ddx if ddx > ddy else ddy
                dmax = dmax if dmax > ddz else ddz
                dmid = ddx + ddy + ddz - dmin - dmax
                h = (_SQRT3 - _SQRT2) * dmin + (_SQRT2 - 1.0) * dmid + dmax
                f = tentative + heuristic_weight * h
                counter += 1
                heapq.heappush(open_heap, (f, counter, nb))

    return None


def _reconstruct_flat(came_from: dict, current: int, nynz: int, nz: int) -> List[Index]:
    out: List[Index] = []
    while current != -1:
        ix, rem = divmod(current, nynz)
        iy, iz = divmod(rem, nz)
        out.append((ix, iy, iz))
        current = came_from[current]
    out.reverse()
    return out


def _to_3d(idx) -> Index:
    if len(idx) == 2:
        return (idx[0], idx[1], 0)
    return tuple(idx)  # type: ignore[return-value]


def _reconstruct(came_from: dict, current: Index) -> List[Index]:
    out: List[Index] = [current]
    while came_from[current] is not None:
        current = came_from[current]
        out.append(current)
    out.reverse()
    return out


def simplify_path_los(path: List[Index], vm: VoxelMap) -> List[Index]:
    """Pull straight-line shortcuts across consecutive waypoints.

    Greedy line-of-sight smoothing — for each anchor, advance the second
    pointer as far as possible while the segment is still
    collision-free. Works in 3D via :func:`_line_clear_3d`.
    """
    if len(path) <= 2:
        return list(path)
    path3 = [_to_3d(p) for p in path]
    out: List[Index] = [path3[0]]
    i = 0
    while i < len(path3) - 1:
        j = len(path3) - 1
        while j > i + 1 and not _line_clear_3d(path3[i], path3[j], vm):
            j -= 1
        out.append(path3[j])
        i = j
    return out


def _line_clear_3d(a: Index, b: Index, vm: VoxelMap) -> bool:
    """3D Bresenham; ``True`` if every cell on the line is free.

    Faithful to the standard 3D-line-rasterisation algorithm — pick the
    driving axis (largest absolute delta), step one cell at a time on
    that axis while accumulating fractional progress on the other two.
    """
    x0, y0, z0 = a
    x1, y1, z1 = b
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0); sy = 1 if y0 < y1 else -1
    dz = abs(z1 - z0); sz = 1 if z0 < z1 else -1

    x, y, z = x0, y0, z0
    if dx >= dy and dx >= dz:
        err_1 = 2 * dy - dx
        err_2 = 2 * dz - dx
        for _ in range(dx + 1):
            if not (0 <= x < vm.nx and 0 <= y < vm.ny and 0 <= z < vm.nz):
                return False
            if vm.occupied[x, y, z]:
                return False
            if err_1 > 0:
                y += sy
                err_1 -= 2 * dx
            if err_2 > 0:
                z += sz
                err_2 -= 2 * dx
            err_1 += 2 * dy
            err_2 += 2 * dz
            x += sx
    elif dy >= dx and dy >= dz:
        err_1 = 2 * dx - dy
        err_2 = 2 * dz - dy
        for _ in range(dy + 1):
            if not (0 <= x < vm.nx and 0 <= y < vm.ny and 0 <= z < vm.nz):
                return False
            if vm.occupied[x, y, z]:
                return False
            if err_1 > 0:
                x += sx
                err_1 -= 2 * dy
            if err_2 > 0:
                z += sz
                err_2 -= 2 * dy
            err_1 += 2 * dx
            err_2 += 2 * dz
            y += sy
    else:
        err_1 = 2 * dy - dz
        err_2 = 2 * dx - dz
        for _ in range(dz + 1):
            if not (0 <= x < vm.nx and 0 <= y < vm.ny and 0 <= z < vm.nz):
                return False
            if vm.occupied[x, y, z]:
                return False
            if err_1 > 0:
                y += sy
                err_1 -= 2 * dz
            if err_2 > 0:
                x += sx
                err_2 -= 2 * dz
            err_1 += 2 * dy
            err_2 += 2 * dx
            z += sz
    return True
