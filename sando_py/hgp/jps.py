"""3D Jump Point Search.

Port of the JPS variant in the C++ ``hgp/graph_search.cpp`` at the
algorithmic level — same natural/forced neighbour rules as the
``JPS3DNeib`` tables, but computed on demand instead of with a giant
precomputed lookup. Same path-quality guarantees as A* on uniform grids
(JPS is admissible) but typically 10-100x fewer node expansions on
sparse maps.

JPS does **not** support heat-weighted edges — it relies on the
uniform-cost property of the grid. Callers asking for heat should use
``astar_heat`` instead; the ``HGPPlanner`` mode dispatch enforces this.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Tuple

from .voxel_map import Index, VoxelMap


_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)

# All 26 directions of motion in 3D (sorted by norm1 for stable
# iteration). norm1 = |dx| + |dy| + |dz|.
_DIRS_NORM1 = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)
_DIRS_NORM2 = (
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
)
_DIRS_NORM3 = (
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
)
_ALL_DIRS = _DIRS_NORM1 + _DIRS_NORM2 + _DIRS_NORM3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def jps(
    start: Index,
    goal: Index,
    vm: VoxelMap,
    *,
    heuristic_weight: float = 1.0,
    max_expansions: int = 100_000,
    timeout_s: float = 0.0,
) -> Optional[List[Index]]:
    """3D Jump Point Search. Returns indices ``[start, ..., goal]`` or
    ``None`` if no path was found, the budget was exhausted, or the
    timeout fired.

    ``timeout_s`` (default ``0`` = unbounded) is the wall-clock budget
    in seconds — mirrors the C++ ``hgp_timeout_duration_ms``.
    """
    start = _to_3d(start)
    goal = _to_3d(goal)

    if start == goal:
        return [start]
    if not vm.in_bounds(start) or not vm.in_bounds(goal):
        return None
    if vm.is_occupied(goal):
        return None

    occ = vm.occupied
    nx, ny, nz = vm.nx, vm.ny, vm.nz
    gx, gy, gz = goal
    t_start = time.perf_counter()

    open_heap: List[Tuple[float, int, Index]] = []
    counter = 0
    # Each open-list entry also carries the direction we arrived from
    # so we can prune the natural-neighbour set on the next expansion.
    # came_from: node -> (parent, parent_direction)
    heapq.heappush(open_heap, (0.0, counter, start))
    came_from: Dict[Index, Tuple[Optional[Index], Optional[Tuple[int, int, int]]]] = {
        start: (None, None)
    }
    g_score: Dict[Index, float] = {start: 0.0}
    expansions = 0

    while open_heap:
        if timeout_s > 0 and (time.perf_counter() - t_start) > timeout_s:
            return None

        _, _, current = heapq.heappop(open_heap)

        if current == goal:
            return _reconstruct(came_from, current)

        expansions += 1
        if expansions > max_expansions:
            return None

        parent_dir = came_from[current][1]
        g_cur = g_score[current]

        for d in _pruned_neighbour_directions(current, parent_dir, occ, nx, ny, nz):
            jp = _jump(current, d, occ, nx, ny, nz, gx, gy, gz)
            if jp is None:
                continue
            # Step cost = Euclidean distance from current to jp.
            dx = jp[0] - current[0]
            dy = jp[1] - current[1]
            dz = jp[2] - current[2]
            cost = math.sqrt(dx * dx + dy * dy + dz * dz)
            tentative = g_cur + cost
            if tentative < g_score.get(jp, math.inf):
                came_from[jp] = (current, d)
                g_score[jp] = tentative
                f = tentative + heuristic_weight * _octile_3d(jp, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, jp))

    return None


# ---------------------------------------------------------------------------
# Jump
# ---------------------------------------------------------------------------

def _jump(
    curr: Index,
    direction: Tuple[int, int, int],
    occ,
    nx: int,
    ny: int,
    nz: int,
    gx: int,
    gy: int,
    gz: int,
) -> Optional[Index]:
    """Recursive jump along ``direction`` from ``curr`` — returns the
    next jump point (goal, obstacle-forced turn, or natural decomposition
    of a diagonal move) or ``None`` if the line runs into an obstacle
    or off the map first.

    Iterative form to avoid Python recursion overhead on long axis-
    aligned runs (the common case in sparse maps).
    """
    dx, dy, dz = direction
    norm1 = abs(dx) + abs(dy) + abs(dz)
    x, y, z = curr

    while True:
        x += dx
        y += dy
        z += dz
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            return None
        if occ[x, y, z]:
            return None
        if x == gx and y == gy and z == gz:
            return (x, y, z)

        # Forced-neighbour check
        if _has_forced(x, y, z, dx, dy, dz, occ, nx, ny, nz):
            return (x, y, z)

        # Diagonal cases: also probe lower-norm sub-directions; if any
        # of them yields a jump point, this cell becomes a successor.
        if norm1 >= 2:
            for sub_d in _sub_directions(direction):
                if _jump(
                    (x, y, z), sub_d, occ, nx, ny, nz, gx, gy, gz,
                ) is not None:
                    return (x, y, z)

        # Otherwise keep jumping along ``direction``.


# ---------------------------------------------------------------------------
# Neighbour pruning
# ---------------------------------------------------------------------------

def _pruned_neighbour_directions(
    curr: Index,
    parent_dir: Optional[Tuple[int, int, int]],
    occ,
    nx: int,
    ny: int,
    nz: int,
):
    """Yield the directions we need to try jumping in from ``curr``.

    With no parent direction (start node), every 3D move is valid. With
    a parent direction, we yield only:
      * natural neighbours (continuations of the parent direction)
      * forced neighbours (directions only reachable through curr
        because of a wall blocking the direct parent path)
    """
    if parent_dir is None:
        yield from _ALL_DIRS
        return

    natural = _natural_neighbours(parent_dir)
    yield from natural

    yield from _forced_directions(curr, parent_dir, occ, nx, ny, nz)


def _natural_neighbours(direction: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], ...]:
    """Natural successors that ``continue'' the parent direction.

    * norm1=1 (axis):     1 direction (same axis)
    * norm1=2 (planar):   3 directions (the planar diagonal + its two
                          axis components)
    * norm1=3 (3D diag):  7 directions (the 3D diagonal, its three
                          planar projections, and its three axes)
    """
    dx, dy, dz = direction
    norm1 = abs(dx) + abs(dy) + abs(dz)
    if norm1 == 1:
        return (direction,)
    if norm1 == 2:
        # The two non-zero axes
        axes = []
        if dx != 0:
            axes.append((dx, 0, 0))
        if dy != 0:
            axes.append((0, dy, 0))
        if dz != 0:
            axes.append((0, 0, dz))
        return (direction, axes[0], axes[1])
    # norm1 == 3
    axes = ((dx, 0, 0), (0, dy, 0), (0, 0, dz))
    planar = ((dx, dy, 0), (dx, 0, dz), (0, dy, dz))
    return (direction,) + planar + axes


def _sub_directions(
    direction: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], ...]:
    """Lower-norm sub-directions of ``direction``. Used in ``_jump`` to
    probe whether a diagonal motion decomposes into a useful axis-or-
    planar move.

    For a planar diagonal (norm1=2), the sub-directions are the two
    axes. For a 3D diagonal (norm1=3), the sub-directions are the three
    axes plus the three planar diagonals.
    """
    dx, dy, dz = direction
    norm1 = abs(dx) + abs(dy) + abs(dz)
    if norm1 == 1:
        return ()
    if norm1 == 2:
        axes = []
        if dx != 0:
            axes.append((dx, 0, 0))
        if dy != 0:
            axes.append((0, dy, 0))
        if dz != 0:
            axes.append((0, 0, dz))
        return tuple(axes)
    # norm1 == 3
    return (
        (dx, 0, 0), (0, dy, 0), (0, 0, dz),
        (dx, dy, 0), (dx, 0, dz), (0, dy, dz),
    )


def _has_forced(
    x: int, y: int, z: int,
    dx: int, dy: int, dz: int,
    occ,
    nx: int, ny: int, nz: int,
) -> bool:
    """``(x, y, z)`` is a jump point iff at least one of its neighbours
    would be unreachable from the parent without passing through here
    — i.e. the wall geometry forces a turn.

    For an axis-aligned move, the parent is at ``(x-dx, y-dy, z-dz)``.
    A perpendicular neighbour ``(x, y+1, z)`` is forced iff
    ``(x-dx, y+1, z-dz)`` is blocked but ``(x, y+1, z)`` itself is free.
    Generalised here to all motion types.
    """
    px, py, pz = x - dx, y - dy, z - dz

    # Iterate the 26 neighbours of (x, y, z); skip directions
    # collinear with motion (those are "natural", not forced).
    for d in _ALL_DIRS:
        nx_, ny_, nz_ = x + d[0], y + d[1], z + d[2]
        if not (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz):
            continue
        if occ[nx_, ny_, nz_]:
            continue
        # Is the corresponding parent-side neighbour blocked or off-map?
        # If yes, then this neighbour is forced.
        ppx, ppy, ppz = px + d[0], py + d[1], pz + d[2]
        if not (0 <= ppx < nx and 0 <= ppy < ny and 0 <= ppz < nz):
            blocked = True
        else:
            blocked = bool(occ[ppx, ppy, ppz])
        if blocked:
            # Direction from current to that forced neighbour
            return True
    return False


def _forced_directions(
    curr: Index,
    parent_dir: Tuple[int, int, int],
    occ,
    nx: int,
    ny: int,
    nz: int,
):
    """Directions from ``curr`` that lead to forced neighbours."""
    x, y, z = curr
    dx, dy, dz = parent_dir
    px, py, pz = x - dx, y - dy, z - dz

    for d in _ALL_DIRS:
        # Skip natural neighbours — they're yielded separately.
        if d == parent_dir:
            continue
        nx_, ny_, nz_ = x + d[0], y + d[1], z + d[2]
        if not (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz):
            continue
        if occ[nx_, ny_, nz_]:
            continue
        ppx, ppy, ppz = px + d[0], py + d[1], pz + d[2]
        if (
            not (0 <= ppx < nx and 0 <= ppy < ny and 0 <= ppz < nz)
            or bool(occ[ppx, ppy, ppz])
        ):
            yield d


# ---------------------------------------------------------------------------
# Heuristic & path reconstruction
# ---------------------------------------------------------------------------

def _octile_3d(a: Index, b: Index) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    d = sorted((dx, dy, dz))
    dmin, dmid, dmax = d[0], d[1], d[2]
    return (_SQRT3 - _SQRT2) * dmin + (_SQRT2 - 1.0) * dmid + dmax


def _to_3d(idx) -> Index:
    if len(idx) == 2:
        return (idx[0], idx[1], 0)
    return tuple(idx)  # type: ignore[return-value]


def _reconstruct(came_from: dict, current: Index) -> List[Index]:
    """JPS only stores jump points; the path between them is a straight
    line. We just walk back through the parent chain and let
    :func:`hgp.simplify_path_los` decide whether to leave the line
    waypoints implicit or split them out."""
    out: List[Index] = [current]
    while came_from[current][0] is not None:
        current = came_from[current][0]
        out.append(current)
    out.reverse()
    return out
