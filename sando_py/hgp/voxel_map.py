"""3D occupancy grid + optional dynamic-heat overlay.

Mirrors the C++ ``hgp/map_util.hpp`` at a much smaller scope. Cells are
identified by ``(ix, iy, iz)`` integer indices; world coordinates are
3D ``np.ndarray``. A 2D scenario is supported as a degenerate case with
``nz = 1`` via :meth:`VoxelMap.from_bounds_2d`.

The optional ``heat`` field stores per-cell soft cost from the heat-map
overlay (dynamic obstacle predictions + static halo). Set by
:mod:`heat_map`; consumed by :mod:`astar` as an additive edge weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np


Index = Tuple[int, int, int]


@dataclass
class VoxelMap:
    """Axis-aligned 3D occupancy grid.

    Built via :meth:`from_bounds` (full 3D) or :meth:`from_bounds_2d`
    (single z layer, ``nz = 1``). Cells default to ``False`` (free);
    :meth:`mark_sphere` / :meth:`mark_circle` are the typical mutators.
    """

    x_min: float
    y_min: float
    z_min: float
    res: float
    occupied: np.ndarray = field(repr=False)  # bool grid shape (nx, ny, nz)
    heat: Optional[np.ndarray] = field(default=None, repr=False)  # float (nx, ny, nz)

    # Version counter — bumped by every mutator. Downstream callers can
    # key caches on this (e.g. ``decomp.sfc._voxel_map_occupied_points``)
    # to avoid recomputing the occupied-cells point cloud each replan
    # tick when the map hasn't changed.
    _occ_version: int = field(default=0, repr=False)
    _cached_obs_pts: Optional[np.ndarray] = field(default=None, repr=False)
    _cached_obs_pts_version: int = field(default=-1, repr=False)

    # ---- construction ----

    @classmethod
    def from_bounds(
        cls,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        z_lim: Tuple[float, float],
        res: float,
    ) -> "VoxelMap":
        """Full 3D voxel map. The cell count along each axis is
        ``ceil((hi - lo) / res)``, with a minimum of 1."""
        x_min, x_max = float(x_lim[0]), float(x_lim[1])
        y_min, y_max = float(y_lim[0]), float(y_lim[1])
        z_min, z_max = float(z_lim[0]), float(z_lim[1])
        nx = max(1, int(np.ceil((x_max - x_min) / res)))
        ny = max(1, int(np.ceil((y_max - y_min) / res)))
        nz = max(1, int(np.ceil((z_max - z_min) / res)))
        occ = np.zeros((nx, ny, nz), dtype=bool)
        return cls(
            x_min=x_min, y_min=y_min, z_min=z_min,
            res=float(res), occupied=occ,
        )

    @classmethod
    def from_bounds_2d(
        cls,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        res: float,
        z_plane: float,
    ) -> "VoxelMap":
        """Degenerate 3D map with ``nz = 1`` — for callers that pin the
        planner to a fixed altitude (the C++ ``z_plane`` mode and the
        original v1 codepath)."""
        return cls.from_bounds(
            x_lim, y_lim, (z_plane, z_plane + res), res,
        )

    # ---- shape / accessors ----

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.occupied.shape

    @property
    def nx(self) -> int:
        return self.occupied.shape[0]

    @property
    def ny(self) -> int:
        return self.occupied.shape[1]

    @property
    def nz(self) -> int:
        return self.occupied.shape[2]

    @property
    def z_plane(self) -> float:
        """Center of the single z layer when ``nz == 1`` — handy for
        callers that still want a 2D-style altitude. For ``nz > 1`` it
        returns the center of the lowest layer."""
        return self.z_min + 0.5 * self.res

    # ---- coordinate conversion ----

    def world_to_idx(self, p) -> Index:
        """3D world point -> ``(ix, iy, iz)``. ``iz`` is clamped into
        the map's z range when ``nz == 1`` (handy for callers that pass
        an altitude that doesn't fall in the single layer)."""
        p = np.asarray(p, dtype=float).reshape(-1)
        ix = int((p[0] - self.x_min) / self.res - 0.5)
        iy = int((p[1] - self.y_min) / self.res - 0.5)
        if self.nz == 1:
            iz = 0
        else:
            iz = int((p[2] - self.z_min) / self.res - 0.5)
        return ix, iy, iz

    def idx_to_world(self, idx) -> np.ndarray:
        """``(ix, iy, iz)`` -> 3D cell-center in world coordinates."""
        if len(idx) == 2:
            ix, iy = idx
            iz = 0
        else:
            ix, iy, iz = idx
        x = self.x_min + (ix + 0.5) * self.res
        y = self.y_min + (iy + 0.5) * self.res
        z = self.z_min + (iz + 0.5) * self.res
        return np.array([x, y, z], dtype=float)

    def in_bounds(self, idx) -> bool:
        if len(idx) == 2:
            ix, iy = idx
            iz = 0
        else:
            ix, iy, iz = idx
        return (0 <= ix < self.nx) and (0 <= iy < self.ny) and (0 <= iz < self.nz)

    def is_occupied(self, idx) -> bool:
        if len(idx) == 2:
            ix, iy = idx
            iz = 0
        else:
            ix, iy, iz = idx
        if not (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz):
            return True  # outside-the-map counts as blocked
        return bool(self.occupied[ix, iy, iz])

    def heat_at(self, idx) -> float:
        """Per-cell heat (0 when no heat overlay or out-of-bounds)."""
        if self.heat is None:
            return 0.0
        if len(idx) == 2:
            ix, iy = idx
            iz = 0
        else:
            ix, iy, iz = idx
        if not (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz):
            return 0.0
        return float(self.heat[ix, iy, iz])

    # ---- mutators (sphere / circle) ----

    def mark_sphere(self, center, radius: float) -> int:
        """Mark all cells within ``radius`` of ``center`` in 3D.
        Returns the number of cells flipped."""
        if radius <= 0:
            return 0
        c = np.asarray(center, dtype=float).reshape(-1)
        ix0, iy0, iz0 = self.world_to_idx(c)
        r_cells = int(np.ceil(radius / self.res))

        ix_lo = max(0, ix0 - r_cells); ix_hi = min(self.nx, ix0 + r_cells + 1)
        iy_lo = max(0, iy0 - r_cells); iy_hi = min(self.ny, iy0 + r_cells + 1)
        iz_lo = max(0, iz0 - r_cells); iz_hi = min(self.nz, iz0 + r_cells + 1)
        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            return 0

        xs = self.x_min + (np.arange(ix_lo, ix_hi) + 0.5) * self.res
        ys = self.y_min + (np.arange(iy_lo, iy_hi) + 0.5) * self.res
        zs = self.z_min + (np.arange(iz_lo, iz_hi) + 0.5) * self.res
        dx2 = (xs - c[0]) ** 2
        dy2 = (ys - c[1]) ** 2
        dz2 = (zs - c[2]) ** 2
        within = (
            dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]
            <= radius * radius
        )
        before = int(self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi].sum())
        self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] |= within
        after = int(self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi].sum())
        if after != before:
            self._occ_version += 1
        return after - before

    def clear_sphere(self, center, radius: float) -> int:
        """Mirror of :meth:`mark_sphere`. Used by ``use_free_start`` /
        ``use_free_goal`` to escape an agent that got swallowed by
        inflated obstacles."""
        if radius <= 0:
            return 0
        c = np.asarray(center, dtype=float).reshape(-1)
        ix0, iy0, iz0 = self.world_to_idx(c)
        r_cells = int(np.ceil(radius / self.res))

        ix_lo = max(0, ix0 - r_cells); ix_hi = min(self.nx, ix0 + r_cells + 1)
        iy_lo = max(0, iy0 - r_cells); iy_hi = min(self.ny, iy0 + r_cells + 1)
        iz_lo = max(0, iz0 - r_cells); iz_hi = min(self.nz, iz0 + r_cells + 1)
        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            return 0

        xs = self.x_min + (np.arange(ix_lo, ix_hi) + 0.5) * self.res
        ys = self.y_min + (np.arange(iy_lo, iy_hi) + 0.5) * self.res
        zs = self.z_min + (np.arange(iz_lo, iz_hi) + 0.5) * self.res
        within = (
            (xs - c[0])[:, None, None] ** 2
            + (ys - c[1])[None, :, None] ** 2
            + (zs - c[2])[None, None, :] ** 2
            <= radius * radius
        )
        flipped = int(
            (self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] & within).sum()
        )
        self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] &= ~within
        if flipped:
            self._occ_version += 1
        return flipped

    def mark_circle(self, center, radius: float) -> int:
        """2D-style mutator: mark cells within ``radius`` in the x-y
        plane, applied across every z layer. Equivalent to
        ``mark_sphere`` when ``nz == 1``; for ``nz > 1`` it inflates
        the obstacle as an infinite vertical cylinder, which is the
        right thing for a UAV at a fixed altitude band."""
        if radius <= 0:
            return 0
        c = np.asarray(center, dtype=float).reshape(-1)
        cx, cy = float(c[0]), float(c[1])
        r_cells = int(np.ceil(radius / self.res))
        ix0, iy0, _ = self.world_to_idx([cx, cy, self.z_min])

        ix_lo = max(0, ix0 - r_cells); ix_hi = min(self.nx, ix0 + r_cells + 1)
        iy_lo = max(0, iy0 - r_cells); iy_hi = min(self.ny, iy0 + r_cells + 1)
        if ix_lo >= ix_hi or iy_lo >= iy_hi:
            return 0

        xs = self.x_min + (np.arange(ix_lo, ix_hi) + 0.5) * self.res
        ys = self.y_min + (np.arange(iy_lo, iy_hi) + 0.5) * self.res
        within = ((xs - cx)[:, None] ** 2 + (ys - cy)[None, :] ** 2) <= radius * radius
        block = self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, :]
        before = int(block.sum())
        block |= within[:, :, None]
        after = int(block.sum())
        if after != before:
            self._occ_version += 1
        return after - before

    def clear_circle(self, center, radius: float) -> int:
        """Mirror of :meth:`mark_circle`."""
        if radius <= 0:
            return 0
        c = np.asarray(center, dtype=float).reshape(-1)
        cx, cy = float(c[0]), float(c[1])
        r_cells = int(np.ceil(radius / self.res))
        ix0, iy0, _ = self.world_to_idx([cx, cy, self.z_min])

        ix_lo = max(0, ix0 - r_cells); ix_hi = min(self.nx, ix0 + r_cells + 1)
        iy_lo = max(0, iy0 - r_cells); iy_hi = min(self.ny, iy0 + r_cells + 1)
        if ix_lo >= ix_hi or iy_lo >= iy_hi:
            return 0

        xs = self.x_min + (np.arange(ix_lo, ix_hi) + 0.5) * self.res
        ys = self.y_min + (np.arange(iy_lo, iy_hi) + 0.5) * self.res
        within = ((xs - cx)[:, None] ** 2 + (ys - cy)[None, :] ** 2) <= radius * radius
        block = self.occupied[ix_lo:ix_hi, iy_lo:iy_hi, :]
        flipped = int((block & within[:, :, None]).sum())
        block &= ~within[:, :, None]
        if flipped:
            self._occ_version += 1
        return flipped

    # ---- point cloud rasterisation ----

    def mark_points(self, points, *, inflation_cells: int = 0) -> int:
        """Mark every cell that a point lands in as occupied.

        ``points`` is an ``(N, 3)`` array of world-coordinate samples
        (e.g. a depth camera point cloud transformed into the map
        frame). Returns the number of cells newly flipped.

        ``inflation_cells`` (default 0) widens each point's footprint
        by that many cells along each axis — a cheap morphological
        dilation that mimics the C++ ``inflation_hgp`` for sensor data.
        Implemented as a manual 3-axis dilation on the cell-index set
        (no scipy dependency).
        """
        if points is None:
            return 0
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return 0
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(
                f"mark_points expects (N, 3); got {pts.shape}"
            )

        ix = ((pts[:, 0] - self.x_min) / self.res - 0.5).astype(np.int64)
        iy = ((pts[:, 1] - self.y_min) / self.res - 0.5).astype(np.int64)
        iz = ((pts[:, 2] - self.z_min) / self.res - 0.5).astype(np.int64)
        mask = (
            (ix >= 0) & (ix < self.nx)
            & (iy >= 0) & (iy < self.ny)
            & (iz >= 0) & (iz < self.nz)
        )
        ix, iy, iz = ix[mask], iy[mask], iz[mask]
        if ix.size == 0:
            return 0

        # Inflation: for each cell, also mark the (2r+1)³ - 1
        # neighbours within ``inflation_cells`` Chebyshev distance.
        # Vectorised via offset arrays.
        if inflation_cells > 0:
            r = int(inflation_cells)
            di, dj, dk = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
            offsets = np.stack(
                [di.ravel(), dj.ravel(), dk.ravel()], axis=-1
            )
            ix = (ix[:, None] + offsets[None, :, 0]).ravel()
            iy = (iy[:, None] + offsets[None, :, 1]).ravel()
            iz = (iz[:, None] + offsets[None, :, 2]).ravel()
            mask = (
                (ix >= 0) & (ix < self.nx)
                & (iy >= 0) & (iy < self.ny)
                & (iz >= 0) & (iz < self.nz)
            )
            ix, iy, iz = ix[mask], iy[mask], iz[mask]
            if ix.size == 0:
                return 0

        before = int(self.occupied[ix, iy, iz].sum())
        self.occupied[ix, iy, iz] = True
        after = int(self.occupied[ix, iy, iz].sum())
        if after != before:
            self._occ_version += 1
        return after - before

    # ---- heat ----

    def ensure_heat(self) -> np.ndarray:
        """Lazily allocate the heat field and return it (callers can
        write to it in place)."""
        if self.heat is None:
            self.heat = np.zeros(self.occupied.shape, dtype=np.float32)
        return self.heat

    def clear_heat(self) -> None:
        self.heat = None

    # ---- bulk helpers ----

    def mark_circles(self, circles: Iterable[Tuple[np.ndarray, float]]) -> int:
        n = 0
        for center, radius in circles:
            n += self.mark_circle(center, radius)
        return n
