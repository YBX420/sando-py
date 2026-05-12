"""Ellipsoid + dilation safety corridor for a single path segment.

Faithful port of DecompROS2's ``EllipsoidDecomp3D`` (and its
``LineSegment<3>`` worker), located at
``deps/DecompROS2/DecompUtil/include/decomp_util/{ellipsoid_decomp,line_segment}.h``.

The algorithm grows a maximum-volume ellipsoid around the segment, then
slices it into half-spaces via its closest hyperplanes to obstacles. The
half-spaces are then padded with the local bbox walls, z bounds, and
global bbox walls. The result is a convex polytope ``Ax <= b`` containing
the segment.

Compared with the v1 AABB (``aabb.py``), this corridor is tighter and
rotated to follow the segment direction, which lets the local solver
issue faster trajectories while still respecting obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from sando_py.core import Polytope


_EPS = 1e-10
_COUNTER_MAX = 50


# ---------------------------------------------------------------------------
# Geometry primitives — direct port of decomp_geometry/{ellipsoid,polyhedron}.h
# ---------------------------------------------------------------------------

@dataclass
class Ellipsoid:
    """Implicit ellipsoid ``{x : (C^-1 (x-d)).norm() <= 1}``.

    Mirrors ``Ellipsoid<3>`` in
    ``deps/DecompROS2/DecompUtil/include/decomp_geometry/ellipsoid.h``.
    """
    C: np.ndarray = field(default_factory=lambda: np.eye(3))
    d: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def dist(self, pt: np.ndarray) -> float:
        return float(np.linalg.norm(np.linalg.solve(self.C, pt - self.d)))

    def inside(self, pt: np.ndarray) -> bool:
        return self.dist(pt) <= 1.0

    def points_inside(self, obs: np.ndarray) -> np.ndarray:
        if obs.size == 0:
            return obs
        C_inv = np.linalg.inv(self.C)
        diffs = obs - self.d
        d = np.linalg.norm(diffs @ C_inv.T, axis=1)
        return obs[d <= 1.0]

    def closest_point(self, obs: np.ndarray) -> np.ndarray:
        """Point in ``obs`` with the smallest ``dist`` value. Caller must
        ensure ``obs`` is non-empty."""
        C_inv = np.linalg.inv(self.C)
        diffs = obs - self.d
        d = np.linalg.norm(diffs @ C_inv.T, axis=1)
        return obs[int(np.argmin(d))]

    def closest_hyperplane(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p, n)`` for the supporting hyperplane at the closest
        obstacle point. ``n`` points outward (away from the center)."""
        pw = self.closest_point(obs)
        C_inv = np.linalg.inv(self.C)
        # n direction = C^{-T} C^{-1} (pw - d) per geometric_utils.h
        n = C_inv.T @ (C_inv @ (pw - self.d))
        nn = float(np.linalg.norm(n))
        if nn < _EPS:
            # Degenerate — fall back to radial direction from center.
            v = pw - self.d
            nv = float(np.linalg.norm(v))
            n = v / max(nv, _EPS)
        else:
            n = n / nn
        return pw, n


@dataclass
class Hyperplane:
    """Plane ``{x : n.dot(x - p) = 0}`` with outward-pointing ``n``."""
    p: np.ndarray
    n: np.ndarray

    def signed_dist(self, pt: np.ndarray) -> float:
        return float(self.n @ (pt - self.p))


@dataclass
class Polyhedron:
    """Sequence of half-spaces ``{x : n_i.dot(x - p_i) <= 0}``."""
    vs: List[Hyperplane] = field(default_factory=list)

    def add(self, h: Hyperplane) -> None:
        self.vs.append(h)

    def signed_dists(self, pt: np.ndarray) -> np.ndarray:
        if not self.vs:
            return np.empty(0)
        return np.array([h.signed_dist(pt) for h in self.vs], dtype=float)

    def inside(self, pt: np.ndarray, eps: float = _EPS) -> bool:
        return bool(np.all(self.signed_dists(pt) <= eps))

    def points_inside(self, obs: np.ndarray) -> np.ndarray:
        if obs.size == 0 or not self.vs:
            return obs
        # signed_dist = N · (obs - P) = N · obs - N · P, vectorised.
        N = np.stack([h.n for h in self.vs], axis=0)   # (m, 3)
        P = np.stack([h.p for h in self.vs], axis=0)   # (m, 3)
        d = obs @ N.T - np.einsum("ij,ij->i", P, N)[None, :]
        mask = np.all(d <= _EPS, axis=1)
        return obs[mask]


def polyhedron_to_polytope(poly: Polyhedron, ref_inside_point: np.ndarray) -> Polytope:
    """Convert hyperplane list to ``Ax <= b`` form. Mirrors
    ``LinearConstraint`` in polyhedron.h: each hyperplane's normal is
    flipped if the reference point ends up on the wrong side."""
    if not poly.vs:
        return Polytope(A=np.zeros((0, 3)), b=np.zeros((0, 1)))
    A_rows: List[np.ndarray] = []
    b_vals: List[float] = []
    for h in poly.vs:
        n = h.n.copy()
        c = float(np.dot(h.p, n))
        if float(np.dot(n, ref_inside_point)) - c > 0:
            n = -n
            c = -c
        A_rows.append(n)
        b_vals.append(c)
    A = np.stack(A_rows, axis=0)
    b = np.array(b_vals, dtype=float).reshape(-1, 1)
    return Polytope(A=A, b=b)


# ---------------------------------------------------------------------------
# Rotation utility — vec3_to_rotation from geometric_utils.h
# ---------------------------------------------------------------------------

def vec3_to_rotation(v: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix aligning the x-axis with ``v``. Zero roll.

    Direct port of ``vec3_to_rotation`` in
    ``deps/DecompROS2/DecompUtil/include/decomp_geometry/geometric_utils.h``.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    n_xy = float(np.hypot(v[0], v[1]))
    pitch = float(np.arctan2(-v[2], n_xy))
    yaw = float(np.arctan2(v[1], v[0]))
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) * Ry(pitch)
    R = np.array([
        [cy * cp, -sy, cy * sp],
        [sy * cp,  cy, sy * sp],
        [-sp,      0.0, cp],
    ], dtype=float)
    return R


def quat_rotate_around_x(R: np.ndarray, roll: float) -> np.ndarray:
    """Equivalent to ``Ri * Quat(cos(roll/2), sin(roll/2), 0, 0)`` — apply
    a roll-around-x to a rotation matrix on the right. Used by the y-axis
    shrink loop in find_ellipsoid."""
    c, s = np.cos(roll), np.sin(roll)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ], dtype=float)
    return R @ Rx


# ---------------------------------------------------------------------------
# Ellipsoid construction — port of LineSegment<3>::find_ellipsoid
# ---------------------------------------------------------------------------

def _inflate_obstacles_in_ellipsoid_frame(
    obs_world: np.ndarray,
    Ri: np.ndarray,
    center: np.ndarray,
    inflate_distance: float,
) -> np.ndarray:
    """Move each obstacle point to the nearest vertex of a cube of half-
    width ``inflate_distance`` centred on the point, in the ellipsoid's
    local frame. This is the per-obstacle inflation used by SANDO instead
    of a radial sphere — see line_segment.h:191-203."""
    if obs_world.size == 0 or inflate_distance <= 0.0:
        return obs_world.copy()
    diffs = obs_world - center
    p_local = diffs @ Ri  # equivalent to (Ri.T @ diff) per row
    s = np.sign(p_local)
    s[s == 0] = 1.0  # treat exact zeros as positive — matches sgn(0)=0 behaviour
    # But the C++ sgn(0) returns 0, so the displacement would be 0. Mirror that:
    s_cpp = np.where(p_local > 0, 1.0, np.where(p_local < 0, -1.0, 0.0))
    new_local = p_local - s_cpp * inflate_distance
    return new_local @ Ri.T + center


def find_ellipsoid(
    p1: np.ndarray,
    p2: np.ndarray,
    obs_world: np.ndarray,
    offset_x: float,
    inflate_distance: float,
) -> Tuple[Ellipsoid, np.ndarray, np.ndarray, bool]:
    """Grow a maximum-volume ellipsoid around segment ``[p1, p2]`` that
    excludes all points in ``obs_world``. Returns
    ``(ellipsoid, Rf, inflated_obs, ok)`` — ``Rf`` is the final rotation
    used to align the ellipsoid's principal axes (needed by callers
    constructing local bboxes), ``inflated_obs`` is the post-inflation
    obstacle cloud (used by find_polyhedron), and ``ok=False`` signals
    that the iterative shrink loop did not converge within
    ``_COUNTER_MAX`` iterations.

    Port of ``LineSegment<3>::find_ellipsoid(offset_x, result)`` in
    line_segment.h:271-389.
    """
    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)
    obs_world = np.asarray(obs_world, dtype=float).reshape(-1, 3) if obs_world.size else np.zeros((0, 3))

    f = 0.5 * float(np.linalg.norm(p2 - p1))
    axes = np.array([f, f, f], dtype=float)
    axes[0] += offset_x
    if axes[0] > 0:
        ratio = axes[1] / axes[0]
        axes = axes * ratio

    Ri = vec3_to_rotation(p2 - p1)
    center = 0.5 * (p1 + p2)

    # Initial C in world frame: Ri * diag(axes) * Ri^T
    C = Ri @ np.diag(axes) @ Ri.T
    E = Ellipsoid(C=C, d=center)
    Rf = Ri.copy()

    # ---- Inflate obstacles in ellipsoid frame ----
    obs_inflated = _inflate_obstacles_in_ellipsoid_frame(
        obs_world, Ri, center, inflate_distance,
    )

    # ---- Vectorised obstacle filtering -----------------------------------
    # Both shrink loops repeatedly compute E.dist for the full obs_inside
    # array. The hot path used to do a Python-level
    # ``np.array([E.dist(pt) for pt in obs_inside])`` per iteration plus an
    # ``np.linalg.inv(self.C)`` inside every call — for 1000+ obstacles
    # this was the single slowest line in the planner. Inline the math:
    #   dist(pt) = || C^{-1} (pt - center) ||
    #   keep iff 1 - dist > epsilon  (i.e. strictly interior)
    # and reuse the cached ``Ri.T`` / ``Rf.T`` per loop iteration.
    if obs_inflated.size > 0:
        Ri_T = Ri.T
        diffs0 = obs_inflated - center
        # First pass: keep only those inside the initial ellipsoid.
        d0 = np.linalg.norm(diffs0 @ np.linalg.inv(C).T, axis=1)
        inside_mask = d0 <= 1.0
        obs_inside = obs_inflated[inside_mask]
    else:
        obs_inside = obs_inflated

    # ---- Shrink axes[1] (and mirror onto axes[2]) until clear ----
    counter = 0
    while obs_inside.size > 0:
        # Pick closest point — vectorised since we already have C^{-1}.
        C_inv = np.linalg.inv(E.C)
        diffs = obs_inside - center
        d_all = np.linalg.norm(diffs @ C_inv.T, axis=1)
        pw = obs_inside[int(np.argmin(d_all))]

        p_local = Ri_T @ (pw - center)
        roll = float(np.arctan2(p_local[2], p_local[1]))
        Rf = quat_rotate_around_x(Ri, roll)
        Rf_T = Rf.T
        p_local = Rf_T @ (pw - center)

        if p_local[0] < axes[0]:
            denom = max(_EPS, np.sqrt(max(0.0, 1.0 - (p_local[0] / axes[0]) ** 2)))
            axes[1] = abs(p_local[1]) / denom

        E.C = Rf @ np.diag([axes[0], axes[1], axes[1]]) @ Rf.T

        # Drop satisfied obstacles. Vectorised E.dist + filter.
        C_inv = np.linalg.inv(E.C)
        d_inside = np.linalg.norm((obs_inside - center) @ C_inv.T, axis=1)
        keep = (1.0 - d_inside) > _EPS
        obs_inside = obs_inside[keep]

        if counter > _COUNTER_MAX:
            return E, Rf, obs_inflated, False
        counter += 1

    # ---- Shrink axes[2] ----
    # Reset E with original axes[2] = f.
    axes[2] = f
    E.C = Rf @ np.diag(axes) @ Rf.T
    # Re-check obstacles inside the freshly-shaped ellipsoid (vectorised).
    if obs_inflated.size > 0:
        Rf_T = Rf.T
        C_inv = np.linalg.inv(E.C)
        d_all = np.linalg.norm((obs_inflated - center) @ C_inv.T, axis=1)
        obs_inside = obs_inflated[d_all <= 1.0]
    else:
        obs_inside = obs_inflated

    counter = 0
    while obs_inside.size > 0:
        C_inv = np.linalg.inv(E.C)
        diffs = obs_inside - center
        d_all = np.linalg.norm(diffs @ C_inv.T, axis=1)
        pw = obs_inside[int(np.argmin(d_all))]

        p_local = Rf_T @ (pw - center)
        dd = 1.0 - (p_local[0] / axes[0]) ** 2 - (p_local[1] / axes[1]) ** 2
        if dd > _EPS:
            axes[2] = abs(p_local[2]) / np.sqrt(dd)
        E.C = Rf @ np.diag(axes) @ Rf.T

        C_inv = np.linalg.inv(E.C)
        d_inside = np.linalg.norm((obs_inside - center) @ C_inv.T, axis=1)
        keep = (1.0 - d_inside) > _EPS
        obs_inside = obs_inside[keep]

        if counter > _COUNTER_MAX:
            return E, Rf, obs_inflated, False
        counter += 1

    return E, Rf, obs_inflated, True


# ---------------------------------------------------------------------------
# Polyhedron construction — port of DecompBase<3>::find_polyhedron
# ---------------------------------------------------------------------------

def find_polyhedron(E: Ellipsoid, obs: np.ndarray) -> Polyhedron:
    """Greedy half-space cut: repeatedly take the supporting hyperplane
    at the obstacle closest to ``E``, drop every obstacle on its outside,
    iterate until no obstacles remain.

    Mirrors ``DecompBase<3>::find_polyhedron`` in decomp_base.h:68-87.

    Optimisation note: the C matrix doesn't change while ``find_polyhedron``
    runs (the ellipsoid is fixed at this point), so the per-iteration
    ``np.linalg.inv(C)`` inside ``closest_hyperplane`` was redundant.
    Hoist it out and reuse the cached inverse throughout the cut loop.
    """
    poly = Polyhedron()
    obs_remain = np.asarray(obs, dtype=float).reshape(-1, 3) if obs.size else np.zeros((0, 3))
    if obs_remain.size == 0:
        return poly
    C_inv = np.linalg.inv(E.C)
    Q = C_inv.T @ C_inv  # used to map "closest point" → outward normal
    while obs_remain.size > 0:
        diffs = obs_remain - E.d
        d_all = np.linalg.norm(diffs @ C_inv.T, axis=1)
        i_min = int(np.argmin(d_all))
        pw = obs_remain[i_min]
        n = Q @ (pw - E.d)
        nn = float(np.linalg.norm(n))
        if nn < _EPS:
            v = pw - E.d
            nv = float(np.linalg.norm(v))
            n = v / max(nv, _EPS)
        else:
            n = n / nn
        poly.add(Hyperplane(p=pw, n=n))
        # Keep only obstacles still "inside" the half-space (signed_dist < 0).
        sd = (obs_remain - pw) @ n
        obs_remain = obs_remain[sd < 0]
    return poly


# ---------------------------------------------------------------------------
# Bounding boxes — port of LineSegment<3>::add_local_bbox and
# EllipsoidDecomp<3>::add_{z_min_and_max, global_bbox}
# ---------------------------------------------------------------------------

def add_local_bbox(
    poly: Polyhedron,
    p1: np.ndarray,
    p2: np.ndarray,
    local_bbox: Optional[np.ndarray],
) -> None:
    """Inject virtual walls parallel to the segment.

    ``local_bbox`` is the C++ ``local_bbox_`` field — a 3-vector of
    *half-widths* (x along segment, y perpendicular horizontal, z
    perpendicular vertical). A zero-norm bbox means "skip" (matches the
    C++ ``if (local_bbox_.norm() == 0) return`` guard)."""
    if local_bbox is None:
        return
    lb = np.asarray(local_bbox, dtype=float).reshape(3)
    if float(np.linalg.norm(lb)) == 0.0:
        return

    direction = p2 - p1
    nd = float(np.linalg.norm(direction))
    if nd < _EPS:
        return
    dir_ = direction / nd

    dir_h = np.array([dir_[1], -dir_[0], 0.0], dtype=float)
    nh = float(np.linalg.norm(dir_h))
    if nh == 0.0:
        # Vertical segment — fall back to -x. (C++ does the same:
        # dir_h << -1, 0, 0 in 3D.)
        dir_h = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        dir_h = dir_h / nh

    # x walls — perpendicular horizontal
    pp1 = p1 + dir_h * lb[1]
    pp2 = p1 - dir_h * lb[1]
    poly.add(Hyperplane(pp1, dir_h))
    poly.add(Hyperplane(pp2, -dir_h))

    # y walls — front/back along segment
    pp3 = p2 + dir_ * lb[0]
    pp4 = p1 - dir_ * lb[0]
    poly.add(Hyperplane(pp3, dir_))
    poly.add(Hyperplane(pp4, -dir_))

    # z walls — perpendicular vertical (dir × dir_h)
    dir_v = np.cross(dir_, dir_h)
    nv = float(np.linalg.norm(dir_v))
    if nv > _EPS:
        dir_v = dir_v / nv
        pp5 = p1 + dir_v * lb[2]
        pp6 = p1 - dir_v * lb[2]
        poly.add(Hyperplane(pp5, dir_v))
        poly.add(Hyperplane(pp6, -dir_v))


def add_z_min_and_max(poly: Polyhedron, z_min: float, z_max: float) -> None:
    """Add horizontal floor / ceiling planes."""
    if z_min == -np.inf and z_max == np.inf:
        return
    poly.add(Hyperplane(np.array([0.0, 0.0, z_max]), np.array([0.0, 0.0, 1.0])))
    poly.add(Hyperplane(np.array([0.0, 0.0, z_min]), np.array([0.0, 0.0, -1.0])))


def add_global_bbox(
    poly: Polyhedron,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> None:
    """Add axis-aligned world bounds. Matches the 3D branch of
    ``EllipsoidDecomp<3>::add_global_bbox``."""
    # +z / -z first to mirror C++ ordering
    poly.add(Hyperplane(np.array([0.0, 0.0, bbox_max[2]]), np.array([0.0, 0.0, 1.0])))
    poly.add(Hyperplane(np.array([0.0, 0.0, bbox_min[2]]), np.array([0.0, 0.0, -1.0])))
    poly.add(Hyperplane(np.array([bbox_max[0], 0.0, 0.0]), np.array([1.0, 0.0, 0.0])))
    poly.add(Hyperplane(np.array([bbox_min[0], 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])))
    poly.add(Hyperplane(np.array([0.0, bbox_max[1], 0.0]), np.array([0.0, 1.0, 0.0])))
    # NOTE: C++ has a bug here — second y-bound uses bbox_max(1) but
    # normal -y. Faithfully port it so behaviour matches.
    poly.add(Hyperplane(np.array([0.0, bbox_max[1], 0.0]), np.array([0.0, -1.0, 0.0])))


# ---------------------------------------------------------------------------
# Public API — EllipsoidDecomp3D
# ---------------------------------------------------------------------------

@dataclass
class EllipsoidDecompResult:
    """Per-segment decomposition outputs — kept for parity with the
    DecompROS2 API (``get_ellipsoids``, ``get_polyhedrons``,
    ``get_constraints``)."""
    ellipsoid: Ellipsoid
    polyhedron: Polyhedron
    polytope: Polytope
    converged: bool


def decompose_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    obstacles: np.ndarray,
    *,
    offset_x: float = 0.0,
    inflate_distance: float = 0.0,
    local_bbox: Optional[np.ndarray] = None,
    z_min: float = -np.inf,
    z_max: float = np.inf,
    global_bbox_min: Optional[np.ndarray] = None,
    global_bbox_max: Optional[np.ndarray] = None,
) -> EllipsoidDecompResult:
    """Single-segment EllipsoidDecomp.

    Args:
        p1, p2: segment endpoints in the world frame.
        obstacles: ``(N, 3)`` array of obstacle points (already cropped to
            something reasonable around the segment — DecompROS2 also
            internally crops with the local_bbox, but it's still
            beneficial to pre-filter for speed).
        offset_x: extension along the long semi-axis (default 0).
        inflate_distance: per-axis cube inflation applied to obstacles in
            the ellipsoid frame (drone-radius-like padding). Default 0.
        local_bbox: 3-vector of half-widths for the virtual walls aligned
            with the segment. ``None`` or all-zeros to skip.
        z_min, z_max: world altitude bounds (added as horizontal walls).
            Use ``-inf`` / ``+inf`` to skip.
        global_bbox_min, global_bbox_max: world AABB walls. ``None`` to
            skip (matches the C++ ``norm() == 0`` guard).
    """
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 3) if obstacles.size else np.zeros((0, 3))

    # Pre-crop obstacles by local_bbox (mirrors DecompBase::set_obs).
    if local_bbox is not None:
        bbox_poly = Polyhedron()
        add_local_bbox(bbox_poly, p1, p2, local_bbox)
        if bbox_poly.vs:
            obstacles = bbox_poly.points_inside(obstacles)

    E, Rf, obs_inflated, ok = find_ellipsoid(
        p1, p2, obstacles, offset_x=offset_x,
        inflate_distance=inflate_distance,
    )
    poly = find_polyhedron(E, obs_inflated)
    add_local_bbox(poly, p1, p2, local_bbox)
    if z_min != -np.inf or z_max != np.inf:
        add_z_min_and_max(poly, z_min, z_max)
    if (global_bbox_min is not None and global_bbox_max is not None
            and (float(np.linalg.norm(global_bbox_min)) != 0.0
                 or float(np.linalg.norm(global_bbox_max)) != 0.0)):
        add_global_bbox(poly, global_bbox_min, global_bbox_max)

    midpoint = 0.5 * (p1 + p2)
    polytope = polyhedron_to_polytope(poly, midpoint)
    return EllipsoidDecompResult(
        ellipsoid=E, polyhedron=poly, polytope=polytope, converged=ok,
    )


def decompose_path(
    path: List[np.ndarray],
    obstacles: np.ndarray,
    *,
    offset_x: float = 0.0,
    inflate_distance: float = 0.0,
    local_bbox: Optional[np.ndarray] = None,
    z_min: float = -np.inf,
    z_max: float = np.inf,
    global_bbox_min: Optional[np.ndarray] = None,
    global_bbox_max: Optional[np.ndarray] = None,
) -> List[EllipsoidDecompResult]:
    """Run ``decompose_segment`` over each ``[path[i], path[i+1]]``."""
    if path is None or len(path) < 2:
        return []
    return [
        decompose_segment(
            path[i], path[i + 1], obstacles,
            offset_x=offset_x, inflate_distance=inflate_distance,
            local_bbox=local_bbox, z_min=z_min, z_max=z_max,
            global_bbox_min=global_bbox_min, global_bbox_max=global_bbox_max,
        )
        for i in range(len(path) - 1)
    ]
