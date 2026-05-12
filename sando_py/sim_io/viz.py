"""ROS 2 visualization helpers — pure-Python types → MarkerArray /
PolyhedronArray messages for the RViz plugins the upstream demo uses.

Imports ``visualization_msgs``, ``std_msgs``, ``geometry_msgs``, and
``decomp_ros_msgs`` at module top — must be loaded inside a sourced ROS
2 environment.

Mirrors the C++ helpers in ``hgp/utils.cpp``
(``pathLineDotsToMarkerArray``, ``stateVector2ColoredMarkerArray``,
``getColorJet``) and ``DecompROS::polyhedron_array_to_ros``.

Conversions
-----------
* ``path_to_marker_array``           — polyline (LINE_STRIP + SPHERE_LIST)
* ``polytopes_to_polyhedron_array``  — ``Ax <= b`` corridors -> ROS msg
* ``pwp_to_colored_marker_array``    — PieceWisePol sampled, coloured by |v|
* ``positions_to_marker_array``      — history polyline (for ``actual_traj``)
* ``delete_all_marker``              — single Marker with DELETEALL action
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

# ROS imports — only available inside a sourced ROS 2 environment.
from builtin_interfaces.msg import Duration as DurationMsg
from decomp_ros_msgs.msg import (
    Polyhedron as PolyhedronMsg,
    PolyhedronArray as PolyhedronArrayMsg,
)
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from sando_py.core import PieceWisePol, Polytope


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def color_rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def color_jet(value: float, vmin: float, vmax: float) -> ColorRGBA:
    """Faithful port of ``getColorJet`` in ``hgp/utils.cpp``. Maps a
    scalar in ``[vmin, vmax]`` to an RGBA Jet-colormap entry."""
    v = max(vmin, min(vmax, float(value)))
    dv = vmax - vmin
    if dv <= 1e-9:
        return color_rgba(0.8, 0.0, 0.0)
    t = (v - vmin) / dv
    # Standard 4-band Jet:   blue → cyan → yellow → red
    if t < 0.25:
        r, g, b = 0.0, 4.0 * t, 1.0
    elif t < 0.5:
        r, g, b = 0.0, 1.0, 1.0 + 4.0 * (0.25 - t)
    elif t < 0.75:
        r, g, b = 4.0 * (t - 0.5), 1.0, 0.0
    else:
        r, g, b = 1.0, 1.0 + 4.0 * (0.75 - t), 0.0
    return color_rgba(r, g, b)


# ---------------------------------------------------------------------------
# Path -> MarkerArray (LINE_STRIP + SPHERE_LIST)
# ---------------------------------------------------------------------------

def path_to_marker_array(
    path: Sequence[np.ndarray],
    *,
    frame_id: str,
    stamp,
    color: ColorRGBA = None,
    line_width: float = 0.03,
    dot_diameter: float = 0.06,
    base_id: int = 50_000,
    lifetime_s: float = 1.0,
    ns: str = "global_path",
) -> MarkerArray:
    """Polyline as a LINE_STRIP + SPHERE_LIST pair. Returns an empty
    MarkerArray if ``path`` is empty.

    Mirrors ``pathLineDotsToMarkerArray`` (``hgp/utils.cpp:61``).
    """
    if color is None:
        color = color_rgba(1.0, 0.0, 0.0)
    ma = MarkerArray()
    if not path:
        return ma

    points = [_array_to_point(p) for p in path]
    lifetime = _seconds_to_duration(lifetime_s)

    line = Marker()
    line.header.frame_id = frame_id
    line.header.stamp = stamp
    line.ns = ns + "_line"
    line.id = base_id + 1
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.lifetime = lifetime
    line.scale.x = float(line_width)
    line.color = color
    line.pose.orientation.w = 1.0
    line.points = list(points)
    ma.markers.append(line)

    dots = Marker()
    dots.header.frame_id = frame_id
    dots.header.stamp = stamp
    dots.ns = ns + "_dots"
    dots.id = base_id + 2
    dots.type = Marker.SPHERE_LIST
    dots.action = Marker.ADD
    dots.lifetime = lifetime
    dots.scale.x = dots.scale.y = dots.scale.z = float(dot_diameter)
    dots.color = color
    dots.pose.orientation.w = 1.0
    dots.points = list(points)
    ma.markers.append(dots)
    return ma


# ---------------------------------------------------------------------------
# Polytopes -> PolyhedronArray
# ---------------------------------------------------------------------------

def polytopes_to_polyhedron_array(
    polys: Iterable[Polytope],
    *,
    frame_id: str,
    stamp,
    lifetime_s: float = 1.0,
) -> PolyhedronArrayMsg:
    """Each ``Polytope`` (``A x <= b``) becomes one ``Polyhedron`` message
    whose ``points`` / ``normals`` arrays carry one entry per half-space:
    a point on the plane and its outward-facing normal.

    Mirrors ``DecompROS::polyhedron_to_ros`` and
    ``polyhedron_array_to_ros``: a hyperplane is encoded as
    ``n · (x - p) <= 0`` which is equivalent to ``A_row · x <= b`` with
    ``n = A_row`` and ``p = A_row · b / ||A_row||²``.
    """
    out = PolyhedronArrayMsg()
    out.header.frame_id = frame_id
    out.header.stamp = stamp
    out.lifetime = _seconds_to_duration(lifetime_s)

    for poly in polys:
        if poly.A.size == 0:
            continue
        ph = PolyhedronMsg()
        A = poly.A
        b = poly.b.reshape(-1)
        for row in range(A.shape[0]):
            a_row = A[row]
            b_val = float(b[row])
            denom = float(a_row @ a_row)
            if denom <= 1e-18:
                continue
            # Point on the plane: projection of the origin
            p = a_row * (b_val / denom)
            n = a_row  # outward normal (RViz only uses direction)
            ph.points.append(_array_to_point(p))
            ph.normals.append(_array_to_point(n))
        if ph.points:
            out.polyhedrons.append(ph)
    return out


# ---------------------------------------------------------------------------
# PieceWisePol -> coloured MarkerArray
# ---------------------------------------------------------------------------

def pwp_to_colored_marker_array(
    pwp: PieceWisePol,
    *,
    frame_id: str,
    stamp,
    v_max: float,
    num_samples: int = 100,
    marker_id: int = 1,
    line_width: float = 0.15,
    ns: str = "state_vec",
    lifetime_s: float = 1.0,
) -> MarkerArray:
    """Sample the trajectory at ``num_samples`` points and emit a
    LINE_LIST whose colour is JET-mapped by ``|velocity|`` in
    ``[0, v_max]``. Mirrors ``stateVector2ColoredMarkerArray``
    (``hgp/utils.cpp:804``)."""
    ma = MarkerArray()
    if not pwp.times or len(pwp.times) < 2:
        return ma

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(marker_id)
    m.action = Marker.ADD
    m.type = Marker.LINE_LIST
    m.lifetime = _seconds_to_duration(lifetime_s)
    m.scale.x = float(line_width)
    m.pose.orientation.w = 1.0

    t0, t1 = pwp.times[0], pwp.times[-1]
    ts = np.linspace(t0, t1, max(2, int(num_samples)))
    p_last = _array_to_point(pwp.eval(t0))
    for t in ts[1:]:
        p_now = _array_to_point(pwp.eval(t))
        vel_norm = float(np.linalg.norm(pwp.velocity(t)))
        c = color_jet(vel_norm, 0.0, max(1e-6, float(v_max)))
        m.points.append(p_last)
        m.colors.append(c)
        m.points.append(p_now)
        m.colors.append(c)
        p_last = p_now

    ma.markers.append(m)
    return ma


# ---------------------------------------------------------------------------
# Position history -> MarkerArray (LINE_STRIP)
# ---------------------------------------------------------------------------

def positions_to_marker_array(
    positions: Sequence[np.ndarray],
    *,
    frame_id: str,
    stamp,
    color: ColorRGBA = None,
    line_width: float = 0.05,
    marker_id: int = 0,
    ns: str = "actual_traj",
    lifetime_s: float = 0.0,
) -> MarkerArray:
    """Append-only history polyline for ``actual_traj``. ``lifetime_s=0``
    means persist until overwritten (the C++ also uses that for the
    running actual-trajectory display)."""
    if color is None:
        color = color_rgba(0.0, 1.0, 0.0)
    ma = MarkerArray()
    if len(positions) < 2:
        return ma
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(marker_id)
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.lifetime = _seconds_to_duration(lifetime_s)
    m.scale.x = float(line_width)
    m.color = color
    m.pose.orientation.w = 1.0
    m.points = [_array_to_point(p) for p in positions]
    ma.markers.append(m)
    return ma


# ---------------------------------------------------------------------------
# DELETEALL
# ---------------------------------------------------------------------------

def delete_all_marker_array(*, frame_id: str, stamp) -> MarkerArray:
    """Single-marker MarkerArray with the DELETEALL action — cleans up
    stale markers in RViz before publishing a fresh set."""
    ma = MarkerArray()
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.action = Marker.DELETEALL
    ma.markers.append(m)
    return ma


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _array_to_point(arr) -> Point:
    a = np.asarray(arr, dtype=float).reshape(3)
    p = Point()
    p.x, p.y, p.z = float(a[0]), float(a[1]), float(a[2])
    return p


def _seconds_to_duration(seconds: float) -> DurationMsg:
    d = DurationMsg()
    sec = int(seconds)
    nsec = int(round((seconds - sec) * 1e9))
    d.sec, d.nanosec = sec, max(0, nsec)
    return d
