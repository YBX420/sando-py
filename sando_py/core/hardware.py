"""SE(3) frame transform for hardware deployments.

Faithful port of ``SANDO::setInitialPose`` /
``applyInitiPoseTransform`` / ``applyInitiPoseInverseTransform`` in
sando.cpp:2207-2308. Used when ``use_hardware`` is true and the
hardware delivers state / goals in a global (e.g. Vicon) frame that
needs to be converted to the planner's local frame, and vice-versa for
published goals.

The actual ``setInitialPose`` is typically called once at boot from a
``geometry_msgs/TransformStamped`` over a service or topic — the
hardware launcher publishes the Vicon mount pose. Until then,
``init_pose_set_`` is False and all the transform functions are no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Normalised quaternion → 3x3 rotation matrix. Mirrors
    ``Eigen::Quaterniond::toRotationMatrix`` (sando.cpp:2221)."""
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)
    return R


@dataclass
class InitPoseTransform:
    """Forward (local→global) and inverse (global→local) SE(3) transforms
    plus the yaw offset, in the same layout C++ keeps them.

    ``set_`` flips to True once :meth:`from_transform` has been called —
    matching the ``init_pose_set_`` flag in sando.hpp. Until then all
    methods are no-ops.
    """
    forward: np.ndarray = field(default_factory=lambda: np.eye(4))
    inverse: np.ndarray = field(default_factory=lambda: np.eye(4))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_inv: np.ndarray = field(default_factory=lambda: np.eye(3))
    yaw_offset: float = 0.0
    set_: bool = False

    @classmethod
    def from_transform(
        cls,
        tx: float, ty: float, tz: float,
        qw: float, qx: float, qy: float, qz: float,
    ) -> "InitPoseTransform":
        """Build from quaternion (w, x, y, z) + translation."""
        R = _quat_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=float)
        forward = np.eye(4)
        forward[:3, :3] = R
        forward[:3, 3] = t
        inverse = np.eye(4)
        inverse[:3, :3] = R.T
        inverse[:3, 3] = -R.T @ t
        yaw_offset = float(np.arctan2(R[1, 0], R[0, 0]))
        return cls(
            forward=forward, inverse=inverse,
            R=R, R_inv=R.T,
            yaw_offset=yaw_offset, set_=True,
        )

    # ------------------------------------------------------------------
    # State transforms (incoming local → planner-global; outgoing
    # planner-global → local). Mirrors the C++ ``updateState`` /
    # ``getNextGoal`` transform blocks (sando.cpp:1525-1547, 1648-1667).
    # ------------------------------------------------------------------

    def apply_state_local_to_global(self, pos: np.ndarray, vel: np.ndarray,
                                    accel: np.ndarray, jerk: np.ndarray,
                                    yaw: float) -> tuple:
        """Convert a state arriving in the local (e.g. MAV body) frame to
        the planner's global frame. Returns ``(pos, vel, accel, jerk, yaw)``."""
        if not self.set_:
            return pos, vel, accel, jerk, yaw
        homo = np.array([pos[0], pos[1], pos[2], 1.0], dtype=float)
        global_pos = self.forward @ homo
        return (
            global_pos[:3],
            self.R @ vel,
            self.R @ accel,
            self.R @ jerk,
            yaw + self.yaw_offset,
        )

    def apply_state_global_to_local(self, pos: np.ndarray, vel: np.ndarray,
                                    accel: np.ndarray, jerk: np.ndarray,
                                    yaw: float) -> tuple:
        """Convert a planner-global state into the local (MAVROS) frame
        for publishing. Returns ``(pos, vel, accel, jerk, yaw)`` with
        ``yaw`` wrapped to ``[-pi, pi]``."""
        if not self.set_:
            return pos, vel, accel, jerk, yaw
        homo = np.array([pos[0], pos[1], pos[2], 1.0], dtype=float)
        local_pos = self.inverse @ homo
        wrapped = float((yaw - self.yaw_offset + np.pi) % (2 * np.pi) - np.pi)
        return (
            local_pos[:3],
            self.R_inv @ vel,
            self.R_inv @ accel,
            self.R_inv @ jerk,
            wrapped,
        )

    # ------------------------------------------------------------------
    # PieceWisePol transforms — faithful port of the C++ coefficient
    # transformations (sando.cpp:2264-2308). Each polynomial coefficient
    # is treated as a 4-homogeneous point [c_x, c_y, c_z, 1]^T and
    # multiplied by the SE(3) matrix. (The C++ also applies the
    # translation to higher-order coefficients — port that behaviour
    # exactly even though it produces non-rigid transformations of the
    # polynomial.)
    # ------------------------------------------------------------------

    def apply_to_pwp(self, pwp) -> None:
        """In-place forward (local→global) transform of ``pwp``."""
        if not self.set_:
            return
        self._apply_pwp(pwp, self.forward)

    def apply_inverse_to_pwp(self, pwp) -> None:
        """In-place inverse (global→local) transform of ``pwp``."""
        if not self.set_:
            return
        self._apply_pwp(pwp, self.inverse)

    @staticmethod
    def _apply_pwp(pwp, M: np.ndarray) -> None:
        for i in range(len(pwp.coeff_x)):
            cx, cy, cz = pwp.coeff_x[i], pwp.coeff_y[i], pwp.coeff_z[i]
            for j in range(len(cx)):
                v = np.array([float(cx[j]), float(cy[j]), float(cz[j]), 1.0])
                v = M @ v
                cx[j] = v[0]
                cy[j] = v[1]
                cz[j] = v[2]


# ---------------------------------------------------------------------------
# Convenience: build from a geometry_msgs/TransformStamped (only used in
# the ROS-env tests; conftest skips the ROS-typed branch in non-ROS env).
# ---------------------------------------------------------------------------

def init_pose_from_transform_stamped(msg) -> Optional[InitPoseTransform]:
    """Build an ``InitPoseTransform`` from a ROS ``TransformStamped``
    message. Returns ``None`` if the message has an obvious all-zero
    quaternion (which would indicate a publisher that hasn't been
    initialised yet)."""
    q = msg.transform.rotation
    t = msg.transform.translation
    if abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z) < 1e-9:
        return None
    return InitPoseTransform.from_transform(
        float(t.x), float(t.y), float(t.z),
        float(q.w), float(q.x), float(q.y), float(q.z),
    )
