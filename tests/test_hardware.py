"""Regression tests for :mod:`sando_py.sim_io.hardware` — the SE(3)
hardware-mode transforms.

The transforms are pure-Python (no ROS imports) so they exercise in the
sando env without rclpy.
"""

from __future__ import annotations

import numpy as np

from sando_py.core import PieceWisePol
from sando_py.core.hardware import InitPoseTransform


def _yaw_quat(yaw: float):
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def test_init_pose_default_is_identity_noop():
    x = InitPoseTransform()
    assert not x.set_
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.0, 0.0, 0.0])
    out = x.apply_state_local_to_global(pos, vel, vel, vel, 0.7)
    # Default is identity; values returned unchanged.
    assert np.allclose(out[0], pos)
    assert out[4] == 0.7


def test_init_pose_yaw_offset_extracted_from_rotation():
    qw, qx, qy, qz = _yaw_quat(0.5)
    x = InitPoseTransform.from_transform(1.0, 2.0, 3.0, qw, qx, qy, qz)
    assert x.set_
    assert abs(x.yaw_offset - 0.5) < 1e-9


def test_init_pose_state_local_global_roundtrip():
    qw, qx, qy, qz = _yaw_quat(0.5)
    x = InitPoseTransform.from_transform(1.0, 2.0, 0.0, qw, qx, qy, qz)
    pos = np.array([3.0, 4.0, 5.0])
    vel = np.array([1.0, 0.0, 0.0])
    pos_g, vel_g, _, _, yaw_g = x.apply_state_local_to_global(
        pos, vel, np.zeros(3), np.zeros(3), 0.0,
    )
    pos_l, vel_l, _, _, yaw_l = x.apply_state_global_to_local(
        pos_g, vel_g, np.zeros(3), np.zeros(3), yaw_g,
    )
    assert np.allclose(pos_l, pos, atol=1e-9)
    assert np.allclose(vel_l, vel, atol=1e-9)
    assert abs(yaw_l) < 1e-9


def test_init_pose_pwp_transform_roundtrip():
    """Applying forward then inverse to a PieceWisePol should be a
    no-op (within float precision)."""
    pwp = PieceWisePol(
        times=[0.0, 1.0],
        coeff_x=[np.array([1.0, 2.0, 3.0, 4.0])],
        coeff_y=[np.array([0.0, -1.0, 0.0, 5.0])],
        coeff_z=[np.array([0.5, 0.0, 0.0, 6.0])],
    )
    qw, qx, qy, qz = _yaw_quat(0.3)
    x = InitPoseTransform.from_transform(1.0, -2.0, 0.5, qw, qx, qy, qz)
    cx0 = pwp.coeff_x[0].copy()
    cy0 = pwp.coeff_y[0].copy()
    cz0 = pwp.coeff_z[0].copy()
    x.apply_to_pwp(pwp)
    x.apply_inverse_to_pwp(pwp)
    assert np.allclose(pwp.coeff_x[0], cx0, atol=1e-9)
    assert np.allclose(pwp.coeff_y[0], cy0, atol=1e-9)
    assert np.allclose(pwp.coeff_z[0], cz0, atol=1e-9)


def test_init_pose_skipped_when_not_set():
    """When ``set_`` is False, both forward and inverse PWP transforms
    are no-ops."""
    pwp = PieceWisePol(
        times=[0.0, 1.0],
        coeff_x=[np.array([1.0, 2.0, 3.0, 4.0])],
        coeff_y=[np.array([0.0, -1.0, 0.0, 5.0])],
        coeff_z=[np.array([0.5, 0.0, 0.0, 6.0])],
    )
    x = InitPoseTransform()
    cx0 = pwp.coeff_x[0].copy()
    x.apply_to_pwp(pwp)
    x.apply_inverse_to_pwp(pwp)
    assert np.allclose(pwp.coeff_x[0], cx0)
