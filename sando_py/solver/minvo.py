"""MINVO control-point conversion for cubic trajectories.

The MINVO basis encloses a polynomial in its minimum-volume simplex —
i.e. the tightest convex hull of the curve. For a degree-d polynomial
over [0, T], MINVO has d+1 control points that form a hull strictly
inside the Bezier hull. SANDO uses MINVO control points for the
velocity / acceleration dynamic constraints so that ``|v| <= v_max``
is enforced *tightly* rather than via slack-y sampling.

The transforms below are hard-coded from the C++ ``BasisConverter``
output (16-digit precision):
    sando_ws/src/sando/include/sando/sando_type.hpp:235-258

Notation
--------
For a cubic ``p(u) = a u^3 + b u^2 + c u + d`` over ``u ∈ [0, 1]``:
  - position MINVO has 4 CPs (matrix shape 4×4).
  - velocity ``v(u) = 3 a u^2 + 2 b u + c`` (degree 2) → 3 CPs.
  - acceleration ``acc(u) = 6 a u + 2 b`` (degree 1) → 2 CPs.

Normalised coefficients (so the curve is over u ∈ [0, 1] regardless of
the physical segment duration T):
    A_n = a * T^3,   B_n = b * T^2,   C_n = c * T,   D_n = d.
"""

from __future__ import annotations

import numpy as np


# Position basis (4×4): MINVO CP_i = sum_k A_pos_mv_rest_inv_[k, i] * [A_n, B_n, C_n, D_n]_k
# Values lifted from the C++ ``BasisConverter`` output reported by the
# audit. Reference: SANDO paper + sando_type.hpp:235-245.
A_POS_MV_REST_INV: np.ndarray = np.array([
    [-0.03203277, -0.09273093,  0.34205725,  1.10233139],
    [-0.05111494, -0.04627261,  0.54582349,  1.09798069],
    [-0.07454782,  0.20395195,  0.79604805,  1.07454782],
    [ 1.00000000,  1.00000000,  1.00000000,  1.00000000],
], dtype=float)

# Velocity basis (3×3): MINVO velocity CP_i = sum_k A_vel_mv_rest_inv_[k, i] * [3*A_n, 2*B_n, C_n]_k
A_VEL_MV_REST_INV: np.ndarray = np.array([
    [-0.07735027, 0.16666667, 1.07735027],
    [-0.07735027, 0.50000000, 1.07735027],
    [ 1.00000000, 1.00000000, 1.00000000],
], dtype=float)

# Acceleration basis (2×2): MINVO accel CP_i = sum_k A_accel_mv_rest_inv_[k, i] * [6*A_n, 2*B_n]_k
A_ACCEL_MV_REST_INV: np.ndarray = np.array([
    [0.0, 1.0],
    [1.0, 1.0],
], dtype=float)


def pos_minvo_cps(
    a: "LinearCoeff", b: "LinearCoeff", c: "LinearCoeff", d: "LinearCoeff",
    T: float,
):
    """Return 4 position MINVO control points (each a ``LinearCoeff``)
    for segment with normalized coefficients A_n=a*T^3, B_n=b*T^2,
    C_n=c*T, D_n=d.

    Output shape: list of 4 LinearCoeff (one per CP index in the MINVO
    basis). Each CP is a linear combination of free vars + boundary
    contributions.
    """
    A_n = a * (T ** 3)
    B_n = b * (T ** 2)
    C_n = c * T
    D_n = d
    cps = []
    for i in range(4):
        m = A_POS_MV_REST_INV[:, i]
        cps.append(
            A_n * float(m[0])
            + B_n * float(m[1])
            + C_n * float(m[2])
            + D_n * float(m[3])
        )
    return cps


def vel_minvo_cps(
    a: "LinearCoeff", b: "LinearCoeff", c: "LinearCoeff",
    T: float,
):
    """Return 3 velocity MINVO control points. The velocity polynomial
    in normalised time ``u`` is ``3 A_n u^2 + 2 B_n u + C_n``."""
    # Normalised velocity coefficients (degree 2):
    A_v = a * (3 * T ** 2)
    B_v = b * (2 * T)
    C_v = c * 1.0
    cps = []
    for i in range(3):
        m = A_VEL_MV_REST_INV[:, i]
        cps.append(A_v * float(m[0]) + B_v * float(m[1]) + C_v * float(m[2]))
    return cps


def accel_minvo_cps(a: "LinearCoeff", b: "LinearCoeff", T: float):
    """Return 2 acceleration MINVO control points. The acceleration
    polynomial in normalised time ``u`` is ``6 A_n u + 2 B_n``."""
    A_a = a * (6 * T)
    B_a = b * 2.0
    cps = []
    for i in range(2):
        m = A_ACCEL_MV_REST_INV[:, i]
        cps.append(A_a * float(m[0]) + B_a * float(m[1]))
    return cps


def jerk_value(a: "LinearCoeff"):
    """Jerk is constant over a cubic segment: ``j = 6 a`` (unscaled)."""
    return a * 6.0
