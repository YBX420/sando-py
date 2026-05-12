"""Tiny generic helpers ported from include/hgp/data_utils.hpp.

C++ ``data_type.hpp`` is a wall of Eigen type aliases (Vec3f, Vec3i, Mat3f, …).
All of those collapse onto ``np.ndarray`` in Python, so we omit that file.
"""

from __future__ import annotations

from typing import List

import numpy as np


# ANSI color codes (mirrors data_type.hpp:26-52)
ANSI_COLOR_RED     = "\x1b[1;31m"
ANSI_COLOR_GREEN   = "\x1b[1;32m"
ANSI_COLOR_YELLOW  = "\x1b[1;33m"
ANSI_COLOR_BLUE    = "\x1b[1;34m"
ANSI_COLOR_MAGENTA = "\x1b[1;35m"
ANSI_COLOR_CYAN    = "\x1b[1;36m"
ANSI_COLOR_RESET   = "\x1b[0m"


def transform_vec(t: List[np.ndarray], tf: np.ndarray) -> List[np.ndarray]:
    """Apply ``tf`` (matrix or callable) to each element of ``t``. (data_utils.hpp:22)"""
    if callable(tf):
        return [tf(v) for v in t]
    return [tf @ v for v in t]


def total_distance(vs: List[np.ndarray]) -> float:
    """Sum of consecutive Euclidean distances. (data_utils.hpp:30)"""
    if len(vs) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(vs)):
        dist += float(np.linalg.norm(vs[i] - vs[i - 1]))
    return dist
