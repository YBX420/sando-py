"""Piecewise cubic polynomial trajectory.

Faithful port of sando_type.hpp:545 (struct PieceWisePol).

Interval i spans [times[i], times[i+1]) and is parameterized by
    u = t - times[i],     pol(t) = coeff * [u^3, u^2, u, 1]^T.

n intervals, n+1 time knots, coeff_{x,y,z}[i] is a length-4 array [a, b, c, d].
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class PieceWisePol:
    # times has n+1 elements                       (sando_type.hpp:555)
    times: List[float] = field(default_factory=list)

    # coeff_{x,y,z} have n elements each           (sando_type.hpp:560-562)
    # each element is a length-4 array [a, b, c, d]
    coeff_x: List[np.ndarray] = field(default_factory=list)
    coeff_y: List[np.ndarray] = field(default_factory=list)
    coeff_z: List[np.ndarray] = field(default_factory=list)

    # --- sando_type.hpp:565 ---
    def getDuration(self) -> float:
        return self.times[-1] - self.times[0]

    # --- sando_type.hpp:568 ---
    def clear(self) -> None:
        self.times.clear()
        self.coeff_x.clear()
        self.coeff_y.clear()
        self.coeff_z.clear()

    # --- sando_type.hpp:576 ---
    def getEndTime(self) -> float:
        return self.times[-1]

    def _segment_index(self, t: float) -> int:
        """Return the segment index containing ``t``. ``bisect`` is
        O(log N) where N is the number of segments; the previous linear
        scan was O(N) per call and dominated goal-publish (100 Hz × 3
        eval flavours = 300 lookups/sec)."""
        n = len(self.times)
        if n < 2:
            return 0
        if t >= self.times[-1]:
            return n - 2
        if t < self.times[0]:
            return 0
        # bisect_right(times, t) returns the first index where times[i] > t;
        # the containing segment is at i - 1.
        return max(0, min(n - 2, bisect.bisect_right(self.times, t) - 1))

    # --- sando_type.hpp:582  inline Eigen::Vector3d eval(double t) const ---
    def eval(self, t: float) -> np.ndarray:
        i = self._segment_index(t)
        if t >= self.times[-1]:
            u = self.times[-1] - self.times[i]
        elif t < self.times[0]:
            u = 0.0
        else:
            u = t - self.times[i]
        tmp = np.array([u * u * u, u * u, u, 1.0])
        return np.array([
            float(np.dot(self.coeff_x[i], tmp)),
            float(np.dot(self.coeff_y[i], tmp)),
            float(np.dot(self.coeff_z[i], tmp)),
        ])

    # --- sando_type.hpp:631  inline Eigen::Vector3d velocity(double t) const ---
    def velocity(self, t: float) -> np.ndarray:
        i = self._segment_index(t)
        if t < self.times[0]:
            return np.array([
                self.coeff_x[i][2], self.coeff_y[i][2], self.coeff_z[i][2],
            ])
        if t >= self.times[-1]:
            u = self.times[-1] - self.times[i]
        else:
            u = t - self.times[i]
        cx, cy, cz = self.coeff_x[i], self.coeff_y[i], self.coeff_z[i]
        return np.array([
            3 * cx[0] * u * u + 2 * cx[1] * u + cx[2],
            3 * cy[0] * u * u + 2 * cy[1] * u + cy[2],
            3 * cz[0] * u * u + 2 * cz[1] * u + cz[2],
        ])

    # --- sando_type.hpp:669  inline Eigen::Vector3d acceleration(double t) const ---
    def acceleration(self, t: float) -> np.ndarray:
        i = self._segment_index(t)
        if t < self.times[0]:
            return np.array([
                2.0 * self.coeff_x[i][1],
                2.0 * self.coeff_y[i][1],
                2.0 * self.coeff_z[i][1],
            ])
        if t >= self.times[-1]:
            u = self.times[-1] - self.times[i]
        else:
            u = t - self.times[i]
        cx, cy, cz = self.coeff_x[i], self.coeff_y[i], self.coeff_z[i]
        return np.array([
            6.0 * cx[0] * u + 2.0 * cx[1],
            6.0 * cy[0] * u + 2.0 * cy[1],
            6.0 * cz[0] * u + 2.0 * cz[1],
        ])

    # --- sando_type.hpp:704 ---
    def print(self) -> None:
        print(f"coeff_x.size()= {len(self.coeff_x)}")
        print(f"times.size()= {len(self.times)}")
        print("Note that coeff_x.size() == times.size()-1")
        for t in self.times:
            print(f"Time: {t:f}")
        for i in range(len(self.times) - 1):
            print(f"From {self.times[i]} to {self.times[i + 1]}")
            print(f"  Coeff_x= {self.coeff_x[i]}")
            print(f"  Coeff_y= {self.coeff_y[i]}")
            print(f"  Coeff_z= {self.coeff_z[i]}")
