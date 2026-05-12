"""DynTraj — dynamic trajectory with Piecewise or Analytic mode.

Faithful port of sando_type.hpp:724-985 (struct DynTraj).

C++ uses ExprTk for analytic expressions; we use SymPy's sympify + lambdify,
which supports the same math intrinsics (sin, cos, exp, log, ...).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np
import sympy as sp

from .piecewise_poly import PieceWisePol


class TrajMode(Enum):
    """sando_type.hpp:726 — DynTraj::Mode { Piecewise, Analytic }"""
    Piecewise = 0
    Analytic  = 1


def _compile_one(label: str, src: str, required: bool):
    """Return a callable f(t) -> float, or None if src is empty and not required."""
    if not src:
        if required:
            print(f"Missing required analytic expression {label}", file=sys.stderr)
            return None, False
        return None, True
    try:
        t = sp.Symbol("t", real=True)
        expr = sp.sympify(src)
        fn = sp.lambdify(t, expr, modules=["numpy"])
        # validate by calling once
        _ = fn(0.0)
        return fn, True
    except Exception as e:
        print(f"sympify failure ({label}): '{src}' -> {e}", file=sys.stderr)
        return None, False


@dataclass
class DynTraj:
    # Which representation                                  (sando_type.hpp:727)
    mode: TrajMode = TrajMode.Analytic

    # ---- piecewise cubic branch ----                      (sando_type.hpp:729)
    pwp: PieceWisePol = field(default_factory=PieceWisePol)

    # ---- analytic expression branch ----                  (sando_type.hpp:732-738)
    traj_x: str = ""
    traj_y: str = ""
    traj_z: str = ""
    traj_vx: str = ""
    traj_vy: str = ""
    traj_vz: str = ""
    t_var: float = 0.0
    analytic_compiled: bool = False

    # internal compiled callables (not in C++ — replace exprtk::expression)
    _fx: Optional[Callable] = None
    _fy: Optional[Callable] = None
    _fz: Optional[Callable] = None
    _fvx: Optional[Callable] = None
    _fvy: Optional[Callable] = None
    _fvz: Optional[Callable] = None

    # ---- shared metadata ----                             (sando_type.hpp:741-752)
    ekf_cov_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ekf_cov_q: np.ndarray = field(default_factory=lambda: np.zeros(3))
    poly_cov:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    control_points: List[np.ndarray] = field(default_factory=list)   # each: 3x4
    bbox:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    goal:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_agent: bool = False
    id: int = -1
    time_received: float = 0.0
    tracking_utility: float = 0.0
    communication_delay: float = 0.0

    # --- sando_type.hpp:757 ---
    def setPiecewise(self, poly: PieceWisePol) -> None:
        self.mode = TrajMode.Piecewise
        self.pwp = poly

    # --- sando_type.hpp:766 ---
    def compileAnalytic(self) -> bool:
        ok = True
        self._fx, success = _compile_one("traj_x", self.traj_x, required=True);  ok &= success
        self._fy, success = _compile_one("traj_y", self.traj_y, required=True);  ok &= success
        self._fz, success = _compile_one("traj_z", self.traj_z, required=True);  ok &= success
        self._fvx, success = _compile_one("traj_vx", self.traj_vx, required=False); ok &= success
        self._fvy, success = _compile_one("traj_vy", self.traj_vy, required=False); ok &= success
        self._fvz, success = _compile_one("traj_vz", self.traj_vz, required=False); ok &= success
        self.analytic_compiled = ok
        return ok

    # --- sando_type.hpp:822 ---
    def eval(self, t: float) -> np.ndarray:
        if self.mode == TrajMode.Piecewise:
            return self.pwp.eval(t)
        return self.evalAnalyticPos(t)

    # --- sando_type.hpp:837  static poly5_abs ---
    @staticmethod
    def poly5_abs(c: np.ndarray, tau: float) -> float:
        v = c[5]
        v = v * tau + c[4]
        v = v * tau + c[3]
        v = v * tau + c[2]
        v = v * tau + c[1]
        v = v * tau + c[0]
        return float(v)

    @staticmethod
    def dpoly5_abs(c: np.ndarray, tau: float) -> float:
        v = 5 * c[5]
        v = v * tau + 4 * c[4]
        v = v * tau + 3 * c[3]
        v = v * tau + 2 * c[2]
        v = v * tau + c[1]
        return float(v)

    @staticmethod
    def ddpoly5_abs(c: np.ndarray, tau: float) -> float:
        v = 20 * c[5]
        v = v * tau + 12 * c[4]
        v = v * tau + 6 * c[3]
        v = v * tau + 2 * c[2]
        return float(v)

    # --- sando_type.hpp:878 ---
    def evalAnalyticPos(self, t: float) -> np.ndarray:
        if not self.analytic_compiled:
            print("[DynTraj] evalAnalyticPos called but analytic_compiled==False",
                  file=sys.stderr)
            return np.zeros(3)
        return np.array([float(self._fx(t)),
                         float(self._fy(t)),
                         float(self._fz(t))], dtype=float)

    # --- sando_type.hpp:893 ---
    def velocity(self, t: float) -> np.ndarray:
        if self.mode == TrajMode.Piecewise:
            return self.pwp.velocity(t)
        return self.velocityAnalytic(t)

    # --- sando_type.hpp:908 ---
    def velocityAnalytic(self, t: float) -> np.ndarray:
        if not self.analytic_compiled:
            return np.zeros(3)
        if self._fvx is not None and self._fvy is not None and self._fvz is not None:
            return np.array([float(self._fvx(t)),
                             float(self._fvy(t)),
                             float(self._fvz(t))], dtype=float)
        dt = 1e-3
        x0 = float(self._fx(t));        y0 = float(self._fy(t));        z0 = float(self._fz(t))
        x1 = float(self._fx(t + dt));   y1 = float(self._fy(t + dt));   z1 = float(self._fz(t + dt))
        return np.array([(x1 - x0) / dt, (y1 - y0) / dt, (z1 - z0) / dt], dtype=float)

    # --- sando_type.hpp:928 ---
    def accel(self, t: float) -> np.ndarray:
        if self.mode == TrajMode.Piecewise:
            return self.pwp.acceleration(t)
        return self.accelAnalytic(t)

    # --- sando_type.hpp:942 ---
    def accelAnalytic(self, t: float) -> np.ndarray:
        if not self.analytic_compiled:
            return np.zeros(3)
        dt = 1e-3
        v1 = self.velocity(t - dt)
        v2 = self.velocity(t + dt)
        return (v2 - v1) / (2 * dt)

    # --- sando_type.hpp:956 ---
    @staticmethod
    def modeName(m: TrajMode) -> str:
        return m.name

    # --- sando_type.hpp:968 ---
    def print(self) -> None:
        print(f"DynTraj id={self.id} mode={self.modeName(self.mode)}")
        if self.mode == TrajMode.Piecewise:
            self.pwp.print()
        else:
            print(f"  traj_x='{self.traj_x}'")
            print(f"  traj_y='{self.traj_y}'")
            print(f"  traj_z='{self.traj_z}'")
            if self.traj_vx or self.traj_vy or self.traj_vz:
                print(f"  traj_vx='{self.traj_vx}'")
                print(f"  traj_vy='{self.traj_vy}'")
                print(f"  traj_vz='{self.traj_vz}'")
            print(f"  analytic_compiled={self.analytic_compiled}")
