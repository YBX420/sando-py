"""Shared dataclasses + pure-math utilities. No I/O, no algorithms above the
level of single-function transforms. See ``core/README.md``.

Public API
----------
StateDeriv, Polytope, RobotState, Parameters    (types.py)
PieceWisePol                                    (piecewise_poly.py)
DynTraj, TrajMode                               (dyn_traj.py)
BasisConverter                                  (basis_converter.py)
load_parameters_from_yaml, warn_if_extras       (config_loader.py)

The ``utils`` module is re-exported as a submodule (``from sando_py.core
import utils``) rather than flattened into the namespace — its surface is
large and mostly used by name-spaced callers (``utils.angle_wrap`` etc).
"""

from . import utils  # noqa: F401  re-export submodule
from .basis_converter import BasisConverter
from .config_loader import load_parameters_from_yaml, warn_if_extras
from .dyn_traj import DynTraj, TrajMode
from .hardware import InitPoseTransform, init_pose_from_transform_stamped
from .piecewise_poly import PieceWisePol
from .types import Parameters, Polytope, RobotState, StateDeriv

__all__ = [
    "StateDeriv",
    "Polytope",
    "RobotState",
    "Parameters",
    "PieceWisePol",
    "DynTraj",
    "TrajMode",
    "BasisConverter",
    "InitPoseTransform",
    "init_pose_from_transform_stamped",
    "load_parameters_from_yaml",
    "warn_if_extras",
    "utils",
]
