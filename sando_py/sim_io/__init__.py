"""ROS adapter layer — wraps the pure-Python algorithm code (core, hgp,
decomp, solver) in a single ``rclpy.Node`` so it can plug into the same
ROS 2 simulation world the upstream C++ project uses (RViz +
``dynamic_forest_node`` + ``fake_sim`` + ``goal_sender``).

This module imports ``rclpy`` and ``dynus_interfaces.msg`` at module
load — those are only available when ROS 2 is sourced. Pure-Python users
(running unit tests in a conda env without ROS) should NOT import this
subpackage; the algorithm code is in the other subpackages and has no
ROS dependency. The repo ``conftest.py`` actively strips ROS paths from
``sys.path`` when ``SANDO_PY_KEEP_ROS_PATHS`` is unset so this stays
enforced automatically.

See ``sim_io/README.md``.

Entry point
-----------
    python -m sando_py.sim_io                  # launches SandoNode

Public API
----------
SandoNode        - rclpy.Node mirroring sando_node.cpp           (sando_node.py)
DroneStatus      - YAWING/TRAVELING/GOAL_SEEN/GOAL_REACHED enum  (sando_node.py)
main             - rclpy.init/spin/shutdown entry                (sando_node.py)
msg_convert      - ROS msg <-> core type converters submodule    (msg_convert.py)
"""

from . import msg_convert  # noqa: F401  re-export submodule
from .sando_node import DroneStatus, SandoNode, main

__all__ = ["SandoNode", "DroneStatus", "main", "msg_convert"]
