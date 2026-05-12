"""Strip ROS-injected Python 3.10 paths before tests collect modules.

Without this, importing pytest plugins picks up /opt/ros/humble/.../python3.10/
which is incompatible with the conda env's Python 3.11.

Tests under ``tests/ros/`` need the opposite — they exercise the rclpy /
dynus_interfaces converters and must keep the ROS site-packages on
``sys.path``. Set ``SANDO_PY_KEEP_ROS_PATHS=1`` in the environment to skip
this stripping (``scripts/with_ros_env.sh`` does that automatically).
"""

import os
import sys

if not os.environ.get("SANDO_PY_KEEP_ROS_PATHS"):
    _ROS_TOKENS = ("/opt/ros/", "sando_ws/install/", "decomp_ws/install/")
    sys.path[:] = [p for p in sys.path if not any(t in p for t in _ROS_TOKENS)]
