#!/usr/bin/env bash
# Run a command inside a sourced ROS 2 Humble environment, with the sando-py
# repo added to PYTHONPATH and the conftest opt-out flag set so /opt/ros
# Python paths are kept on sys.path.
#
# Usage:
#   scripts/with_ros_env.sh python -m pytest tests/ros/
#   scripts/with_ros_env.sh python -m sando_py.sim_io
set -e

source /opt/ros/humble/setup.bash
if [ -f "$HOME/code/sando_ws/install/setup.bash" ]; then
    source "$HOME/code/sando_ws/install/setup.bash"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export SANDO_PY_KEEP_ROS_PATHS=1

exec "$@"
