#!/usr/bin/env bash
# Run a command in the conda `sando` env with ROS PYTHONPATH stripped.
# Usage:   scripts/with_sando_env.sh python -m pytest tests/
#          scripts/with_sando_env.sh python my_script.py
set -e
SANDO_PY=/home/boxuan/miniconda3/envs/sando/bin
export PATH="$SANDO_PY:$PATH"
unset PYTHONPATH AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX COLCON_PREFIX_PATH \
      CMAKE_PREFIX_PATH ROS_DISTRO ROS_PYTHON_VERSION ROS_VERSION \
      ROS_LOCALHOST_ONLY ROS_DOMAIN_ID
exec "$@"
