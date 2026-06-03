# setup_overlay.bash — source this to get the C++ ROS2 overlay on top of ROS Humble.
# NOTE: colcon's top-level install/setup.bash mis-orders packages when the workspace
# lives on a Windows DrvFs mount (/mnt/...), leaving AMENT_PREFIX_PATH = just humble.
# Sourcing each package's own local_setup.bash directly is reliable, so we do that.
#   usage:  source /opt/ros/humble/setup.bash && source setup_overlay.bash
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/install/dynus_interfaces/share/dynus_interfaces/local_setup.bash"
source "$_here/install/sando_cpp/share/sando_cpp/local_setup.bash"
unset _here
