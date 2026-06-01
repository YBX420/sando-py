"""Launch the Python SANDO planner with the same parameter file as the C++ node.

Designed to be a drop-in replacement for `sando_node` in the existing
launch graph — the planner reads sando.yaml exactly like the C++ version, and
publishes the same ROS topics so downstream nodes (RViz, MAVROS, goal_sender)
work unchanged. fake_sim_py, the converters, and obstacle_tracker_py are
exposed as toggles so individual nodes can be swapped one at a time.

Args:
  namespace          : ROS namespace (default NX01)
  x, y, z, yaw       : initial pose for fake_sim
  use_hardware       : load sando_hw_quadrotor.yaml instead of sando.yaml
  use_fake_sim       : start the Python fake_sim node (default true for sim)
  use_obstacle_tracker : start the Python obstacle tracker
  config_pkg         : ROS package owning the YAML config (default `sando`)
  sim_env            : override sim_env value from config
  data_file          : path for benchmark CSV (empty to disable)
"""
import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _bool(s: str) -> bool:
    return s in ("true", "True", "1", 1)


def generate_launch_description():
    args = [
        DeclareLaunchArgument("namespace", default_value="NX01"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="2.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("use_hardware", default_value="false"),
        DeclareLaunchArgument("use_fake_sim", default_value="true"),
        DeclareLaunchArgument("use_obstacle_tracker", default_value="false"),
        DeclareLaunchArgument("config_pkg", default_value="sando"),
        DeclareLaunchArgument("sim_env", default_value=""),
        DeclareLaunchArgument("data_file", default_value=""),
    ]

    def launch_setup(context, *_args, **_kwargs):
        ns = LaunchConfiguration("namespace").perform(context)
        x = float(LaunchConfiguration("x").perform(context))
        y = float(LaunchConfiguration("y").perform(context))
        z = float(LaunchConfiguration("z").perform(context))
        yaw = float(LaunchConfiguration("yaw").perform(context))
        use_hardware = _bool(LaunchConfiguration("use_hardware").perform(context))
        use_fake_sim = _bool(LaunchConfiguration("use_fake_sim").perform(context))
        use_obstacle_tracker = _bool(LaunchConfiguration("use_obstacle_tracker").perform(context))
        config_pkg = LaunchConfiguration("config_pkg").perform(context)
        sim_env_override = LaunchConfiguration("sim_env").perform(context)
        data_file = LaunchConfiguration("data_file").perform(context)

        try:
            share_root = get_package_share_directory(config_pkg)
        except Exception:
            share_root = ""

        yaml_name = "sando_hw_quadrotor.yaml" if use_hardware else "sando.yaml"
        parameters = {}
        if share_root:
            candidate = os.path.join(share_root, "config", yaml_name)
            if os.path.exists(candidate):
                with open(candidate, "r") as f:
                    raw = yaml.safe_load(f) or {}
                parameters = raw.get("sando_node", {}).get("ros__parameters", {})

        if sim_env_override:
            parameters["sim_env"] = sim_env_override
        elif "sim_env" not in parameters:
            parameters["sim_env"] = "fake_sim"
        if data_file:
            parameters["file_path"] = data_file

        sando_node = Node(
            package="sando_py",
            executable="sando_node",
            name="sando_node",
            namespace=ns,
            output="screen",
            emulate_tty=True,
            parameters=[parameters],
        )

        fake_sim_node = Node(
            package="sando_py",
            executable="fake_sim",
            name="fake_sim",
            namespace=ns,
            output="screen",
            parameters=[{
                "start_pos": [x, y, z],
                "start_yaw": yaw,
                "send_state_to_gazebo": False,
                "default_goal_z": parameters.get("default_goal_z", 2.0),
                "visual_level": parameters.get("visual_level", 1),
            }],
        )

        obstacle_tracker_node = Node(
            package="sando_py",
            executable="obstacle_tracker_node",
            name="obstacle_tracker_node",
            namespace=ns,
            output="screen",
            parameters=[parameters],
            remappings=[("point_cloud", "d435/depth/color/points")],
        )

        nodes = [sando_node]
        if use_fake_sim and not use_hardware:
            nodes.append(fake_sim_node)
        if use_obstacle_tracker:
            nodes.append(obstacle_tracker_node)
        return nodes

    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
