"""One-command per-class RViz demo (headless-verifiable, GUI for the visual).

Brings up the Python MINCO planner + fake kinematic sim + a moving per-class
obstacle (HARD human crossing + SOFT static wall) + an auto goal + RViz.

  ros2 launch sando_py perclass_demo.launch.py            # with RViz
  ros2 launch sando_py perclass_demo.launch.py rviz:=false # headless

The drone flies (0,0,2)->(10,0,2); a human crosses at x=5 (kept >= d_safe, HARD);
a thin wall at (5,1.2,2) may be grazed (SOFT). Per-class is visible: the human
pushes the plan away hard, the wall only gently.
"""
import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _bool(s) -> bool:
    return str(s).lower() in ("true", "1", "yes")


def generate_launch_description():
    args = [
        DeclareLaunchArgument("namespace", default_value="NX01"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("goal_x", default_value="10.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("config_pkg", default_value="sando"),
    ]

    def setup(context, *_a, **_k):
        ns = LaunchConfiguration("namespace").perform(context)
        use_rviz = _bool(LaunchConfiguration("rviz").perform(context))
        goal_x = LaunchConfiguration("goal_x").perform(context)
        goal_y = LaunchConfiguration("goal_y").perform(context)
        config_pkg = LaunchConfiguration("config_pkg").perform(context)

        # planner params from the C++ sando.yaml, with demo overrides
        params = {}
        try:
            cand = os.path.join(get_package_share_directory(config_pkg),
                                "config", "sando.yaml")
            if os.path.exists(cand):
                with open(cand) as f:
                    raw = yaml.safe_load(f) or {}
                params = raw.get("sando_node", {}).get("ros__parameters", {})
        except Exception:
            params = {}
        params["sim_env"] = "rviz_only"     # seed empty map -> replan-ready w/o cloud
        params["local_solver"] = "minco"     # the new per-class MINCO solve
        params.setdefault("visual_level", 2)
        # COMMIT HORIZON: the splice point A must be only ~1 replan ahead, else the
        # drone executes a long committed (stale) segment and never follows the new
        # avoidance near the obstacle (SANDO's k_value_factor=5 was tuned for a slow
        # solver; with the fast MINCO replan it puts A ~2 m ahead -> drone flies
        # straight through the human). Commit ~= one computation period.
        params["k_value_factor"] = 3.0
        params["default_k_value"] = 40

        sando = Node(package="sando_py", executable="sando_node", name="sando_node",
                     namespace=ns, output="screen", emulate_tty=True, parameters=[params])
        fake_sim = Node(package="sando_py", executable="fake_sim", name="fake_sim",
                        namespace=ns, output="screen",
                        parameters=[{"start_pos": [0.0, 0.0, 2.0], "start_yaw": 0.0,
                                     "send_state_to_gazebo": False,
                                     "default_goal_z": 2.0, "visual_level": 1}])
        obstacles = Node(package="sando_py", executable="perclass_obstacle_pub",
                         name="perclass_obstacle_pub", namespace=ns, output="screen",
                         parameters=[{"frame_id": "map",
                                      "human_amp": 2.0,        # realistic person speed:
                                      "human_period": 10.0}])  # peak vy ~1.26 m/s
        goal = Node(package="sando_py", executable="auto_goal_pub", name="auto_goal_pub",
                    namespace=ns, output="screen",
                    parameters=[{"frame_id": "map", "goal_x": float(goal_x),
                                 "goal_y": float(goal_y), "goal_z": 2.0, "delay_s": 3.0}])
        nodes = [sando, fake_sim, obstacles, goal]

        if use_rviz:
            rviz_cfg = os.path.join(get_package_share_directory("sando_py"),
                                    "config", "perclass.rviz")
            rviz_args = ["-d", rviz_cfg] if os.path.exists(rviz_cfg) else []
            nodes.append(Node(package="rviz2", executable="rviz2", name="rviz2",
                              arguments=rviz_args, output="screen"))
        return nodes

    return LaunchDescription(args + [OpaqueFunction(function=setup)])
