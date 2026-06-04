"""Launch the Python SANDO planner with the same parameter file as the C++ node.

中文说明(这个文件是干嘛的):
  这是「在真机/标准仿真上跑 Python 版 SANDO 规划器」的 launch 脚本,设计成能直接顶替
  C++ 的 sando_node:读同一个 sando.yaml,发同样的 ROS 话题,所以下游(RViz、MAVROS、
  发目标点的节点)不用改就能接上。假仿真、各种转换节点、障碍跟踪节点都做成了开关,
  方便一个一个换着调试。和上面那个 perclass_demo 的区别:这个更偏「正经部署/对齐 C++」,
  那个是「一键演示 per-class 效果」。

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


# 中文:把 launch 字符串参数转布尔。
def _bool(s: str) -> bool:
    return s in ("true", "True", "1", 1)


# 中文:ROS 2 launch 入口,返回要启动的节点列表。
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

    # 中文:运行时回调,在这里把各 launch 参数解析成具体值并据此组装节点。
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

        # 中文:找到放 YAML 配置的那个包的 share 目录;找不到就置空,后面走默认。
        try:
            share_root = get_package_share_directory(config_pkg)
        except Exception:
            share_root = ""

        # 中文:真机用硬件参数文件,否则用仿真默认 sando.yaml。
        yaml_name = "sando_hw_quadrotor.yaml" if use_hardware else "sando.yaml"
        parameters = {}
        if share_root:
            candidate = os.path.join(share_root, "config", yaml_name)
            if os.path.exists(candidate):
                with open(candidate, "r") as f:
                    raw = yaml.safe_load(f) or {}
                parameters = raw.get("sando_node", {}).get("ros__parameters", {})

        # 中文:sim_env 优先用命令行传的;没传又 YAML 里也没有,就默认 fake_sim。
        if sim_env_override:
            parameters["sim_env"] = sim_env_override
        elif "sim_env" not in parameters:
            parameters["sim_env"] = "fake_sim"
        # 中文:给了 data_file 就把基准测试 CSV 的输出路径塞进去(留空=不记录)。
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

        # 中文:障碍跟踪节点(从点云里聚类/预测动态障碍)。这里把它订阅的话题重映射到
        #       RealSense D435 相机的点云话题。
        obstacle_tracker_node = Node(
            package="sando_py",
            executable="obstacle_tracker_node",
            name="obstacle_tracker_node",
            namespace=ns,
            output="screen",
            parameters=[parameters],
            remappings=[("point_cloud", "d435/depth/color/points")],
        )

        # 中文:规划器一定启;假仿真只在「仿真且非真机」时启;障碍跟踪按开关启。
        nodes = [sando_node]
        if use_fake_sim and not use_hardware:
            nodes.append(fake_sim_node)
        if use_obstacle_tracker:
            nodes.append(obstacle_tracker_node)
        return nodes

    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
