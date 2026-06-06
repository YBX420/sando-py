"""One-command per-class RViz demo (headless-verifiable, GUI for the visual).

中文说明(这个文件是干嘛的):
  这是「一条命令把整个 per-class 演示拉起来」的 ROS 2 launch 脚本。它同时启动:
  Python 版 MINCO 规划器节点(sando_node)、一个轻量假仿真(fake_sim,只算运动学不跑 Gazebo)、
  一个会动的 per-class 障碍发布器(横穿的「人」=HARD 硬约束 + 一面静止的「墙」=SOFT 软场)、
  一个自动发目标点的节点,以及可选的 RViz 可视化。
  这就是你那个「跑得不太好的 demo」的入口——要调参数(比如下面那个 k_value_factor 提交时域)
  通常先看这里给规划器灌了什么。

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


# 中文:把 launch 传进来的字符串参数转成布尔(launch 参数都是字符串)。
def _bool(s) -> bool:
    return str(s).lower() in ("true", "1", "yes")


# 中文:ROS 2 launch 的入口函数,框架会自动调用它来拿到「要启动哪些节点」。
def generate_launch_description():
    args = [
        DeclareLaunchArgument("namespace", default_value="NX01"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("goal_x", default_value="10.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("config_pkg", default_value="sando"),
    ]

    # 中文:OpaqueFunction 的回调——launch 参数要到运行时才有具体值,所以建节点的逻辑
    #       必须放在这里,用 .perform(context) 才能把参数解析成真正的字符串。
    def setup(context, *_a, **_k):
        ns = LaunchConfiguration("namespace").perform(context)
        use_rviz = _bool(LaunchConfiguration("rviz").perform(context))
        goal_x = LaunchConfiguration("goal_x").perform(context)
        goal_y = LaunchConfiguration("goal_y").perform(context)
        config_pkg = LaunchConfiguration("config_pkg").perform(context)

        # planner params from the C++ sando.yaml, with demo overrides
        # 中文:先读 C++ 基线的 sando.yaml 拿一套默认规划参数,读不到就空着(不让 demo 挂掉),
        #       下面再按本 demo 的需要覆盖几个关键项。
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
        # 中文:rviz_only=不接点云/不跑感知,直接给空地图,让规划器能立刻进入重规划循环。
        params["local_solver"] = "minco"     # the new per-class MINCO solve
        # 中文:关键!把局部求解器切到新的 per-class MINCO(而不是上面那个旧 Gurobi MIQP)。
        params.setdefault("visual_level", 2)  # 中文:可视化详细程度,2=多画点东西方便看。
        # COMMIT HORIZON: the splice point A must be only ~1 replan ahead, else the
        # drone executes a long committed (stale) segment and never follows the new
        # avoidance near the obstacle (SANDO's k_value_factor=5 was tuned for a slow
        # solver; with the fast MINCO replan it puts A ~2 m ahead -> drone flies
        # straight through the human). Commit ~= one computation period.
        # 中文(这段是 demo 跑不好最常见的坑,重点看):k_value_factor 控制「提交时域」——
        #   每次重规划时,无人机会先「锁定执行」当前轨迹靠前的一段(splice 拼接点 A),这段
        #   是不会被新规划改的。这个因子越大,锁定得越远(A 越靠前)。
        #   SANDO 原值 5 是给慢求解器调的;现在 MINCO 重规划很快,如果还用 5,A 会落在
        #   障碍物前约 2 米,等于「锁死了一段直冲人的老轨迹」,新算出来的避让根本来不及生效,
        #   无人机就直直穿过人。所以这里压到 3,让提交时域约等于「一个计算周期」那么短,
        #   尽快用上最新的避让。default_k_value 是提交点的基准步数。
        params["k_value_factor"] = 3.0
        params["default_k_value"] = 40

        # 中文:规划器主节点,把上面那套 params 灌进去。
        sando = Node(package="sando_py", executable="sando_node", name="sando_node",
                     namespace=ns, output="screen", emulate_tty=True, parameters=[params])
        # 中文:假仿真,从 (0,0,2) 起飞,只算运动学(不发给 Gazebo)。
        fake_sim = Node(package="sando_py", executable="fake_sim", name="fake_sim",
                        namespace=ns, output="screen",
                        parameters=[{"start_pos": [0.0, 0.0, 2.0], "start_yaw": 0.0,
                                     "send_state_to_gazebo": False,
                                     "default_goal_z": 2.0, "visual_level": 1}])
        # 中文:per-class 障碍发布器。human_amp/period 控制那个横穿的人来回摆动的幅度和周期,
        #       这里调成接近真人走路的速度(峰值横向速度约 1.26 m/s)。
        obstacles = Node(package="sando_py", executable="perclass_obstacle_pub",
                         name="perclass_obstacle_pub", namespace=ns, output="screen",
                         parameters=[{"frame_id": "map",
                                      "human_amp": 2.0,        # realistic person speed:
                                      "human_period": 10.0}])  # peak vy ~1.26 m/s
        # 中文:自动发目标点,延迟 3 秒再发(等其他节点起来),目标在 (goal_x, goal_y, 2)。
        goal = Node(package="sando_py", executable="auto_goal_pub", name="auto_goal_pub",
                    namespace=ns, output="screen",
                    parameters=[{"frame_id": "map", "goal_x": float(goal_x),
                                 "goal_y": float(goal_y), "goal_z": 2.0, "delay_s": 3.0}])
        nodes = [sando, fake_sim, obstacles, goal]

        # 中文:rviz:=true 时再加一个 RViz 节点;有现成的 perclass.rviz 配置就加载它。
        if use_rviz:
            rviz_cfg = os.path.join(get_package_share_directory("sando_py"),
                                    "config", "perclass.rviz")
            rviz_args = ["-d", rviz_cfg] if os.path.exists(rviz_cfg) else []
            nodes.append(Node(package="rviz2", executable="rviz2", name="rviz2",
                              arguments=rviz_args, output="screen"))
        return nodes

    return LaunchDescription(args + [OpaqueFunction(function=setup)])
