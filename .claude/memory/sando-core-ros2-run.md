---
name: sando-core-ros2-run
description: "sando-core 的 ROS2 端到端怎么在这台 Ubuntu 笔记本跑(Humble 已装);colcon 自带 setup.bash 在 /media 挂载盘上不生效,必须手动 export 环境。"
metadata:
  type: project
---

**sando-core 的 C++ 核心在 Ubuntu 上走 ROS2 闭环(2026-06-17 首次跑通)。** 工作区 = `sando-core/cpp/ros2`(colcon),包 `dynus_interfaces`(消息)+ `sando_cpp`(节点)。这台移植笔记本**装了 ROS2 Humble + colcon**(`/opt/ros/humble`,系统 python3.10)。

**坑(必记):colcon 生成的 `install/setup.bash` 在 `/media/boxuan/Data2`(root:root 777、像 NTFS)挂载盘上不生效**——source 后 `AMENT_PREFIX_PATH` 不更新、`ros2 pkg list` 找不到包。**绕法 = 手动 export**(且 ROS2 用系统 python,别让 conda 掺和):
```bash
cd sando-core/cpp/ros2 && WS=$(pwd)
conda deactivate; unset PYTHONPATH
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=$WS/install/sando_cpp:$WS/install/dynus_interfaces:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=$WS/install/sando_cpp/lib:$WS/install/dynus_interfaces/lib:$LD_LIBRARY_PATH
# python 消息模块在 local/lib/.../dist-packages(不是 site-packages!),ros2 topic echo 才认:
export PYTHONPATH=$WS/install/dynus_interfaces/local/lib/python3.10/dist-packages:$PYTHONPATH
colcon build --packages-up-to sando_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release   # 首次/改了 C++ 后
ros2 launch sando_cpp perclass_demo_cpp.launch.py    # 无头闭环(无 RViz 节点)
```
**注意**:① C++ 测试用 conda `sando`,**ROS2 用系统 python**,两套别混(`docs/UBUNTU22_PORT.md` 是 conda 那套)。② 改了 `cpp/include` 的 C++ 头后,colcon 要重 build(节点把头编进去),capi.so 也要单独重编(那是 python bridge 路径,见 [[sando-core-ros2-run]] 外的 capi 纪律)。

**走通判据(2026-06-17 实测)**:`perclass_demo_cpp` = sando_node(planner.hpp)+ fake_sim(闭环)+ perclass_obstacle_pub + auto_goal_pub,全无头。目标在 [10,0,2]↔[0,0,2] 每 7s 切换;`/state` 的 pos.x 实测从 0 飞到 9.999(≈目标 10)再回 0,~190Hz、3038 样本/16s、零 error。证明 C++ 核心经 ROS2 topics 闭环健康。

**话题闭环**:auto_goal_pub→`term_goal`→sando_node→`goal`(setpoint)→fake_sim→`state`→sando_node;障碍 perclass_obstacle_pub→`predicted_trajs`(launch remap 成 `trajs`)→sando_node。

**和 label-set ABI 的关系**:`DynTraj.msg` 目前**无类别字段**(只有 id/is_agent/bbox/function/velocity/pwp/ekf)。conformal label-set 已在 C++ 核心 + capi + bridge 落地(见 `docs/dyntraj-labelset-abi.md`,ctest 20/20),但要让它走 ROS2 真路径,需给 `DynTraj.msg` 加 `int32[] label_set` + perclass_obstacle_pub 填 + sando_node 映射进 C++ `DynTraj.label_set`(下一步)。
