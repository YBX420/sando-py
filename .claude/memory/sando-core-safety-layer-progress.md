---
name: sando-core-safety-layer-progress
description: "安全层(safety 包裹)在 sando-core 仓库的施工进度:第一块砖 DynTraj conformal label-set ABI 已端到端做完并验证(含 ROS2);下一波是预测器(命门)与监视器。"
metadata:
  type: project
---

**安全层 = planner 无关的「包裹」**(见 [[sando-py-pivot-2026-06]]):C++ 规划器(planner.hpp)当**夹心载体**不动;安全层包在**进料口**(conformal 分类集合→改写避障软硬/tube)和**出料口**(独立监视器 + 最小修正 QP + 垂直爬升 RTA)。施工在 **sando-core** 仓库(纯 C++ 核心,见 [[sando-core-ros2-run]])。

**进度(2026-06-17):**
- ✅ **A = 进料口第一块砖:DynTraj conformal label-set ABI** —— 端到端做完并验证。
  - C++:`DynTraj.label_set`(Mondrian 码 0=HUMAN/1=VEHICLE_LIKE/2=OTHER)+ `derived_class()`(集合优先,human∈集合→硬;空集合→退回 id 启发式 id≥200→软)。读回:`obst_hard`(真硬度,含快 wall→dynamic 重分类)、`obst_id`、`obst_snapshot_time`(新鲜度)。
  - 三条路径都通:capi(`traj_set_label_set` + `sando_get_obst_*`,bridge 返回 `(id,code)` 对)、ROS2(`DynTraj.msg` 加 `int32[] label_set` → `sando_node` 映射 → `obst_class_codes`/`obst_snapshot_time` 话题)。
  - 测试:**ctest 20/20**(新 `dyntraj_labelset` 单测含 id 边界 199/200/300)+ **ROS2 端到端 9/9**(`cpp/ros2/test_labelset_ros2.py`,跑法 `run_labelset_test.sh`)。
  - 过了 **5-lens 对抗 review 工作流**,确认的 9 条全修(按 id 对齐读回、真硬度、staleness 标记)。权威细节 = `docs/dyntraj-labelset-abi.md`(§6 = review 结果)。提交 `sando-core@7cca63b`。
  - **Phase 2 待办**(已记 doc §6):真分类器接线;空集合二义(无分类器 vs 分类器吐空集→应硬);id 段重载长期弃用。

**下一波(按致命度,见 [[sando-py-pivot-2026-06]] / safety-layer-plan):**
1. **B 存亡 = 学习型预测器**(W4 gate:tracker 输出 3s q95≤0.6-0.7m,tube 空不空全看它)——需先下 SDD 预处理。
2. **C 出料口 = 独立监视器节点 + 最小修正 QP + 垂直爬升 backup**(capi 已有 committed-traj 读回的地基:`obst_class_codes`/snapshot_time)。
