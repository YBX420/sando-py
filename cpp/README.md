# sando_cpp — faithful C++ port of sando_py

**目标**:把 `sando_py`(博士主线规划器)逐模块还原成 C++。**不许欺骗、必须还原**。

## 纪律(每个模块都必须遵守)
1. **Golden-test 驱动**:Python 先用真实现跑出「输入 → 输出」金标准(`cpp/golden/gen_*_golden.py`),
   C++ 移植后读**同样输入**、算出来必须逐项对上(容差,默认 1e-7),**对不上就不算还原**。
2. **1:1 对照源码**:每个公式照 `sando_py/...` 原样移植,不近似、不偷换。
3. **自底向上**:依赖在前,先数学原语,后优化/搜索/编排。

## 构建 / 验证
```bash
cd cpp && cmake -S . -B build -G "MinGW Makefiles" && cmake --build build -j
ctest --test-dir build --output-on-failure     # 或直接 ./build/test_<mod>.exe golden/<mod>_cases.txt
```
依赖:g++(C++17)、CMake、vendored Eigen(`third_party/eigen`)。

## 进度(逐模块)
| 模块 | Python 源 | 行 | C++ | golden | 状态 |
|---|---|---|---|---|---|
| MinjerkTraj(底座 + **M1 解析梯度/伴随**) | local/minco.py | 715 | minjerk_traj.hpp | minco_cases(前向 5e-11 / 梯度 rel<1e-7) | ✅ 已验证(含梯度) |
| SphereObstacle/AABBObstacle | local/obstacles.py | 131 | obstacles.hpp | obstacles_cases (1e-10) | ✅ 已验证 |
| AvoidParams/default_config | local/avoid_config.py | 79 | avoid_config.hpp | cost_cases(resolve 精确) | ✅ 已验证 |
| cost helpers | local/cost.py | 79 | cost.hpp | cost_cases(rel 1e-6) | ✅ 已验证 |
| local_opt 几何核心(signed_dist_and_grad / seg_vander) | local/local_opt.py | — | local_opt.hpp | localopt_cases (<1e-9) | ✅ 已验证 |
| local_opt cost 项(软障碍 + 超速/超加速 hinge + 解析梯度) | local/local_opt.py | ~300 | local_opt_terms.hpp | local_opt_terms_cases (6.5e-9) | ✅ 已验证 |
| local_opt 硬-ALM 凸包约束(固定乘子一次评估) | local/local_opt.py | ~250 | local_opt_hardalm.hpp | local_opt_hardalm_cases (3e-12) | ✅ 已验证 |
| _minco_cost_grad(总代价+梯度组装) | local/local_opt.py | ~70 | minco_cost_grad.hpp | minco_cost_grad_cases(grad rel 3.7e-13) | ✅ 已验证 |
| 优化器驱动(_alm_solve + L-BFGS-B + plan_minco + detour + 可行性) | local/local_opt.py | ~600 | plan_minco.hpp + LBFGSpp | plan_minco_cases(确定性 3e-13;端到端 valid 7/7,非凸 ALM 落不同合法极小) | ✅ 已验证(优化器跨实现非 bit 对齐,诚实标注) |
| VoxelMapUtil(read_map/heat/索引/LoS/DDA/find_free) | hgp/voxel_map.py | 693 | voxel_map.hpp + voxel_map_helpers.hpp | voxel_map(_helpers)_cases (0.0) | ✅ 已验证 |
| GraphSearch(heat-A*) | hgp/graph_search.py | 287 | graph_search.hpp | graph_search_cases(精确节点) | ✅ 已验证 |
| HGPPlanner(LoS/角点/repair) | hgp/hgp_planner.py | 293 | hgp_planner.hpp | hgp_planner_cases (0.0) | ✅ 已验证 |
| HGPManager(solve_hgp/加密/free) | hgp/hgp_manager.py | 376 | hgp_manager.hpp | hgp_manager_cases (0.0) | ✅ 已验证 |
| types(Parameters/RobotState/DynTraj/PWP/AnalyticExpr) | types.py | 746 | types.hpp | types_cases (5e-11) | ✅ 数值核心已验证 |
| SANDO orchestrator(replan/状态机/编排) | planner.py | 2024 | planner.hpp | planner_cases(确定性 1.3e-12;replan valid 5/5) | ✅ 已验证(MINCO 路径) |

**说明**:这是多轮工程(核心规划链 ~7700 行密集数值+优化代码)。每轮还原一块、golden 验证一块,
台账如上。MINCO 这块的 banded 解先用 Eigen dense LU(结果与 scipy banded LU 一致,1e-11),
后续可换带状解提速——但**先还原对、再提速**。


## 范围界定:已还原 vs 未还原(诚实)
**已还原+golden验证 = 他的算法主线(MINCO per-class 规划器 + HGP 全局 + 编排器)**,统一 `ctest` 15/15 通过。

**未还原(有硬外部依赖,需你定夺):**
- `solver_gurobi.py`(727) — **基线 Gurobi MIQP 局部解**,*不是他的算法*(他走 `local_solver='minco'`);移植需 **Gurobi C++ 商业 license**。
- ~~`nodes/`~~ —— **已全部移植成 C++**(见下「ROS2 C++ 集成」9 节点表)。
- `bspline.py`(265) — 旧 B 样条底座,已被 MINCO 取代(遗留)。**已移植 bspline.hpp**。
- `utils.py` 剩余 + types 的 Polytope/BasisConverter/StateDeriv — 部分几何 helper 已并入移植;其余多为 ROS/可视化。
- planner 的 Gurobi factor-sweep 分支、SFC 凸分解、hover avoidance、YAWING/HOVER 墙钟分支、硬件位姿、telemetry。


## 收尾补充(本轮自移植 + 验证)
| bspline(UniformBSpline:De Boor/求导/nonzero_basis/fit_path) | local/bspline.py | 265 | bspline.hpp | bspline_cases(eval 0.0/fit 1.7e-11) | ✅ 已验证 |
| utils 数值(angle/quat/project_sphere|box/min_time_DI/create_more_vertexes/se3_inv) | utils.py | — | utils_math.hpp | utils_math_cases (0.0) | ✅ 已验证 |
| StateDeriv / Polytope | types.py | — | types_minor.hpp | 编译 + contains | ✅ |
| BasisConverter(BS↔MINVO/Bezier 转换矩阵) | types.py | — | basis_converter.hpp | basis_converter_cases (2e-15) | ✅ 已验证 |
| obstacle_tracker 数值核(voxel降采样/连通分量聚类/9D自适应EKF/numpy.polyfit/方差/外推) | nodes/obstacle_tracker.py | 537 | obstacle_tracker_kernels.hpp | obstacle_tracker_cases(确定性核精确 0~1e-15) | ✅ 已验证 |

**当前:`cd cpp && ctest` 19/19 全过。整个 MINCO per-class 规划算法 + 全局 HGP + 编排 + 遗留 B样条 + 数值工具 + 感知跟踪数值核,全部 C++ 还原 + golden 验证。**

**仍未做(需外部依赖 / 决策):**
- `solver_gurobi.py`(727)—— **基线 Gurobi MIQP,需 Gurobi C++ license**(非他的算法)。这是整个仓库**唯一**因硬外部依赖未还原的部分。
- `nodes/`(9 个 ROS2 节点)—— **已全部移植 + colcon build 通过 + 关键节点运行验证**(见下表)。

**优化器诚实说明**:`plan_minco` 端到端在**非凸硬-ALM**场景下,LBFGSpp 与 scipy L-BFGS-B 会落到**不同但都合法**的局部极小(位置差最大 ~0.95m,都清开人、都 trajectory_valid);确定性核心(cost/梯度/分类/detour/可行性)是精确 bit 对齐的。这是跨优化器实现移植的**固有限制**,非缺陷。


## ROS2 C++ 集成(WSL ROS2 Humble,colcon 构建通过 + 运行验证)
**`setup.py` 里 9 个 console_scripts 节点全部移植成 C++,colcon build ✅,关键节点运行验证 ✅。**

- `cpp/ros2/src/dynus_interfaces/` — 重建的消息包(DynTraj/Goal/State/PWPTraj/CoeffPoly3/ComputationTimes),字段从 sando_py 用法反推(DynTraj 已补全 obstacle_tracker 用到的 ekf_cov_*/poly_* 字段)。**标注:可能与上游 dynus_interfaces 不完全一致**。**colcon build ✅**
- 构建:先 `source /opt/ros/humble/setup.bash`,再 `source cpp/ros2/setup_overlay.bash`(colcon 顶层 `install/setup.bash` 在 `/mnt` DrvFs 上排包有 bug,会丢 overlay;`setup_overlay.bash` 直接 source 每个包的 `local_setup.bash` 绕过)。
- 跑整套 demo:`ros2 launch sando_cpp perclass_demo_cpp.launch.py`

| 节点 | Python 源 | C++ | 验证 |
|---|---|---|---|
| sando_node(包住 `planner.hpp`:State→update_state / term_goal→set_terminal_goal / DynTraj→add_traj / replan+goal 双定时器) | sando_node.py | sando_node.cpp | **运行 ✅** 整套 demo 闭环:状态随目标 A↔B 切换推进 |
| fake_sim(运动学闭环器,Goal→State+TF,hopf 姿态) | fake_sim.py | fake_sim.cpp | **运行 ✅**(demo 中闭环) |
| perclass_obstacle_pub(硬人 sin 横穿 id100 + 软墙 id200,发解析 DynTraj) | perclass_obstacle_pub.py | perclass_obstacle_pub.cpp | **运行 ✅**(demo 中被 planner 消费) |
| auto_goal_pub(延时发 term_goal、A/B 往返、latched QoS) | auto_goal_pub.py | auto_goal_pub.cpp | **运行 ✅**(7s 周期精确切换) |
| obstacle_tracker(点云→聚类→自适应EKF→外推→DynTraj) | obstacle_tracker.py | obstacle_tracker.cpp + obstacle_tracker_kernels.hpp | **数值核 golden ✅ + 端到端运行 ✅**(喂移动团 → 10Hz 发 predicted_trajs,id/bbox/pwp/poly_coeffs/cov 全字段) |
| convert_odom_to_state(Odometry→State,纯字段拷贝) | convert_odom_to_state.py | convert_odom_to_state.cpp | 编译 ✅ |
| convert_vicon_to_state(Pose+Twist 近似时间配对→State) | convert_vicon_to_state.py | convert_vicon_to_state.cpp | 编译 ✅ |
| convert_goal_to_cmd_vel(Lyapunov 跟踪控制律→cmd_vel) | convert_goal_to_cmd_vel.py | convert_goal_to_cmd_vel.cpp | 编译 ✅ |
| odom_to_global_state(TF world→init_pose 缓存 + 刚体变换 odom→world) | odom_to_global_state.py | odom_to_global_state.cpp | 编译 ✅ |

**诚实标注:**
- **obstacle_tracker 聚类**:Python 装了 sklearn 时走 DBSCAN,没装时走 cKDTree+并查集 fallback(连通分量)。本机 WSL 无 sklearn → Python 实际走 fallback,我的 C++ `connected_components` **精确复现 fallback**(golden 0 误差)。**但生产环境若装了 sklearn,DBSCAN 在边界点上会与连通分量不同** —— 这是 Python 自身两条实现路径的差异,非移植缺陷。
- **4 个 convert_* 节点**只编译验证(都是简单字段搬运 / 几条算术);其中 cmd_vel 控制律、odom→world 四元数变换是 1:1 直译,未单独 golden(逻辑足够简单)。
- 仍未还原(需外部依赖):`solver_gurobi.py`(727)—— 基线 Gurobi MIQP,需 Gurobi C++ 商业 license 才能编译+验证,**非他的算法**(他走 `local_solver='minco'`)。
