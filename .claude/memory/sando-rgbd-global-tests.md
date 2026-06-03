---
name: sando-rgbd-global-tests
description: sando-rgbd 全局段(地图/A*/后处理)测试清单:每个测试文件覆盖哪段代码、验了什么、找到并修了哪些 bug(截至 2026-05-25)。代码在本机 /home/boxuan/code/sando_ws/src/sando_py/。
metadata: 
  node_type: memory
  type: reference
  originSessionId: 606a95c7-0a2c-44df-8c70-61a6b1f56252
---

# 全局段测试清单(2026-05-25)

代码:本机 `/home/boxuan/code/sando_ws/src/sando_py/`。测试脚本都在 `test/`,独立可跑(不依赖 ROS/gurobi),改了地图/A* 随时重跑。

## 测试文件 → 覆盖代码 → 验了什么
| 测试 | 覆盖代码 | 验证内容 | 结果 |
|---|---|---|---|
| `stage1_map_heat.py` | `voxel_map.py`(占据+热度) | 坐标来回转;空地图=内部 UNKNOWN + y 墙;点云障碍占据+膨胀;动态障碍 AABB + dyn_mask;热度近高/远低/归 0/受 Hmax 限;静态障碍周边热度 | 16/16 |
| `stage2_astar.py` | `graph_search.py`(A* 核心) | 全 3D:空地→直线;墙留缝→穿缝;堵死→报无路;热度→绕开;**最优性 vs Dijkstra(6 随机图)** | 11/11 |
| `stage2b_hgp_postprocess.py` | `hgp_planner.py`(视线捷径+简化) | 开阔→捋直;3D 障碍块→简化路不穿障碍;端点不变;窄缝墙安全 | 12/12 |
| `stage_stress.py` | 三者合一(随机/真实) | A:200 随机 3D 图后处理 0 穿障碍;B:带 UNKNOWN+w_unknown 的 A* vs Dijkstra(30 次);C:真 read_map 地图热度真推开路(on<off);D:动态预测热度跟未来走 | 8/8 |
| `dynamic_astar.py` | graph_search + 动态热度 | 有预测→绕未来扫过的走廊;时机→快穿过躲得狠、慢穿过几乎不躲。注:A* 无时间轴,真空间+时间避让归 local_opt | 4/4 |

## 测试中找到并修复的 bug
1. **坐标写/读口径不一致**(stage1):占据写用 floor、查询 float_to_int 用 -0.5,差 ~1 格,三套口径并存。修:float_to_int 统一改 floor。→ `voxel_map.py`
2. **简化删点不查障碍**(stage2b):angle_spacing_filter / collapse_short_edges 只看几何 → 简化路穿障碍。修:删点前加碰撞检查 + hgp_planner 末尾 `_repair_blocked_segments` 安全网。→ `utils.py` `hgp_planner.py`
3. **A* 斜穿墙角**(stress):26 连通对角步擦过被占角格 → 引导线擦障碍。修:A* 加 no-corner-cutting(`_corner_cut`)。→ `graph_search.py`
4. **碰撞检查点采样漏格**(stress):is_blocked / line_of_sight_capsule 漏判薄格/掠过。修:换精确体素遍历 DDA(Amanatides-Woo)。→ `voxel_map.py`
5. **静态热度在真实(全 UNKNOWN)地图死掉**(stress):默认不写 UNKNOWN。修:`static_heat_apply_on_unknown` 默认 True。→ `voxel_map.py`

## 改过的源文件
`graph_search.py`、`voxel_map.py`、`utils.py`、`hgp_planner.py`(都没碰 solver/gurobi,那块阶段3 要删)。

## 状态
全局段(地图 / A* / 后处理)= 已扎实验证(非抽查:含随机批量、跟 Dijkstra 对最优、真 read_map 地图、动态预测)。下一步阶段3 写 local_opt。详见 [[sando-rgbd-plan]] / [[sando-rgbd-plan-v2]]。

## 补充测试 + 修复(2026-05-25 下午,按用户优先级补"该测没测的")
新增测试:
- `regression.py` — 5 个修过的 bug 各钉一个回归用例(防复发)。5/5
- `invariants.py` — 路径安全不变量(**固定 seed 可复现**):每段不穿障碍内部、顶点不越界、能到终点;覆盖随机图 / 真 read_map(膨胀) / 动态障碍占据(后处理动态下安全)。3/3

invariants 又揪出**第 6 个 code bug 并修复**:
6. **DDA 完美对角顶点漏判**:is_blocked 的 Amanatides-Woo 在两轴同时跨界的角点只走一边,漏掉 OCC 角格。修:平局时把擦到的角格也查(tie-aware supercover)。→ `voxel_map.py`

**碰撞语义澄清**:碰撞 = 穿过 OCC 格**内部(有长度)**;擦角点/贴面(零长度、在膨胀边界上)**不算**。测试校验器改成"只算穿内部"(frac 距各面 > margin)。这把几个"失败"识别为校验器假阳性,非代码 bug。

**全套 7 个测试文件全绿**:stage1 16 / stage2 11 / stage2b 12 / regression 5 / invariants 3 / dynamic 4 / stress 8(共 59 项)。

动态时间维度(用户要的"同一空间不同时间安全"):全局 A* **没有时间轴**,只能验到"绕时间加权的预测管"(dynamic_astar 已验)。真正"在正确时刻避开"是 **local_opt(阶段3)** 的事,届时补测。

## 中优先级测试补完(2026-05-25)
- `coords.py` — 坐标系一致性(world↔voxel↔path,3 种 origin/res 配置,floor 约定,负坐标,lin_index,路径连续性)。14/14
- `boundary.py` — 极端边界(贴墙/角落/负坐标/3³微地图/无解/start==goal/起点在障碍内/voxel 边界坐标/目标越界):不崩、安全、该报无路报无路。9/9
- `cost_sanity.py` — 代价函数单调性:heat_weight↑→热度曝光↓;w_unknown↑→少走未知格;inflation↑→离障碍更远。6/6
- `perf.py` — 性能(只为抓挂死/爆炸,阈值 30s):大图 10万格 A* 4.66s;read_map+20 动态障碍 3.07s;45% 密障碍 0.72s。3/3
  - **基线提示**:纯 Python A* 在 10万格要 ~4.7s,慢 → 适合研究/离线,高频上机需 C++/numba(符合计划定位)。

**全套 11 个测试文件全绿,共 91 项**:stage1 16 / stage2 11 / stage2b 12 / regression 5 / invariants 3 / dynamic 4 / stress 8 / coords 14 / boundary 9 / cost_sanity 6 / perf 3。
用户优先级清单:**高**(回归/安全不变量/后处理动态安全/可复现 seed)✅;**中**(边界/代价 sanity/性能/坐标一致)✅。剩**动态时间维度深测** → 等 local_opt(阶段3)一起做。
