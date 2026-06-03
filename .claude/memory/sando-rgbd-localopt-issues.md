---
name: sando-rgbd-localopt-issues
description: "local_opt 第一版 6 个 formulation bug 的具体清单 + 5 步修复优先级。用户 2026-05-28 拆出来,比 \"LBFGS 局部最优\" 标签深刻得多。"
metadata: 
  node_type: memory
  type: project
  originSessionId: 95acb254-cbe2-4e07-9a70-0255d93c13fc
---

# local_opt 第一版的 6 个 formulation 问题(2026-05-28 拆解)

测试场景:直线 path (0,0,0)→(10,0,0),球 obstacle center=(5,0,0) r=0.6,LBFGS 16 iter 后 `converged=True`,但 obs_cost 6642 / max|v|=6.94/vmax=3。我之前把这归因到 "LBFGS 非凸局部最优",**用户 push:这不是根因**。真根因是 6 件 formulation / 设计层面的事:

1. **seed 不携带绕障拓扑**。`UniformBSpline.fit_path(astar_path)` 只做 LSQ,不管 clearance。astar_path 本身直穿 obstacle 时,LSQ spline 也直穿。LBFGS 拿到一个撞障碍的初值,后面只能做"形状附近微调",做不到"重新选绕法"。
2. **自由度太少**。num_ctrl=15 + 首末各锁 p+1=6 → 中间只 3 个 ctrl 可动。3 个 ctrl 要负责绕障 + 平滑 + 限速 + 限加速。一拉就曲率/snap/vel/accel 全爆,smooth/vel/accel cost 把它拉回去 → 半绕开。
3. **obstacle 是 soft penalty,不是 hard constraint**。`max(0, d_safe-d)^3` 给 weight 再大仍是 cost 一项。优化器允许撞一点换 smooth/vel 便宜。"obstacle cost 还高" ≠ "optimizer 没找到全局",也可能 = "soft cost 平衡下撞这一点是当前权重的最优解"。
4. **dt0 / T_target 基于原始 path length,绕远后不跟着放大**。`dt0 = path_len/vmax/(num_ctrl-p)`。直线 path_len=10m,绕障后实际 ≥10m。给的初始时间预算偏紧 + `c_time = w_time·((T-T_target)/T_target)²` 不鼓励 T 涨 → 同样 T 路径变长 = 速度涨 = vel cost 涨。max|v|=6.94 不是单纯 optimizer 问题,是 **空间变形 + 时间不跟着放大** 的连锁。
5. **finite-diff 梯度在非光滑 cost 上不可靠**。cost 含 `max(0,…)^k`、`|v|`、AABB 距离 — 这些在 hinge / 边界点 / 中心点附近梯度不连续或数值不稳。对称场景(球 obstacle 中心轴上)forward diff 可能两个方向都"看起来下降",central diff 可能对称项消掉接近 0。`gtol=1e-6` 触发不能解释为"真没下降方向"。
6. **判据用 optimizer flag 而不是 trajectory feasibility**。`result.success / info.converged` 只说数值收敛,跟"轨迹能不能用"完全是两件事。要分开 — 输出 valid 应该看 `min_clearance ≥ d_safe and max_v ≤ vmax and max_a ≤ amax`。

# 5 步修复优先级(用户 push 的顺序)

**不要先换 optimizer**,先按这个顺序修 formulation:

1. **detour multi-start seed**。每个直线穿障碍的情况,生成 ±u / ±v detour seeds(在 obstacle 垂直方向人为推开),各自 LSQ fit + LBFGS,选 feasible 的最小 cost。**这一步直接给 seed 注入拓扑信息**,比通用 multi-start 对症得多。
2. **加 num_ctrl / 不锁那么死**。num_ctrl=15 锁 12 个 = 自由度灾难。要么提 num_ctrl 到 20+,要么用 partial clamp(只锁位置,不锁速度/加速度=0)。让 obstacle 附近至少 5-8 个可动 ctrl。
3. **feasibility 和 optimizer convergence 分开**。`info` 加 `min_clearance / max_v / max_a / is_feasible`,**`is_feasible` 跟 `converged` 是两个独立字段**,上层看 `is_feasible` 决定 hover / replan,看 `converged` 看数值健康度。
4. **time scaling**。空间优化后,如果 max|v| > vmax,就放大 T(等比例放 dt),再评估 vel/accel。可以在 plan() 末尾做 post-process 一次性 fix,也可以做迭代。
5. 1-4 都做了还有"合理 seed 仍卡 obstacle",**才**考虑 graduated optimization / SCP / topology 这些算法层升级。

**关键认知**:LBFGS 是最后一个表现出来的环节,不是根因。表现 = symptom,根因在前面 5 件 formulation 设计。

参考:[[sando-rgbd-plan]] 当前进度,[[feedback-rootcause-first]] 这次的方法论 lesson。
