---
name: sando-py-pivot-2026-06
description: "2026-06-11 方案 B pivot:博士主论文 = planner 无关的认证语义风险安全层(投 RA-L ~2026-09-15);原 per-class MINCO 规划器降为 side paper(10-11 月写)。权威 = docs/safety-layer-spec.md + plan.md。"
metadata:
  type: project
---

**2026-06-11 pivot(方案 B)——博士主线换了。** 旧主线(per-class 差异化 MINCO 规划器,冲顶会)降级为 **side paper**;新主线 = **planner 无关的认证语义风险安全层**,投 **RA-L,deadline ~2026-09-15**。

**安全层是什么**:无人机在行人附近飞行,对**每个被检测到的行人轨迹、每个规划回合**,保证 `P(撞该人) ≤ ε_cls + ε_pred`(ε=0.1 = 0.02 分类 + 0.08 预测)。三个感知分支:① 检测+分类 agent → conformal 分类集合(human→硬约束)+ per-agent conformal tube;② 未观测/遮挡空间 → 确定性遮挡阴影 `r_occ + v_max·t`;③ 看见但没识别 → depth→occupancy body-clearance 门 + ~0.4m 延迟膨胀。保证在 **类别×密度 Mondrian 分层**(3×3=9 格)内成立;时间整形分数用整段 sup/max-over-horizon(避 per-step union)。证书在 **tracker 输出**上、用 **Isaac 机载渲染**标定(**绝不用 SDD 标定证书**)。

**三条铁律**(plan §0,赶 deadline 也不准破):① 9/15 前不碰 side paper(两评委独立点名这是 B 唯一失败模式);② 永不砍 FN 三分支 + 端到端 tracker 输出标定 + 学习型预测器;③ 摘要防雷句式见 spec §7,不准把「whole chain / 每时间步」写回去。

**关键 gate**:W1 = **Boyle 签字**(弱化定理一页纸,不同意 → 回退 C = planner 论文为主);W4 末 = 预测器 tracker 输出 3s q95 ≤ 0.6-0.7m(成败手,不达标砍 horizon);W7 中 = 总 go/no-go。

**Why**:确定性版交集已被占(2505.11376/2404.16826),且「假装预测精确」是最致命洞 → conformal 不确定度证书才够分量;但塞进 MINCO 优化器内部 = planner-绑定、卖点窄,外置成 planner 无关安全层后竞争面更宽、可独立投。

**How to apply**:在本仓库做任何研究判断前,以 `docs/safety-layer-spec.md`(§1 三分支 / §2 ε 数学+标定铁律 / §3 四贡献 / §5 planner 冻结 / §7 防雷)+ `docs/safety-layer-plan.md`(13 周时间线 + 砍单顺序)为准。规划器侧(side paper 载体)地基已成、基本冻结(ctest 19/19),只在安全层需要时动(如 DynTraj label 字段、committed-traj getter)。旧 memory [[sando-py-core-idea]] / [[sando-py-conformal-cert]] / [[sando-py-defense-map]] / [[sando-py-sim2real-fakes]] / [[sando-rgbd-plan]] 的 framing 已过时(conformal 数学 / C++ 结论仍可查),取数学不取定位。
