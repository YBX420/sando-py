---
name: sando-py-conformal-cert
description: "conformal 连续时间安全证书:定理(over-interval coverage,sup-norm score 避 union bound)2026-06-01 推通 + 合成验证(coverage 0.9008、比 union 紧 1.60x);和 relative-traj bug 同一套数学;下一步接代码+加噪实验。"
metadata: 
  node_type: memory
  type: project
  originSessionId: 19193e4c-df5f-40d8-a717-7ec143270399
---

> ⚠️ **2026-06-11 pivot 后:本文的 sup-norm score / over-interval coverage 定理是主论文的数学前身,被安全层继承** —— 但应用场景从「MINCO 优化器内部」变成 planner 无关监视器里的 **per-detected-track / per-episode** tube(spec §2)。新增的是分类侧 conformal 集合 + Mondrian 类×密度分层 + FN 三分支。读本文取数学,定位/预算/标定铁律以 `docs/safety-layer-spec.md` §2 为准。

**conformal 连续时间安全证书**(核心算法 novelty 升级,2026-06-01 定理推通 + 合成验证 = 命门过了)。为什么要它见 [[sando-py-core-idea]](确定性版交集已被占 2505.11376/2404.16826 + 假装预测精确=最致命洞)。

**定理(推通)**:relative 证书(bug 已修,见 [[sando-py-defense-map]])保证 ||P(s)-c_pred(s)||>=R。接预测误差 e(s)=c_true(s)-c_pred(s):三角不等式 ||P(s)-c_true||>=R-||e(s)||。conformal 从校准集 {e_j} 每条算**标量** R_j=sup_s||e_j(s)||(整段最大误差),取 (1-α) 分位 Q_α。
> **定理**:若 relative margin>=Q_α(即取 R=r+d_safe+Q_α),则 Pr(∀s 不撞)>=1-α。
> 证:conformal 给 Pr(sup_s||e(s)||<=Q_α)>=1-α;该事件下 ||P-c_true||>=R-sup||e||>=(r+d_safe+Q_α)-Q_α=r+d_safe。∎

**算法深度(避 naive union bound)**:nonconformity score **直接取整段 sup-norm**(一条轨迹一个标量),conformal 一次给 Q_α 覆盖整段——不对 N 个时刻各分 α/N(那过保守)。"整段"被压进 score 的定义里,不是事后 union。sup_s||e|| 还能用 Bernstein 凸包紧算,和确定性骨架同构。这就是 novelty 要的"把 per-time coverage 抬成 over-interval coverage"的真归约。

**合成验证(test/_conformal_coverage_probe.py,200 次校准平均)**:trajectory-conformal mean coverage **0.9008**(达标>=0.90),naive union bound 0.9979(过覆盖)但需 **1.60x** 大的 Q。→ 紧 1.60x。**正面回应 novelty "必须有 measurable consequence":省 60% 安全裕度 = 实际飞得动,不是换膨胀公式的话术。这是从 engineering 抬成 algorithm 的第一个数值证据。** 单次校准 coverage 会在 1-α 附近波动(0.891),因 conformal 保证是 marginal,要多次平均看(已确认 0.9008)。

**和 relative-traj bug 修复同一套数学**:R 从 r+d_safe(确定性)变 r+d_safe+Q_α(conformal),骨架 = relative-trajectory Bernstein 凸包一模一样。bug 修复 → conformal 升级是一条连续的线。

**下一步(以后做)**:① 接进 _certificate_margin_spacetime(R+=Q_α,确认端到端还可微)② Bernstein 凸包紧算 sup_s||e||(代替合成里的 grid)③ **加噪闭环实验**(急转弯/遮挡的预测噪声场景,证"带不确定度 vs 确定性膨胀"的成功率优势 = 给证书/per-point 找回 consequence)④ adaptive conformal(在线残差时间相关、非可交换,exchangeability 假设 paper 要讲诚实)。

**诚实标注**:2026-06-01 是**框架验证**(合成误差模型 e~t³ + grid sup + 假设可交换),非最终实现。但定理 + 核心机制(sup-norm score 避 union bound)已站住。临时脚本 test/_conformal_coverage_probe.py 未 commit(探查,接代码时整理成正式实验)。
