---
name: feedback-rootcause-first
description: "调试时先穷举 formulation / 设计层面的具体 bug,别一上来贴\"算法局部最优\"之类的标签就跳 paper 方案。用户 2026-05-28 在 local_opt LBFGS-卡-在-高-cost 问题上 push 过这个。"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 95acb254-cbe2-4e07-9a70-0255d93c13fc
---

**规则:debug / 诊断时,先把 formulation 和设计层面的可疑点全穷举一遍(seed 质量、自由度、约束 vs 软罚分、时间预算、梯度数值稳定性、对称性),再考虑"换算法"。**

**Why:** 2026-05-28 在 sando-rgbd local_opt 第一版 LBFGS-卡-高-cost 问题上,我直接贴了"LBFGS + 非凸 cost = 局部最优"标签,跳去问用户走 multi-start / graduated / topology 哪条 paper 方案。用户 push 回来,从代码里拆出 6 个**具体的 formulation bug**:seed 直穿 obstacle、num_ctrl=15 但锁 12 个只剩 3 自由度、obstacle 是 soft 不是 hard、T_target 不跟绕远走、finite-diff 在 hinge cost 上 gtol 不可靠、symmetric 场景对称扰动方向混乱。这些每一条都是可直接改的代码 bug,不需要换算法。**把症状当根因 = 浪费大量算法精力解一个伪问题。**

**How to apply:**
- 看到"算法卡了 / 收敛慢 / 结果不好",先列**当前代码里**所有可疑点(输入质量、自由度、目标函数形状、初值、约束 vs 罚、数值稳定性、对称性),逐条验证。
- "result.success / converged=True" 跟 "结果可用" 是两回事 —— 数值收敛准则跟业务可行性准则要分开,**输出判据写 feasibility check 而不是只看 optimizer 的 flag**。
- soft penalty(`max(0, d_safe - d)^k`)不等于约束。给 weight 再大,它仍是 trade-off 的一项,优化器可以"违反一点换别的便宜"。要 "永不违反" 必须 hard constraint / barrier / augmented Lagrangian。
- 自由度审查:控制点 / DoF 数减去锁住的,剩多少真正能动?调高 num_ctrl 前先核算这个数。
- 时间预算 / 尺度估计要跟着实际轨迹走,不能凭输入估计就锁死。
- finite-diff 梯度在分段函数(hinge / abs / max)的开关点不可靠;对称问题下 forward diff 可能选了"看起来都下降"的方向,gtol 不能解释成"真的到底了"。

参考案例:[[sando-rgbd-localopt-issues]]。
