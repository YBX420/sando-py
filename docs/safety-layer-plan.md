# 安全层论文 · 13 周施工计划（v1，2026-06-11 → 投稿 ~2026-09-15）

> 配套 `docs/safety-layer-spec.md`（修正版蓝图，先读它）。
> 方案 B：100% 时间给安全层主论文；MINCO side paper **9/15 之后**才动笔（10-11 月写、11 月中 arXiv，
> 与主论文不重叠分赃：letter 认领确定性 per-class 规划/证书/recovery，主论文引它当载体）。

## 0. 三条铁律

1. **9/15 前为 side paper 花一小时都是双输**（两个评委独立点名这是 B 的唯一失败模式）。
2. **永不砍**：FN 三分支、端到端 tracker 输出标定、学习型预测器。
3. 摘要里防雷句式见 spec §7——赶 deadline 时也不准把「whole chain / 每时间步」写回去。

## 1. 时间线（W1 = 移植笔记本完成后第一周，约 6/15 起）

| 周 | 干什么 | Gate / 产出 |
|---|---|---|
| **W1** | ① Boyle 签字（弱化定理一页纸，见下）② 钉 Isaac 版本 + IRA `groups[].num` 跑通 5/15/40 三档 ③ 飞行笼审批材料提交 ④ 预测器选型 + SDD 下载预处理启动 ⑤ DynTraj label-set ABI 设计稿 ⑥ 4 个组的 arXiv alert | **Gate 0：Boyle 不接受弱化叙事 → 回退 C**（planner 论文为主） |
| W2-3 | conformal 标定工具链（split-CP、Mondrian 分格、精确算术分位秩、覆盖审计）+ ABI 类别通道（C++ DynTraj 字段 + traj_create + bridge，2-3 天）+ 日志 harness 骨架 | 标定工具在合成数据上过单测 |
| W2-4 | 学习型预测器（SDD 训练；轻量级，CA-EKF 残差或小型轨迹网络皆可——目标只是把 q95 打下来） | **Gate 1（W4 末）：tracker 输出上 3s q95 ≤ ~0.6-0.7 m**；不达标 → horizon 砍到 2s / H≤3，叙事照常 |
| W4-5 | tube 标量化接入（零手术方案：[0,τ_trust] 窗内最大半径 → 现有 R 通道）+ Isaac 场景与数据生成（拥挤格 ~1,250 条轨迹 + 每密度档 ~200-475 个 human 首检帧） | 数据生成吞吐验证（这是隐性瓶颈） |
| W5-6 | FN 遮挡阴影分支（复用 depth→occupancy + body-floor 门 + 0.4 m 延迟膨胀）+ 端到端标定跑通（tracker 输出、每 track 单窗、首检帧 latch） | 9 格覆盖审计表（按格报有效 n） |
| W6-7 | 监视器独立节点（capi 补 committed-traj getter ~0.5 天）+ 最小修正 QP + 垂直爬升 backup（显式高度预算；递归可行性继承分支②） | RTA 三件套单元 demo |
| W7-9 | 实验矩阵：按格覆盖 / 5/15/40 密度扫 / **freeze-yield 率**（保守度专项）/ 崩溃-修复对照图 / liveness 指标 | **Gate 2：诚实证书把 planner 冻住？→ 上 per-时间步 tube 手术（4 处、3-5 天）+ golden 重基线** |
| **W7 中** | **总 go/no-go**：分类器+预测器+标定链没有可演示成果 → 预案：planner letter 升级为第一篇投稿，安全层降级 workshop paper + 申请材料素材 | 预先和 Boyle 约好,不临场改 |
| W9-11 | 定理写作（三分支、命名假设、commit<τ_trust 条件）+ 论文主体 + **重扫 2026-04~06 arXiv** + 确认 2509.25124 venue | 初稿给 Boyle |
| W12-13 | 打磨、内审、投 RA-L + 当天挂 arXiv | 9/15 |
| 10-11 月 | MINCO side paper（现成结果写作，零新实验）→ 11 月中 arXiv；若主论文 W7 回退，则此项提前 | 12 月申请：两篇在手 |

## 2. 砍单（滑了按序砍，砍前不犹豫）

1. 笼飞（→ sim-only，提前声明真机定性）
2. QP + 垂直爬升（→ 退回现有 monitor + yield，RTA 段落降格）
3. per-时间步 tube 手术（→ 保持标量化，论文如实写保守化）
4. CopulaCPTS / Cleaveland 对比基线
5. Mondrian 缩到 3 类 × 2 密度档

## 3. W1 清单（移植完成当天起）

- [x] **Boyle 一页纸**：**已签字，Gate 0 通过**
- [x] 笔记本环境照 `docs/UBUNTU22_PORT.md` 走完（ctest **20/20** + 闭环 PNG 为准）
- [x] DynTraj label-set ABI（C++ core + capi + bridge + ROS2 端到端，ctest 20/20，ROS2 9/9）
- [ ] Isaac Sim 安装 + **版本钉死记录在案** + IRA 三档密度 demo（5/15/40）跑通出 GT json
- [ ] SDD 下载 + 预处理脚本启动（lost 帧清洗、3 类合并、密度 bin 统计复核）
- [ ] 飞行笼审批材料提交（长周期项，先点火）
- [ ] arXiv alert：Lindemann / Kantaros / Atanasov / Pappas 四组
- [ ] （可选）统计 co-author：仅当 Boyle 要求——最终数学是 union bound,不前置阻塞

## 4. 风险监控（每周过一遍）

| 信号 | 动作 |
|---|---|
| 四组任何一家挂出「动态 + 语义 + conformal」preprint | 立刻评估重叠度,必要时压缩到核心贡献②提前投 |
| W4 预测器 gate 未过 | horizon 2s / H≤3,不拖延 |
| Isaac 数据吞吐 < 需求 | 减格(3×2)或并行机器 |
| W7 中 go/no-go 任一支柱缺席 | 启动回退预案,当天和 Boyle 定 |
| golden 重基线连锁(改了 R 公式) | 按 UBUNTU22_PORT.md 的 .so 重编纪律,重基线一次性做完 |
