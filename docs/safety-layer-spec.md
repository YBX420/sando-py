# 安全层论文 · 修正版 Spec（v1，2026-06-11）

> 来源：2026-06-11 多 agent 评估（11 agent：ε 数学双盲推演+对抗复核 / ~20 篇对手论文联网核查 /
> 代码逐文件盘点 / SDD+Isaac 数据核查 / 三立场评委裁决）。原始档案：`docs/safety-layer-dossier.json`。
> 本文档**取代**之前对话总结作为施工蓝图；与原总结的差异都标了【修正】。
> 裁决：**方案 B** —— 主论文 = 安全层（9/15 投 RA-L），MINCO 论文 = side product，
> **严格排在主论文投稿之后**（10-11 月写，11 月中 arXiv）。

---

## 0. 一句话定位

**Planner 无关的认证语义风险安全层**：无人机在行人附近飞行时，对**每个被检测到的行人轨迹、
每个规划回合**，P(撞该人) ≤ ε_cls + ε_pred（ε=0.1），未检测到的行人由两个**确定性分支**兜底；
保证在冷清标定→拥挤部署（类别×密度 Mondrian 分层内）依然成立。

【修正】不再说「覆盖从看错到预测错到执行的全链条」「每时间步」「任意人群」——这三个表述
都会被一句话毙（见 §2、§7）。

---

## 1. 定理（诚实版）

**三个感知分支**（必须明写进定理和摘要）：

| 分支 | 对象 | 机制 | 预算 |
|---|---|---|---|
| ① 统计 tube | 检测到 + 分类的 agent | conformal 分类集合（含 human → 硬约束）+ per-agent conformal tube | ε_cls + ε_pred |
| ② 遮挡阴影 | 没看见的空间 | 确定性：未观测/遮挡空间视为最坏有人，可达阴影 r_occ + v_max·t（v_max ~1.5-4 m/s 分档） | 0（确定性） |
| ③ body 底线 | 看见但没识别成 agent 的 | 现有 depth→occupancy body-clearance 门 + 重规划延迟膨胀 ~0.4 m（0.1s × 4 m/s） | 0（确定性，但只保证重规划分辨率级不穿透） |

**保证粒度**：per-detected-track、per-planning-episode（一次遭遇 = 整段 2-3s horizon）。
per-mission 不承诺（要不就显式对重规划次数做 union——不做）。

**预算**：ε = 0.1，拆 ε_cls = 0.02 + ε_pred = 0.08。ε=0.05 只有 H≤3 或合并密度档才可行。
执行端（求解失败/数值/跟踪）由事后校验 + fallback 确定性吸收，预算 0；跟踪误差以实测上界
确定性膨胀进 tube，写成命名假设。

**前提条件（命名假设，写进定理）**：分层内交换性（类×密度 Mondrian 格内）、行人速度上限、
非反应行人（performativity 进 limitation，有 Lindemann/Dixit 先例 + 引 arXiv 2511.11567）、
**commit horizon < tau_trust（0.75s）**——信任窗外硬约束是关的（local_opt_hardalm.hpp:230-262），
定理必须 condition 在滚动执行上，否则越权。

---

## 2. ε 数学设计（已验算，唯一活法）

- **计分函数**：每个行人一条整段分数 s_i = max_t ‖e_t‖/ρ(t)，ρ(t) 为时间整形函数（拟合 σ_t），
  tube 半径 r(t) = q·ρ(t) 随 horizon 增长。**不是** Lindemann 的 per-step + δ/T union
  （已从原文核实他用的是 per-step union，且自承保守、列为 future work——这是差异化点，引用时别和
  Cleaveland LCP 续作搞混）。
- **union 结构**：只对人数 union：α = ε_pred/H_bin。H≤10 时 α=0.008 → 每格最少 124 条、
  舒服 1,250 条标定轨迹。
- **死刑确认**：per-(agent,timestep) 设计 α≈1.3e-4 → tube ~4.6 m（4× 于 95% 分位，一人堵死
  4 m 走廊）+ 9-18 格 × 7,499~75,000 条 = 6.7 万~135 万条轨迹，不可能。
- **Mondrian**：3 类 {human, vehicle-like, other} × 3 密度档 = 9 格，总舒适需求 ~5,250 条。
- **分类侧**：marginal 覆盖不够（条件漏检率可达 ε/π_human = 50%）→ 用 **human 类条件**
  （Mondrian-by-class）门限；**latch 规则**：track 一旦进过 human 集合永远当 human——这让
  per-frame 保证合法变 per-track（否则 600 帧 union 直接 vacuous）；**标定单位 = 每 track 的
  首次检测帧**（首检帧是协变量偏移的，不能用随机帧标定）。每密度档需 ~200-475 个标注 human
  首检 ≈ 共 ~1,900 个。
- **标定数据三铁律**：(i) 端到端在 **tracker 输出**上标定（吸收 association 误差，否则交换性破）；
  (ii) **每 track 只取一个评分窗**（重叠滑窗 = 伪复制样本，交换性破）；(iii) **绝不用 SDD 标定证书**
  （鸟瞰 vs 机载 D435i 视角偏移 = 交换性破）——SDD 只配训练预测器，证书标定在 Isaac 机载渲染
  或笼录里做。
- **ACI**：只给长期时间平均覆盖（错可以集中在最危险的密集时段），**不准进 headline**，
  降级为经验自适应层并明标 time-average（Dixit L4DC'23 是诚实先例）。
- **预测器 = 成败手**：匀速预测 q95@3s≈1.2m → tube 2.0 m（封死走廊，证书在宣传场景恰好空洞）；
  学习型预测器 q95≈0.6m → 0.9 m（可过）。整个余量在预测器 q95 里，不在 conformal 机器里。
  （注意：这些半径数字来自 lognormal 拟合模型，**要在真残差上验证**，W4 gate。）
- 实现脚注：分位秩 ceil((n+1)(1-α)) 要用精确算术（浮点边界案例会错）。

---

## 3. 四个贡献（修正版）

1. 【修正·改名】**类条件 × 密度 Mondrian 分层标定**（不叫「coverage 崩溃诊断」——泛密度偏移
   框架已被 Lindemann arXiv 2602.12616 占走，且 scene-level 计分下的「崩溃」半是计分方式自己
   造的伪现象，per-agent 计分 + 显式 H union 下大半消失，审稿人会指循环论证）。崩溃图保留，
   但 framing = 类条件覆盖在密度偏移下的失效与分层修复。
2. **感知端证书**：conformal 分类集合门控约束硬度（含 human → 硬）+ per-agent tube 组合成单一
   P(撞人) 证书——这是真正 OPEN 的核心（没人组合过分类集合与预测 tube 进控制）。
   【修正】必须带 §1 的三分支 FN 处理，否则被 Timans 原文一句话毙（FN explicitly out of scope）。
3. **端到端组合定理**：union bound 任意相关性下成立（对的），但 novelty 在**组合对象**
   （类门控硬度 × tube × RTA 递归可行性），不在不等式本身——对这批审稿人 Boole 不等式是 trivial 的。
4. 【修正·降级】**RTA 三件套 = 单 planner**（监视器 + 最小修正 QP + 垂直爬升 backup）。
   **双 planner 即插即用从贡献列表删除**（数学/文献/代码三条线独立提名它是唯一可砍项；
   0 篇文献真做过但单独立不住「这只是评测」；还被 Gurobi license 拖着）。多 planner 至多
   作为评测段落/future work。垂直爬升注意：现 recovery 故意 z=0（recovery.hpp:28），且爬升
   爬进 D435i 没看过的空间——递归可行性必须继承分支②的未知空间处理 + 显式高度预算。

---

## 4. 流水线（沿用原设计 + 两处修正)

双通道并行保留：监视器独立线程跑自己的感知（YOLO→预测器→CP tube），不站在策略推理路径上；
前馈 guidance 只写地图/约束层；出口 enforcement 自身动力学推演 + backup 可行性校验，
串行延迟 <1-2ms；tube 陈旧性用 margin 付。Liveness 日志现在就埋。

【修正 1】监视器以独立 ROS 节点挂 /NX01/goal + state + 障碍 topic（reach_avoid.hpp 无状态
header-only，零接触策略路径）；capi 缺 committed-trajectory getter（pwp_to_share 未暴露），补 ~0.5 天。
【修正 2】**实机 D435i 环路现在没有动态障碍通道**（bridge 从不调 add_traj；C++ DynTraj 无
class 字段 types.hpp:346；ABI 不传类别 sando_capi.cpp:122）——不补（~8-12 人天）conformal
感知故事在真机闭环不了；不补就明说 Isaac 扛定量、真机只做定性 demo。

---

## 5. 代码映射（盘点结论：planner 基本可冻结）

| 组件 | 状态 | 关键位置 | 工作量 |
|---|---|---|---|
| per-class 硬软分发 + unknown→hard fail-safe | ✅ 已建 | avoid_config.py:72-79 / planner.py:1217-1229 / planner.hpp:923-975 | 0 |
| class 占位源（要换掉的） | id 区间 if | planner.py:737-740；C++ 只有 id 规则 planner.hpp:496-499 | Py 侧 ~0.5 天 |
| 分类集合进 C++（DynTraj label 字段 + ABI） | ❌ 新建 | types.hpp:346 / sando_capi.cpp:122 | 2-3 天 |
| 真分类器（set 输出）+ 实机接线 | ❌ 新建 | — | 5-8 + 3-4 天 |
| per-(障碍,段,控制点) 约束索引 | ✅ 已有 | local_opt_hardalm.hpp:208-216（Nc=M·6·H） | 0 |
| tube 半径接入 | 半径被提成单标量 :237；时间增长 reach 模型已有但压在 τ=tau_trust :72-78 | **零手术方案**：取 [0,τ_trust] 窗内最大 tube 半径标量化（sound、保守）；真 per-时间步 = 4 处手术（hardalm R+dgdt / py _alm_constraints ~1257 / _certificate_margin_spacetime :1601 / 门 plan_minco.hpp:915-919）+ golden 重基线 | 0 或 3-5 天 |
| 事后校验 + fallback | ✅ 已建 | check_feasibility plan_minco.hpp:394-442；信任窗硬门 915-936；retime 复验 planner.hpp:1146-1177；监视器门控 yield 1224-1289 | 定理是纸面活 |
| 最小修正 QP + 垂直爬升 | ❌ 新建（现 yield = 5 候选离散 argmax，z=0） | recovery.hpp:28 | 7-10 天（含监视器节点） |
| conformal 标定工具链 + Mondrian/ACI + 崩溃实验 | ❌ 新建 | 仅 q_conformal 标量钩子已通（OptParams→ALM） | 5-8 + 8-12 天 |
| 实验日志 harness | 部分（PlanInfo ~30 字段、certificate_margin 都在；无持久化 CSV/rosbag） | C++ plan_minco 无分段计时 | 3-4 天 |
| Isaac 人群场景 | ❌ 新建 | — | 4-6 天 |

**总计 ~35-55 人天 vs ~60-65 个工作日（零缓冲，且并行申请）→ 只有砍掉第二 planner 才装得下。**
坑提醒：改 C++ 必重编 `cpp/capi/sando_capi.so`（加载第一优先级；Windows 上是 python/sando_capi.dll
旧库陷阱）；任何 R 公式改动触发 golden 19/19 重基线。

---

## 6. 数据计划

- **SDD**（只训预测器）：6 类确认但严重偏斜（ped 11.2k / biker 6.4k / car 1.3k / skater 0.3k /
  cart 0.2k / bus 0.1k）→ 合并 {human, vehicle-like(car+cart+bus ~1.6k), other}；整体密度
  <0.1 人/m²（没有真高密档）；像素坐标无官方 homography（社区 scale ~10% 级误差）；
  68-91% 轨迹开头有 lost 帧——预处理要清。
- **Isaac Sim**（证书标定 + 密度三档主战场）：用 **IRA**（isaacsim.replicator.agent；
  omni.anim.people 已弃用）——`character.groups[].num` 直接编程控密度（如 5/15/40 三档）、
  32-bit seed 确定性、GT 全输出（sem-seg / 2D/3D box / skeleton / object_detection.json 含
  每帧位置/速度/任务）。几十个 agent 常规，100+ 要 prim-recycling workaround。
  **W1 钉死版本**（API 在换代，不钉脚本烂在半路）。拥挤格 ~1,250 条轨迹的生成吞吐要早验。
- **ATC 商场数据集**（可选）：补真实密度轴（92 天、49 传感器），但无类别标签——双数据集分工要在
  paper 里说清，或全靠 Isaac。
- **笼录**：降为「反应行人下的经验 tube 覆盖检查」（化解 performativity 质疑的便宜招）+
  定性 demo；审批 W1 启动，~W8 没批就 sim-only。

---

## 7. 竞争地图 + 写作防雷

**必引 + 精确定位**（错引会被抓）：
- Sundarsingh et al. 2509.25124（RA-L 2026?）：静态、guarantee 是 joint 非类条件——头号近邻 +
  头号被抢源（动态续作是该组自然下一篇，作者群=审稿人群）。**投稿前确认 venue**（目前从 Xplore 推断）。
- Lindemann 2210.10254（RA-L'23）：per-step + δ/T union（非 max-over-horizon），自承保守——
  我们 trajectory-level 分数的差异化锚点；别与 Cleaveland LCP 续作混引。
- Dixit 2212.00278（L4DC'23）：ACP=time-average 诚实先例；其 future work 点名 backup
  controllers/递归可行性——可引来 motivate RTA。
- Timans 2403.07263（ECCV'24）：FN explicitly out of scope——我们三分支的引证基础。
- Strawn 2306.02551（RA-L'23）：**不是无人机论文**（2D gym RL filter，guarantee 条件于训练目标）。
- Perceive With Confidence 2403.08185：静态 planner-agnostic 端到端——「planner 无关」表述要
  与它划界。
- 2602.12616（Lindemann 2026-02）：泛密度偏移已被占——我们只认领 Mondrian 类×密度切片。
- 还有：Kalluraya 2209.06323（动态语义、非 conformal）、COPPOL 2510.18485（FN bound 工具源）、
  2603.08958（Mondrian 进机器人但按状态区域分层）。
**窗口**：名义 6-12 月，按 6-9 算。9/15 投稿 + 当天挂 arXiv；**投稿前重扫 2026-04~06 arXiv**
（本次检索自报有索引盲区）。

**摘要里绝不出现**：whole chain / 每时间步 / ACI 撑的 headline / 任意人群。
**摘要里必须出现**：per-detected-track + 三分支 scoping + 密度分层有效域（H≤10）。

---

## 8. Open risks（按致命度）

1. 被抢（2509.25124 动态续作）——唯一对策是准时 + arXiv。
2. 预测器 q95 不达标 → tube 空洞（W4 gate，砍 horizon 2s / H≤3 兜底）。
3. 诚实三分支证书在密集场景把 planner 冻住 → 数学活了性能故事死了（W7-9 专门测 freeze/yield 率；
   不行上 per-时间步 tube 手术换精度）。
4. 重尾/异常值：n 接近下限时分位数=标定集最大值，一条 ID-switch 轨迹膨胀所有 tube——按格报告
   有效 n（审稿人会自己算）。
5. 真机证据缺口：9 月前 conformal 层可能上不了实机闭环——Isaac 扛定量 + 真机定性，提前声明。
