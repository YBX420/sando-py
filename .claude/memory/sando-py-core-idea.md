---
name: sando-py-core-idea
description: sando-py 核心创新定位:按障碍语义类别选避障『机制』(非权重)+ 对物体类连续时间证书 + 可解释。主轴=机制选择。名字 TBD。2026-06-01 塔菲大人拍板 idea 坚持。
metadata: 
  node_type: memory
  type: project
  originSessionId: 19193e4c-df5f-40d8-a717-7ec143270399
---

> ⚠️ **2026-06-11 已 pivot —— 本文是 pivot 前的核心创新定位,framing 已过时。** conformal 连续时间证书的数学被 pivot 继承,但它从「在 MINCO 优化器**内部**、per-class 为主轴」**外置**成了 planner 无关的独立安全层(现用于 per-detected-track / per-episode 的监视器);per-class 机制选择不再是主 contribution。权威方向以 `docs/safety-layer-spec.md`(尤其 §1 三分支、§2 ε 数学、§3 四贡献)为准。

**★ 真核心突破(里子)vs 故事包装(壳)—— 2026-06-01 workflow `wptoask76` 检验收敛(砍壳+SOTA web+红队三透镜一致,本机重跑测试确认非 vaporware)**:
- **真突破(审稿人买账的硬通货,paper 重心要押这)**:对**会动的**障碍,在全解析可微 MINCO 优化器**内部**,给出连续时间硬安全证书——每个 Bernstein 控制点锚到**它自己墙上时刻** t_{i,k}=cum[i]+(k/5)T_i 上人的 CA 预测位置,用 supporting-halfspace + 膨胀半径 R=r+d_safe+||v||T_i+0.5||a||T_i^2 证明**整段曲线(非采样点)每个时刻**都清出预测管,**无走廊、无 MIQP、无 SDP**,ALM 端到端解析反传 (q,T),白送 λ 影子价格。
- **gap = 五元交集空地**:连续时间 ∩ 可微 ∩ 动态 ∩ 无走廊 ∩ 每控制点锚自身时刻的凸包硬证书——各轴单独都被占(GCOPTER 软采样罚/EGO 软场/FASTER 走廊+MIQP/SOS 静态+SDP/Freire-Xu 静态B样条凸包+CBF-QP/CBF 优化器外 QP-filter),凑齐**无单篇重合**。窄而真。
- **「按类选机制」(per-class)= 壳/故事包装**:好讲(social-nav)、当 framing + 一组消融,但单独是个 if 分派、撑不起顶会。名字可继续用「机制选择」,但 **paper load-bearing contribution 是上面那个动态连续时间证书,不是 per-class**。
- **最危险对手 = 时空走廊 / Safe-Interval(arXiv 2409.10647, 2024;swarm 2106.12481)**:也保动态连续时间安全。唯一硬区别:**它要先建走廊前端、对整段用单一时刻;你不建走廊、锚每点自身时刻**。→ **必做正面 head-to-head 实验**(corridor-free vs corridor-based 成功率/时间/绕路 + per-point-time vs segment-single-time(代码已有 _seg_rep_time 退化臂)泄漏率),否则审稿人判「时空走廊已解决」=整个 gap 站不住。**这是顶会成色的头号待办。**
- **两个必须主动认的软肋**:① 静态凸包安全不新(Bernstein/Freire-Xu 已做)→ 创新明确押动态维,别吹静态;② 硬保证只在 tau_trust(0.75s)信任窗口内,窗外降软场 → abstract 老实标注,讲成「预测不可信不装硬保证」的设计美德(_trust_mask 已实现=证据)。
- **非 vaporware 实证**:本机重跑 stage3_minco_grad 24/24 + perclass_grad 17/17 + perclass 38/38;消融拿掉就崩(DROP-T 0.41、DROP-dgdt 0.46、未膨胀漏 1.21、soft-only 撞穿 0.77<0.8)证每零件 load-bearing。
- **★ 2026-06-01 升级(算法 novelty workflow `ww8qldyt6`,关键修正)**:确定性连续时间证书现状 = borderline 偏下(够 RA-L/IROS,顶会会判 incremental)。原因:① 数学浅(凸组合+Cauchy-Schwarz+三角,三个教科书一行拼接);② **确定性版交集已被占**(arXiv 2505.11376 时变分离超平面+Bernstein 连续时间避障、2404.16826 Successive Convexification 节点间连续时间+动态障碍)——之前以为的"五元交集空地"确定性版**不空了**;③ per-point anchor 被自己实验枪毙(无 consequence)。→ **必须升级成 conformal 不确定度的连续时间 over-interval coverage 证书**:定理"每段裕度≥0(裕度带时变预测带 Σ(t)/conformal q_α)⟹ Pr(∀t∈段 无碰)≥1-α"。**算法深度在"连续时间 ∀t × 概率"耦合**——逐点 chance 合取≠整段 chance,naive union bound 过保守=不够,要真归约论证(把 per-time coverage 抬成 over-interval coverage)。**主打 conformal(distribution-free,非高斯,非 κσ padding)**;高斯 chance-constraint 当 baseline。六元交集"动态∩连续时间∩可微∩无走廊∩per-arrival-time∩分布无关不确定度"空地(conformal 那派 Lindemann 2210.10254/SIPP+conformal 2511.18170/Diff-Opt+conformal 2605.16327 都差一维)。**这同时补最大洞(条件性保证)。** 风险:union bound 过保守 / conformal 需 exchangeability 但在线残差非可交换(要 adaptive conformal ACI)/ CA 短 horizon 够准时优势测不出(实验要造预测噪声场景)。防御地图见 [[sando-py-defense-map]]。**★定理已推通 + 合成验证过(coverage 0.9008、比 union 紧 1.60x),见 [[sando-py-conformal-cert]]。**

---

**核心创新故事包装(坚持当 framing,2026-06-01 塔菲大人拍板)**:**按障碍的语义类别选避障「机制」本身,而不是选权重**。
- 人(物体)= 凸包 supporting-halfspace 硬约束 + ALM + **连续时间安全证书**(整段曲线清出人球,非采样点惩罚)+ **λ 影子价格可解释** + Stage-4 时空预测(在正确时刻躲)。
- 墙(环境)= ESDF 软场(见 [[sando-py-esdf-sfc-decision]])。
- 两者共用同一个 signed-distance oracle 接口 + 同一个 (d_safe-d)^3 + 同一个 MINCO 解析梯度框架(**一个框架、两个 backend、两档机制**)。全解析可微、无 MIQP/Gurobi。

**wedge(没人占的组合)**:SANDO 硬但类别统一 + MIQP;EGO 全软;RA-Nav(2026)per-class 但全软无保证 → 本工作 = per-class **选机制** + 对物体类**连续时间硬证书** + **可解释** + **无整数规划**。

**主轴 = 机制选择(2026-06-01 选定)**:abstract 第一句侧重「选机制不是调权重」。

**主轴最大软肋 + 挡法**:「选机制」易被审稿人贬成工程拼凑(if 人 then 硬)。挡法:机制由**类别的安全语义**决定(物体撞了致命→硬+证书;环境可蹭→软),不是 ad-hoc;且对应真实感知栈两条流(语义检测出物体 / RGBD 稠密占据出环境)。理论重量压在硬那半边(连续时间证书 + λ 可解释 + 无 MIQP)——「选机制」是 representation 的壳,证书+可解释是里子。

**名字:待定(TBD)**。曾用工作名 CLASP(Class-conditional Avoidance with Selective mechanisms and Proofs)、CLASP-CH(convex-hull certified)、备选 CCAM(Class-Conditional Avoidance Mechanisms)。塔菲大人 2026-06-01:idea 坚持,名字以后再定。

进度/路线见 [[sando-rgbd-plan]];当前在还的占位债见 [[sando-py-sim2real-fakes]]。
