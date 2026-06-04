# 研究方向:Deadline 下硬保证行人裕度的 per-class 实时规划

> 2026-06 定。来自 4 个 deep-research(landscape / NN-gap / NN-vs-确定性决策 / 范式 roast),凡验证器没被限流杀掉的结论都过了 3 票对抗验证。
> 一句话:**把局部求解器从「optimize-then-check」改成「safe-by-construction」——让 replan 硬 <50ms,且截断/超时只降轨迹质量、可证明绝不降低行人净空裕度。**

---

## 0. 裁决前提
- **NN warm-start 不是最优解**(已确认)。它只给 PAC-Bayes/残差(平均情形)保证,没有硬 <50ms / 硬可行证书;learned warm-start 在**密集人群**(恰恰是触发多起点的 regime)最容易落不可行、甚至不如朴素「上一帧解」warm-start(JMLR 2024 自己的消融)。审稿人易视作工程。→ NN 至多当**非承重加速器**。
- **越炫的范式越软**:cuRobo/GPU 的 <50ms 靠**软避障**(NVIDIA 自己说 "approximate constraints as cost terms",99.8% 经验,记录过 success=True 仍碰撞);扩散类同理。**违反「绝不牺牲行人裕度」红线,不能当主架构。**
- 唯一稳的方向:**确定性、证书优先**。

---

## 1. 现在的系统(baseline = 你自己,SANDO/MINCO 线)
`heat-A* 全局` → `per-class MINCO 局部(人=硬 ALM 凸包,墙=软 EGO)` → `detour 6 种子多起点` → `后验 validity gate` → `receding-horizon commit`。
痛点:密集行人区单种子解不出 → 退回 **detour 6 解(1 detour-off + 5 detour-on,每解 ~30–70ms)** → **~200ms 尖峰,~10% replan 超 50ms**。

## 2. 目标架构(只改承重的局部求解器,全局/per-class/MINCO/receding-horizon 全留)
| 现在 | 换成 |
|---|---|
| detour 6 种子盲搜(±u/±v 绕最近行人挑最优) | **H-signature 确定性选 passing-side**:绕数积分(时间嵌成一维)算出每个行人左/右绕,直接构 **1 条**正确 homotopy 类的种子 |
| ALM 内层 optimize-then-check(迭代可不可行无所谓,末尾才查) | **anytime-feasible ALM**:人体凸包约束**每步迭代都满足**(可行集前向不变),deadline 一到就停,停在哪都可行 |
| 后验 validity gate + 解不出隐式 hold | **显式 gatekeeper**:只 commit 验证过的轨迹 + 一条**进入不变安全集的刹停 backup**;超时/不可行就执行 backup(可证明安全) |

## 3. 和现在的差距(诚实 diff)
**已有 ~70% 零件**:per-class 分派、ALM 凸包约束、净空证书(validity gate)、MINCO banded 解、heat-A*、能跑的 C++ 引擎。
- **差距 1(最大,真算法活)**:证书现在是**后验检查** → 要变**前向不变**(求解全程成立)。需把 ALM 内层从「LBFGSpp 自由优化 + 末尾查」改成**可行方向 / safeguarded 步长,每步不越界**(对标 SS-QCQP 的 backtracking)。
- **差距 2**:盲搜 4 方向找绕行侧 → **H-signature 确定性算**(少解、可解释)。
- **差距 3**:解不出**隐式 hold(无可证明安全 backup)** → **显式刹停 backup**。
- **差距 4(命门,必须真数据堵)**:**<50ms 对谁都还没证过**(SS-QCQP 唯一实验 49–186 秒)。要在自己 C++ 栈实测 anytime-feasible 单解在 5 行人下能否进 50ms。**这是 paper 成立与否 + 你比所有 paper 都强的点(你有真栈)。**
> 差距不在「重写系统」,在「把可行性从结果变成不变量」+ 确定性拓扑 + 安全 backup。

## 4. 创新性 + 对手划界
| 对手 | 它有 | 它**没有**(你的洞) |
|---|---|---|
| **SANDO**(你 baseline) | per-class、MINCO | 无 deadline 保证、optimize-then-check、盲搜多起点 |
| **T-MPC++**(T-RO'24,最危险) | 确定性 H-signature 拓扑 + 每拍占优证书 | **软代价**、**仍跑并行多解**、地面机器人、**无 deadline 下逐行人硬证书** |
| **SS-QCQP**(arXiv 2511.19675) | anytime-feasible 可行不变 | **不实时(49–186s)**、非 per-class、非 UAV/动态行人、要可行起点 |
| **gatekeeper**(arXiv 2211.14361) | deadline-safe commit | 通用系统、**没接进 per-class 规划器 / 没做到优化器层的可行不变** |
| **cuRobo / 扩散** | GPU/生成快 | **软避障**,违反红线 |

**novelty = 三件拧成一个 per-class 动态行人 UAV 规划器 + 没人给过的保证**(anytime-feasible 优化器层 + 确定性拓扑决策层 + gatekeeper commit 层)。命中你四个轴:②可解释(passing-side 人能读)③安全裕度(前向不变证书)①动态反应(<50ms 硬)。

## 5. 最锋利的一句 claim
> 第一个在**硬计算 deadline**下保证**逐行人硬净空裕度**的实时无人机局部规划器——通过让 human-class ALM 约束**前向不变(anytime-feasible)**、并**确定性**选 passing-side 拓扑,使 50ms 超时只降平滑度、**可证明绝不降低行人安全裕度**。

## 6. 诚实风险 + 防御
- 审稿人攻击:「不就是 SS-QCQP + T-MPC + gatekeeper 拼起来?」
- 防御必须有三样:**(a)** 真算法贡献——anytime-feasible 怎么在 **per-class ALM + MINCO banded 结构**里高效实现(利用带状结构做可行方向 QP = 技术肉);**(b)** 真实时数据——你证了 <50ms 而 SS-QCQP 没有;**(c)** 前人没有的曲线:「deadline 截断率 vs 行人密度」「截断后净空恒 ≥ 安全阈」。
- 没 (a)(b) → 掉成「工程集成」;有了 → RAL/ICRA 级硬贡献。

## 7. 第一步实验(先堵命门,再写 paper)
在已搭好的 C++ 栈里做**最小可行性验证**:
1. 把**单条 ALM 解**改成「**可行方向 + deadline 截断**」,实测:① 任意时刻截断后**人体净空 ≥ 安全阈**(前向不变成立);② 5 行人下能否进 **<50ms**。
2. 用 **H-signature** 选 homotopy,看能否把 6 种子塌成 **1–2 个**。
3. 出两条防御曲线(截断率 vs 密度;截断后净空)。
→ 同时(i)堵唯一致命窟窿,(ii)产出 novelty 防御里最硬的数据。

## 8. 关键文献
- 反 NN:Sambharya/Stellato, "Learning to Warm-Start Fixed-Point Optimization", JMLR 25 (2024).
- anytime-feasible:SS-QCQP, Wang & Fazlyab, arXiv 2511.19675(2025-11 preprint);safe gradient flow, Allibhoy & Cortés, arXiv 2204.01930 / IEEE TAC 2024。
- 确定性拓扑:T-MPC / T-MPC++, de Groot/Ferranti/Gavrila/Alonso-Mora, IEEE T-RO 2024, arXiv 2401.06021。
- 安全后备:gatekeeper, arXiv 2211.14361。
- 反例(软安全,别上):cuRobo, NVIDIA, arXiv 2310.17274。
- NN 若保留当加速器,要划界:LISCO 2409.08066(学 primal-dual)、NEO-Planner 2309.10683(学种子+时间)、GNN active-set DAQP 2511.13174。

## 9. 一句话路线
**先做 §7 的可行性实验 → 拿到 <50ms + 截断净空数据 → 再按 §4/§5 写骨架。** 倾向先实验后写,把最大风险用真数据消掉。

---

## 10. Nature 级扩展裁决(2026-06,3 个 deep-research:A 计算无关安全 / B 学习×证书 / C 人群可信)
**诚实前提**:从一个 planner 出发,Nature/Nature-MI 是 stretch;现实高影响天花板 = **Science Robotics(系统+真实世界+一条干净原理)** 或旗舰 T-RO/RAL。三条方向都是「积木已证、但统一/组合真空且未命名」,都卡在 Nature 门神「demonstration 广度」。

- **① A 计算无关安全(最佳载体)** —— 最深、**真没人命名**「safety decoupled from computation」(最接近的 Hsu/Hu/Fisac, Annual Review 2024 只统一安全**机制** CBF/HJ/MPC/Lyapunov,对算力/deadline/中断只字未提)。**真空且打你强项**:没有通用定理把「anytime-feasible 优化器 ⊗ 证书后备(gatekeeper, Panagou T-RO'24)」在**任意截断**下耦合、覆盖一大类 embodied 系统;每个后备都吃「backup/不变集存在性」非普适假设(gatekeeper 作者自承首要局限);最接近的 Pant TCST'21 把 anytime 放在估计器而非控制优化器。**你的栈 = 第一个实例。** 风险:耦合定理浅了就是「RAL 穿衣服」——那条定理本身必须是真新的。
- **② C 人群中可信(最高影响、最重 demo)** —— 真空(认证∧可读没人占;SHINE/IJRR 是「学了 homotopy 但没认证」;social-nav 52 作者共识 2306.16740 把 safety/legibility 当 8 条并列原则分列)。代价:真实人群现场 demo + 领域无 benchmark。**最好当 A 的影响外壳**:「围着人的、计算无关的安全」。
- **③ B 学习×证书(别当主攻)** —— 真空但实例层太挤(FOSSIL/CEGIS TACAS'21 是最干净实例:SMT 验证器证过才接受,与训练质量无关,但只在抽象 ODE)+ **有人公开唱反调**(PPS arXiv 2506.05171:硬证书扩不到 embodied AI,建议改概率安全)。把 neural-certificate 当**工具**不当 thesis。

**最锋利的 Science-Robotics/Nature-野心 claim**:*安全不必拿算力换——我们提出并证明 computation-invariant safety:具身机器人的硬安全保证(行人净空)在任意 real-time deadline / 中断 / 算力削减下可证明地不退化,并在人群中高速飞行里证明算力被掐时安全裕度一寸不让。*

**逐级门槛**:RAL/ICRA = 你的栈 + 单域 + 两条截断不变曲线(现在就能);Science Robotics = + 命名原理 + optimizer⊗fallback 耦合定理 + 有说服力的(近)真实人群 demo;Nature = + 跨域(无人机+车/机械臂)**或**深到意外的通用定理。
**关键**:不管冲哪级,**第一步实验都是 §7 那个**——它既是会议论文核心,又是证明这条原理的第一个实例。
