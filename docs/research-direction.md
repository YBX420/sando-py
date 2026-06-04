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

---

## 11. 综合裁决:锁定「A 主攻 + B 前端」(2026-06-04,第 5 个 deep-research:时空穿移动缝 SOTA + novelty gap)
> 24 篇主源(UPenn GRASP / TU Delft AMR / MIT-ACL / HKUST / JHU)、25 条断言 3 票对抗验证(23 确认 2 枪毙)。**结论:确认并收紧 §0–§10 —— 主攻 A(per-class anytime-feasible 证书),B(timing-aware 时空拓扑)降级为前端配件,C 为最终落地。**

**新增的承重证据(literature 没有,你这两天实测独有 = 最强 motivation 图):**
- **「算力越多越差」**:budget 给足 → 完全收敛的解反而在密集移动 clutter 卡死(avg 0.7,90% 无效);anytime 截断在中间迭代反而能走。这是 optimize-then-check 架构坏掉的铁证 → 直接论证「要的是可证明可行的中间解,不是收敛解」。
- **超速→validity gate 拒→gatekeeper 刹停**:卡死的直接根因 = 收敛解不可行被后验拒。**anytime-feasible(每迭代可行)从构造上消灭这个拒绝** → §7 实验的 before/after 对照天然成立(before=卡死,after=穿过)。
- 实测可行配置:dyn_base_inflation_m=1.2 + minco_w_time=2500 + v_max=12 → **持续巡航 8.48 m/s 穿密集移动场**(安全 clr+1.15);但高 w_time/v_max 时**解算崩(plan_minco 抛 all-seeds-failed)** → 印证「需要可行性恢复 / 可行种子」这条命门。

**三轴 novelty gap(每个 SOTA 只占两格,空交集 = 你):**
| 方法 | 时空穿移动缝 | 硬安全(证书级) | anytime-feasible <50ms |
|---|:-:|:-:|:-:|
| T-MPC++ (2401.06021) | ✓ | ✗ 软 | ~ 截断**选优**,非逐迭代可行 |
| MIGHTY (MIT-ACL 2511.10822) | ✓ | ✗ 软 EGO 线 | ✗;硬件 6.7 m/s 静态 |
| SS-QCQP (JHU 2511.19675) | ✗ | ✓ 离散时间前向不变 | ✓ |
| P-CBF (2606.00297) | ~ horizon | ✓ | ~ |
| ST-GCS (IROS'25) / SSC (HKUST RA-L'19) | ✓ 硬走廊 | ✓ | ✗ 秒级离线/多机预约 |
| SPOT (2602.01189, 2026-02) | ✓ 4D | ✗ 软 + 纯静态势场兜底 | ✗ |
| **你(目标)** | ✓ | ✓ | ✓ |

**A 与 B 互锁(不是两个独立大工程)**:A 的命门是「SS-QCQP 要每拍可行初值 g(x⁰)≤0」;**B(时空拓扑向导)正是喂可行种子 + time 准移动缝的前端**,且直接复用现有 detour/topology 扩到时空即可,别在 B 上猛投 novelty。报告 open-Q#3 原话:**「没有任何文献把 per-class 差异化机制和 anytime-feasible 求解器结合」——这就是最独有的空位**。

**收紧后三篇必读**:SS-QCQP `2511.19675`(anytime-feasible primitive + 修 Maratos 丢不变性)、T-MPC++ `2401.06021`(时空 H-signature 拓扑,有开源 tud-amr/mpc_planner)、SIMP/UTVD `2409.10647`(穿移动缝的时空拓扑等价判据,UPenn Kumar 组 ICRA'25)。补充:P-CBF `2606.00297`(整段 horizon 硬证书,和你 ALM 证书哲学契合)。

**第一步 = §7 实验**(不变):把一次 ALM 内层换成 feasible-direction + deadline 截断,量(i)任意截断时刻人净空 ≥ 阈值、(ii)anytime 版穿过去而 optimize-then-check 卡死(对照基线已有)、(iii)5 行人 <50ms。

### 11.1 第一个结果(2026-06-04,原型已跑通 → 核心主张证实)
- **探针**:`python/sando_py/local/local_opt.py` 加了门控 `_ALM_OUTER_TRACE`(默认 None,零开销,golden 安全)记录每个外层 ALM 迭代的 (max_g, x)。
- **BEFORE**(`python/_exp_anytime_margin.py`):密集移动行人下,当前 optimize-then-check 的 ALM **每个迭代都人体不可行**(min_clr<0,在撞人),first-certified-feasible = None → 早截断没有可 commit 的安全轨迹。**额外挖到**:冻结法向证书会报 max_g<0(声称可行)而真实 min_clr<0(在撞)—— 现有 validity gate 的 soundness 漏洞,待复核。
- **AFTER**(`python/_exp_anytime_feasible.py` → `media/_exp_anytime_feasible.png`):实现了 **feasible-direction 内层原型**(SS-QCQP-lite:方向 u=argmin ½‖u+∇f‖² s.t. ∇gₐ·u ≤ −α gₐ via SLSQP + 回溯到真实可行)。从一个可行起点出发:**feasible-direction 的每一个迭代 min_clr ≥ d_safe(绿线全程在阈值上,边优化边保持)**;而 plain cost-descent(当前内层本质)第 3 迭代就跌破 d_safe(橙线进不安全区)。**= computation-invariant 行人安全这条主张的第一张实证图。**
- **原型 caveats(诚实)**:① 约束 Jacobian 用有限差分(非 <50ms 级)→ 下一步要用 MINCO banded 结构的解析 ∇g;② 静态行人 + 手工可行起点 → 下一步上动态行人 + 可行性恢复(命门 open-Q#1:每拍可行 warm-start);③ 还没接进生产 `_alm_solve`(门控接入是后续)。
- **下一步顺序**:(1) 解析 banded ∇g 替掉有限差分 + 量单解 ms;(2) 动态行人 + 用拓扑/上一帧解做可行 warm-start;(3) 门控接进 `_alm_solve`(默认关→golden 安全)+ 闭环测「截断率 vs 行人密度」「截断后净空恒≥阈」两条防御曲线。

### 11.2 A+B 集成原型:卡住问题(机制级)已解(2026-06-05,`python/_exp_solve_stall.py` → `media/_exp_solve_stall.png`)
动态密集行人场景(3 个横穿走廊的行人)端到端跑通 A+B,**当前求解器撞人/失败处 A+B 安全穿过**:
- **B(time-aware 拓扑种子)**:每个路点按「它将抵达的时刻」选过障侧、取偏离最小的清空 y(紧贴缝穿)。
  实测 **time-aware 种子 min-clr=0.766 vs static-snapshot 0.049** —— 量化了时空价值:静态快照选错侧正是当前规划器卡死的根因。
- **Phase-1 可行性恢复**(`restore_feasible`,梯度下降总违反量)把种子推进可行集(0.984 ≥ d_safe)—— 堵 A 的命门 open-Q#1(每拍可行 warm-start)。
- **A(feasible-direction)+ 走廊引导**:紧贴缝穿过,**每个迭代 min 人体净空恒 = 0.800 ≥ d_safe**(任意截断都安全);对照当前 `_optimise_one_minco` 撞人(min_clr=−0.14, valid=False)。
- **意义**:这是「绕行=局部穿不动」这个症状的**机制级解** —— B 给对的 homotopy 可行种子破 pocket + time 移动缝,A 保证截断安全。证明了路线成立。
- **离生产「完全解决」还差(诚实)**:① 单次开环解、3 行人(非 120 障碍闭环);② 有限差分 Jacobian(非 <50ms);③ 没接进 receding-horizon 闭环。→ 生产化 = §11.1 的 (1)-(3)。
