---
name: sando-rgbd-plan
description: "sando-rgbd 当前定版计划(2026-05-25):放弃 Gurobi/MIQP/走廊,改 B-spline+梯度的 per-class 软/硬场;按管道顺序分阶段构建+验证。取代 probe 锁定的旧设计。"
metadata: 
  node_type: memory
  type: project
  originSessionId: 606a95c7-0a2c-44df-8c70-61a6b1f56252
---

# sando-rgbd 新架构 + 分阶段计划(定于 2026-05-25)

## 一、大方向(相比 probe 锁定旧版的重大调整)
基于 SANDO 做新规划器,但**放弃 Gurobi / MIQP / 凸走廊**,局部改用 **B-spline + 梯度优化**。
核心创新升级:per-class 不再只是"不同膨胀数",而是**"不同躲避机制"——人用硬约束、墙用软推**(融合 SANDO 硬约束 + EGO-Planner 软力场,由障碍类别决定用哪套)。
纯 Python 做研究/调 ML;上飞机时把学到的参数搬 C++/numba,**Python planner 不上机**。

## 二、为什么放弃 Gurobi/MIQP
- 部署:Gurobi 是商业 license,上机麻烦/要钱。换掉才能免费上机(凸 QP 用 OSQP,可 codegen 成 C)。
- ML:MIQP 的整数变量不可微,挡住端到端学习。
- 那个整数决策(选走廊/从哪侧绕)= 绕行拓扑,heat-A* 全局规划器已经做过了,MIQP 是重复劳动。

## 三、数据流(每拍 replan)
1. 全局 hgp:heat-A* → 粗路(只给大方向)
2. 局部 local_opt:粗路 + 障碍 → 光滑/不超速/按类躲的曲线
3. 交控制器/仿真执行
动态障碍:每段曲线躲它**在那一刻的预测位置**;复用现有 obstacle_tracker(EKF 预测)。

## 四、local_opt 设计(最简第一版)
B-spline + scipy L-BFGS + 每障碍"离近了就罚"的罚分,力度/安全距离按类别变(人狠墙轻)。
- 第一版"硬"也先用很硬的罚分(纯梯度,好写);之后再升级真硬约束(OSQP)。
- 文件:local/bspline.py(曲线数学)、cost.py(打分)、avoid_config.py(类别→软硬/距离/力度表)、local_opt.py(铺初值→优化→出曲线)。

## 五、前端(全局)决定
- 暂用 heat-A*;别过度投入,它只是向导。
- 只有 local_opt 老卡住/出不可行路才升级 kinodynamic A*(场景越快越可能要);中间挡 = 给 A* 路配时间让初值飞得动。
- 嫌格子锯齿 → Theta*;嫌每拍重算慢 → D* Lite。后面再说。

## 六、分阶段构建(一段段写+单独验证,最后串)
1. 地图+热度 voxel_map [已有 → 验证:占格对不对、障碍附近热度高]
2. 全局 A* graph_search+hgp_planner [已有 → 验证:空地直线/留缝穿缝/堵死无路/热度绕开/对 Dijkstra]
3. 局部 local_opt [新写 → 验证:顺/不超速/躲开/对人比对墙远]
4. 串一拍 planner.py [验证一次 replan 跑通]
5. 接仿真闭环 nodes+fake_sim+obstacle_tracker(加 class) [验证真飞]
6. 打分脚本 [撞没撞/顺不顺/离人多远]
原则:每段配独立小测试脚本,喂假输入查输出;阶段 1~4 不用 ROS/仿真。**从阶段 1 开始。**

## 七、sando-py 现状/坑(开工前知道)
- 没验证过:smoke test 只测 import,没跟 C++ 数值对照。
- 要删:solver_gurobi.py;hgp_manager.py 里 cvx_ellipsoid_decomp(切泡泡,逻辑可疑)。
- (C++ 用 MINVO + 共享 Gurobi env,Python 用 Bezier + 每次新建 env——要删 solver 了,无所谓。)
- 代码在本机 /home/boxuan/code/sando_ws/src/sando_py/(注意:mammoth /data/boxuan/sando_ws 下没有 sando_py)。

## 八、感知 / 数据
- 仿真先用**真值类别**;YOLO oracle 后接。
- 类别框先用 bbox(现成);人贴墙过度躲再升 mask。
- 训练数据:mammoth 的 Open Images(oid4)。规划本身基本不用数据集,靠仿真跑场景打分。

## 九、可选(以后加)
CMA-ES 调参 / tiny MLP 按情况调数 / 接真 YOLO / mask / 余量随预测不确定度变大 / 硬安全兜底(CBF) / 上机 C++ 或 numba 提速。

## 工作规矩
sando-rgbd 这条线**每步先问再做**,不要连做多步;回答用大白话(见 [[feedback-plain-language]])。
mammoth 上有同名计划 [[sando-rgbd-plan-v2]] + 更全的旧背景 memory。

## 决策(2026-05-29,用户拍板)
- **优化指标(超越 SANDO 的"更好"= 这四个)**:① 动态反应性(在正确时刻躲动态障碍,尤其人——per-class 硬约束主战场)② 可解释性**本身当 paper 卖点**(可视化每步为什么这么躲,不只 debug)③ 安全裕度/成功率(离人更远、撞击率更低)④ 轨迹质量(平滑/省力/时间)。
- **项目范围 = 博士主线完整新框架**,顶会为目标。→ **不走 pragmatic 修补,走完整新框架**(值得 MINCO/解析梯度 + per-class 硬软完整做)。
- 连带要考量(2026-05-29 讨论,尚未逐一拍板):公平对比(Python 不上机 vs SANDO C++ 上机,时间没法直接比)/ per-class 感知错误+不确定度(分类错=撞人,置信低默认硬躲?)/ 硬约束无解死锁的 fallback(对标 SANDO HOVER_AVOIDING)/ 动态预测不确定性随时间放大安全管 / ablation 需求倒逼架构"每个机制可一键开关" / 上机认真程度影响骨架选型(避重型自动微分库)。

## 架构方案 CLASP-CH(2026-05-29 workflow 产出 + 用户拍板)
19-agent workflow(survey→5设计→4评委+5红队→综合)收敛的方案。名 CLASP-CH(Class-conditional Avoidance, convex-hull certified),名可改。
- **核心新颖点**:per-class 选**机制**(不是权重)。墙=软 EGO 场;人=**硬:每段控制点清出"人球"→凸包性质保证整条连续曲线段清出(真连续时间证书,不是采样点惩罚)+ ALM 外循环**。λ 乘子=安全约束影子价格,可视化"为什么这里被推开这么多"=可解释卖点。wedge:SANDO 硬但 class 统一+MIQP;EGO 全软;RA-Nav(2026)per-class 但全软无保证 → 你=class 选机制 + 硬半边连续时间保证 + 无 MIQP/Gurobi。新颖性 representation-agnostic(B-spline/MINCO 都能承载)。
- **用户拍板(2026-05-29)**:① backbone = **直接上 MINCO**(不走 B-spline-first;接受 big-bang 重写风险)。② 起步 = **先做速度地基**。→ 合并:MINCO 下"速度"="MINCO 核心+解析梯度",二者合一。原 Stage 0(B-spline 基矩阵加速)作废;**bspline.py+84测试不删,降级为交叉验证 oracle**。
- **重定义起步(MINCO)**:M0=min-jerk quintic 核心 `c=M(T)⁻¹b(q)`+求值器,`MinjerkTraj` 复用 `UniformBSpline.eval_deriv` 接口(下游无感);M1=解析梯度穿 min-jerk 映射(`∂c/∂q`、`dM/dT`),**头号高风险件(off-by-one 静默腐蚀梯度)**。三道闸:bspline oracle 交叉验证 / check_grad<1e-6(避尖点) / 红队审索引。MINCO 红利:精确穿 waypoints,Stage 2 保拓扑 seed 更稳。
- **6 阶段**(去风险序):0+1 速度(→MINCO M0/M1,解 invariants>400s 超时)→2 拓扑 seed+自由度+bug-iv 时间(对称破缺是 seed 的活,ALM 做不到:实测直穿侧向梯度 6.8e-15)→**3 per-class 凸包+ALM 硬人+软墙(贡献核心)**→4 动态人(EKF 是 CA 9D 出 quintic poly,非代码假设的 CV 球,要适配器;硬约束只在 EKF 可信窗 ~0.5-1s,外退软)→5 接管道+class 标签源(EKFState 现无 class field)+删 gurobi。
- **红队逼出的 3 修正**(所有原始设计都犯):per-segment τ 破坏均匀节点导数恒等式→保标量 dt+事后 1D 时间重分配;采样点 ALM 非保证(段间会垂下穿人)→凸包连续时间证书;ALM 解不了直穿卡住→拓扑 seed 负责。
- **待用户拍的 open Q**:headline 指标具体哪个数 / class 标签源(启发式规则 vs 真分类器,新颖性命脉)/ EKF 预测 v 被 clip 到 ±0.5(1.5m/s 的人被低估,worst-case 余量不可信,要不要独立 v_max_human config)。
- 完整 workflow 输出存于 transcript:`tasks/w1mqdtadi.output`(116KB JSON)。

### MINCO 地基 M0+M1 ✅ 完成 2026-05-29(workflow wfj5n0khd,4 agent 实现+对抗验证)
- 新文件 `sando_py/local/minco.py`:`MinjerkTraj`(s=3 min-jerk quintic,banded `M(T)`,`eval_deriv` 复用 UniformBSpline 接口)。M(T) 逐字照 GCOPTER `MINCO_S3NU`,系数 ASCENDING 存(vs GCOPTER descending,下游读 .c 注意)。
- **M0**:81/81 测试(`test/stage3_minco*.py` 5 文件)+ bspline 回归 84/84 不变。红队 verdict=pass、无 bug。三重独立 oracle 验到 ~1e-13(闭式 min-jerk / 独立 Hermite 6x6 / 相邻多项式连续);精确穿 waypoint 1e-13;banded O(M)(M=800 banded 11.8ms vs dense 18s,~1500x)。红队加证:只约束 C0/C1/C2+waypoint+BC 的真 min-jerk QP 重现 MINCO 系数到 8.95e-13 → C3/C4 连续是 KKT 最优条件,解是真·minimum-jerk。
- **M1(头号风险件:解析梯度)**:`energy_grad()` + 通用 hook `grad_from_dcost_dc(dcost_dc, dcost_dT_explicit)`。24/24 测试。**生死线门禁过:dJ/dq vs FD rel=4.97e-08、dJ/dT rel=1.35e-09(都 << 1e-6)**。红队 verdict=pass、无梯度 bug:用**复数步微分(无截断)**独立验到机器精度(dJ/dq 7.7e-11、dJ/dT 2.7e-13)+ symbolic 验 (dM/dT)c bit-exact;转置方向实证(M^T 残差 1.5e-14 vs M 7.8e3);时间梯度两项都在(explicit + 隐式 -λᵀ(∂M/∂T_k)c,无丢项)。O(M) backprop。
- **注意/待接**:① 通用 hook 的 `dcost_dT_explicit` 必须由 cost provider 提供(采样点随 T 重参时),否则静默少算 dCost/dT——M2 接 obstacle cost 时要 wire。② 还没接进 cost.py/local_opt.py、没加 __init__ 导出(纯地基,测试用 importlib)。③ 极端段时长比(>1e6)丢精度(MINCO 本征 cond~ratio^5),优化器保持时长比合理。
### M2 接进求解循环 ✅ 实质完成 2026-05-29(workflow w4ix9fclt;框架报 failed 但只是 verify agent 撞 API 500 中断,代码+测试已落地)
- **代码**:`local_opt.py` 加 `plan_minco(astar_path, obstacles, avoid_cfg, ...)`(additive,旧 `plan()` byte-for-byte 不动,回归 17/17+15/15)。决策变量 `x=[q.ravel(), T]`,端点结构性 pin(`from_endpoints`),T_i≥dt_min box,`jac=True` 解析梯度。cost = w_smooth·energy() + w_obs·obstacle + vel/accel hinge + time anchor。采样 = per-segment 中点 quadrature(GCOPTER 风格 κ=16)。smooth 走 energy_grad,obstacle/vel/accel 汇进**一个** grad_from_dcost_dc adjoint。
- **explicit dCost/dT 项**(头号坑)实现了 4 个子项:(a) quadrature 权重 (1/κ)Σφ、(b) moving-local-sample (T_i/κ)Σφ'·s_j、(c1) same-seg abs-time、(c2) cross-seg abs-time(加到所有上游 k<i)。
- **梯度门禁过**:check_grad(Richardson O(h⁴),120 smooth config,kink 排除)dq=5.02e-10、dT=8.65e-11、full=8.71e-11;cross-seg c2 专测 6.54e-10。**红队 ablation:丢掉 c2 行 → 门禁炸到 5.70e-3/4.36(门禁非空、explicit dT 确实 load-bearing)**。
- **速度**:单 seed 16.1x(0.62s vs 10.04s)、per-iter 23.5x;原 >400s 超时场景现在 75s(43 个 detour solve),单 solve 2.74s。
- **直穿球心 R.A**:obstacle cost 1.13e3→0.01,min_clearance 0.765>0,绕开。
- **★ 重要诚实发现(印证计划核心论点)**:dense 场景 + hard human(d_safe=0.8,w=1e4),soft penalty 停在 min_clearance≈0.706,差 0.094 没到 d_safe → trajectory_valid=False。**不是梯度 bug,是 EGO 三次罚分边界梯度消失(φ'→0 as d→d_safe)的 formulation 限制**。→ **这实证了 Stage 3 的必要性**:"hard=大权重 soft"(bug iii)不足以保证对人的硬避让,正是要换 convex-hull+ALM 硬约束的理由。给 paper 一个干净的 ablation 动机(soft-only 会差 0.094m 进安全区)。
- **挂账**:AABB 内部 L∞ 非光滑(次梯度,门禁排除);T 下界 active 一侧梯度(未见问题);cost+grad 还有 over-M Python loop(单 solve 亚秒,batch 75s,numba 缺席)。verify 独立 complex-step 红队被 API 500 中断未跑完(build 自做了 ablation+Richardson FD,已较强);可 resume workflow 只重跑 verify 补独立验证。
- **下一步**:M2 复核全绿(门禁 9/9、旧 plan 17/17、M1 24/24、real 14/14)后,用户拍板**直接上 Stage 3**(workflow wvg62jg8w 进行中)。M2 独立红队(complex-step)留待可 resume 补。

### 路线图全景(7 块,2026-05-29 整理)
- **地基**:M0 表示 ✅ / M1 解析梯度 ✅ / M2 接求解循环 ✅。
- **核心算法**:Stage 3 per-class 硬人(凸包+ALM)软墙 ✅(见下详记)+ 向量化提速 3.3x + warm-start + 闭环 RT 反应性测试 ✅(见「RT/速度」记)/ Stage 4 动态人:S4a 时空 per-控制点 + S4b 可信窗硬软 ✅(见「Stage 4」记;梯度+连续时间证书重推已验)、剩 S4c 不确定性膨胀 + S4d 真 EKF(CA 9D quintic)适配(归 Stage 5)⬜ / Stage 2 多绕法 seed 精化(M2 已半接旧 detour)⬜。
- **集成+评估**:Stage 5 核心 = MINCO 接进 planner.py(local_solver 标志 dispatch、默认 minco、Gurobi 留 oracle)✅(见「Stage 5 核心」记);剩 Stage 5 收尾 = 静态体素→软墙源 + 障碍速度接线(S4d EKF 适配)+ 接 sando_node ROS 层 + 删 solver_gurobi.py(待确认)⬜ / Stage 6 打分+benchmark+ablation(全软 vs per-class),出 paper ⬜。
- 顺序逻辑=去风险:危险的"梯度对不对"在地基层门禁锁死 → 最值钱最难的核心贡献 → 最后系统+评估。

### Stage 3 per-class 硬人/软墙 ✅ 完成 2026-05-29(workflow wvg62jg8w,92min;verify verdict=pass,无 must-fix)
- **核心贡献落地**:墙=EGO 软场(原样),人=连续时间硬约束+ALM。Stage 3 新增 60 测试全绿,全套回归绿(MINCO M0/M1/M2 + bspline + cost + 旧 plan)。
- **★ 抓到并修了一个真 formulation bug(本身=研究贡献点)**:design 选的"每控制点离人球心≥R"(per-control-point **ball** 约束)**不是有效连续时间证书**——build 实测控制点全 clear(cert +0.05)但曲线在控制点间垂到 0.667<0.8 泄漏;verify 独立构造泄漏(弧上控制点 r1.02>R1.0 全 clear,40001 点密采曲线垂到 0.877,泄 0.143m)。改用 **supporting-halfspace 形式**(a^T(P−c)≥R,frozen normal a=unit(centroid(P)−c)),Cauchy-Schwarz 严格证明 ||p−c||≥a^T(p−c)≥min_k a^T(P_k−c)≥R → 真连续时间保证。verify:2576 certified 案例 0 违反,收敛轨迹密采 50000 点无泄漏。**paper 点:naive 球约束不够、halfspace 才严格**。
- **梯度门禁过**:ALM 内层梯度 vs Richardson FD worst 7.24e-10(<<1e-5)。explicit dP/dT(C2B@diag(T^j))load-bearing:丢掉 → T-grad rel-err 0.41(M2 trap 重现)。complex-step 验各链节 ~1e-14。
- **连续时间对照(paper 图)**:M2-breach 场景 soft-only 密采 min clearance 0.7021(钻进 0.098m,valid=False)vs hard 1.3526(margin +0.55,max_violation 0,valid=True)。
- **ablation 三臂**:per-class/all-soft/all-hard 人 clearance 区分明显(如 2.293/0.757/2.408);all-soft 复现 M2 breach;fail-safe unknown class→hard 已验。switch=OptParams.avoid_override(None/'soft'/'hard')。
- **STOP**:挡人窄缝→trajectory_valid=False/failure_reason='clearance_violation',从不假报 valid。**可解释证书**:每 active 约束 {clearance,λ,ρ,force},λ 单调≥0,ρ 增长,max_violation→0。
- **挂账**:① 性能单次 replan ~350-440ms(>100ms 目标),瓶颈是 **M2 老代码 _seg_vander**(每内层重算 hinge basis)非 Stage 3 ALM,缓存可解(留上机);② frozen-normal 每外循环冻结(标准 GCOPTER SFC);③ t_rep=段末单一保守时间(静态/固定预测人),完整 per-控制点时空=Stage 4;④ AABB 硬墙仍用保守 per-控制点 surrogate(墙可蹭);⑤ 2-point 直线 seed 穿正中人→detour 退化过保守 STOP(fail-safe,真实多点 A* 路径能绕)。
- 新增/改:minco.py(C2B/control_points/control_points_dT_explicit)、local_opt.py(_seg_normals/_alm_constraints/_alm_term/_alm_solve/hard_clearance/_certificate_margin/_select_best_minco)、avoid_config.py(resolve_mode)。

### RT/速度 + 闭环反应性 ✅ 2026-05-29(用户点出"没以 rt 视角 test",这条线之前全是离线一锤子)
- **先修一个真语义 bug(Stage 3 verdict 层)**:`check_feasibility` 对所有障碍一视同仁判 clearance → 软墙被蹭(按设计允许)也触发 `clearance_violation`/valid=False,per-class 的"软"只在 cost 成立、valid 判定没成立。修法在 `_optimise_one_minco`:clearance 验只认硬障碍(`safety_obs`=override=None 的硬集),软墙蹭了记录但不算不合格,再回落 vel/accel 动力学检查(不碰 `check_feasibility` 本体,不影响旧 plan)。回归钉 `stage3_minco_perclass.py` SOFT.0–6(非空场景:start 钉在距软墙 0.3m< d_safe,必然蹭;naive check_feasibility 判 invalid、修后 valid)。
- **延迟真相**:`plan_minco` 单次重规划 Python ~965ms(detour 关)/ ~3.5s(detour 开)——**根本不实时**(无人机要 5–10Hz)。M2 报的"快 16x"只是相对慢基线,绝对值从没和实时预算比过。瓶颈 **不是算法**(MINCO 是 O(M) 轻),是**没向量化的 Python 逐点循环**:`_basis`(每采样点标量+双重循环,一次规划调 ~9.5万次)、AABB 批量距离居然逐点标量、`_seg_vander` 每段每迭代重建。
- **向量化(逐位等价,梯度门禁全程绿)**:① `_seg_vander` 整段 numpy;② `MinjerkTraj.eval_deriv` 批量(searchsorted+gather+einsum)；③ AABB `_signed_dist_and_grad_batch` 向量化(outside/inside 掩码);④ 固定基缓存 `_fixed_basis`+`_scale_basis`(B(s·Ti)=B(s) 按列乘 Ti^(p-o),s 固定只算一次)。结果 **965→290ms(detour 关,3.3x)、3.5s→1.14s(detour 开)**。check_grad 全程 <1e-9。**延迟分场景**:人远(约束不激活)~30ms,贴身避让(约束紧)~290ms;p95 闭环跑里到 ~260-380ms。再往下要么整段向量化(碰已验梯度)要么减迭代(碰质量)——性价比到此,Python 这层 ~200-300ms 是地板,C++ 部署再快一个量级才是真 RT。
- **warm-start(receding-horizon 状态连续)**:核心其实已支持 v0/a0(`_minco_cost_grad`/`from_endpoints` 都有),只差 plumbing。给 `plan_minco`/`_optimise_one_minco`/`_alm_solve` 接出 `v0,a0`(默认静止)。不接的话每 tick 从静止重规划 → 原地爬到不了目标。接好后闭环能正常推进到目标。
- **新测试 `test/stage3_minco_realtime.py`(10/10)= 真·rt 视角**:滚动重规划 rollout(每 tick 从当前 pos+vel 重规划、commit dt、人按真实速度走、子步密采测真实闭环 clearance)。**铁证(动态反应性指标)**:人横穿、定时在无人机经过时到达路径——**开环只规划一次 min_clr=0.080 撞穿;闭环每 tick 重规划 min_clr=2.19 安全到目标**;4 个横穿场景闭环全 SAFE。RT.warm 验 v0 被尊重;RT.lat 报 median 50ms/p95 261ms/max 377ms(105 次重规划)。
- **demo 场景修好**:`demo_stage3_perclass.py` 改通道场景(人在下顶路径上去、软墙在上盖顶)——硬约束贴住(cert_margin 0.013 出力箭头)、墙被蹭(穿进 0.2m)、valid=True;扫描脚本(临时,已删)选的 human_y=-0.8/wall_bottom=0.6。挂账①(_seg_vander 350-440ms)**已由本次向量化解决**。

### Stage 4 S4a+S4b 时空 ALM ✅ 2026-05-29(workflow wigprxr1n,8 agent ~51min;独立 complex-step 红队 verdict=pass)
- **S4a per-控制点时空**:每个 Bernstein 控制点 P_{i,k} 用**它自己的时刻** t_{i,k}=cum[i]+(k/5)·T_i 去看人的预测位置 c_h(t_{i,k}),取代 Stage 3 的"整段用段末单一时间"。这才是"在正确时刻躲"。新 `MinjerkTraj.control_point_times()→(M,6)`。
- **S4b 可信窗**:t_{i,k}≤`tau_trust`(新 OptParams,默认 0.75s)的控制点硬(ALM),窗外退软(g 设大负、w=0)。trust mask **每外循环冻结**(否则 L-BFGS 里控制点穿越窗边界 → cost 非光滑 → 破 check_grad)。
- **梯度新增 Source 2(头号坑,M2-trap 同类)**:c_h(t_{i,k}) 现在也显式依赖 T(经 t_{i,k}),dCost/dT 多一条时间链:dL/dt_{i,k}=w·(a·vel)=w·dgdt;`dt_{i,k}/dT_j = 1 (∀j<i) + k/5 (j=i)`。实现:同段 einsum(k/5,S)、跨段 **reverse-cumsum**(`dT_abs[:-1]+=cumsum(seg_sum[::-1])[::-1][1:]`)。dcost_dc 无新项(c_h 不依赖 MINCO 系数)。**门禁(我独立复跑过)**:S4.0 总梯度 1.19e-09、S4.1 dT 1.66e-09;红队独立 complex-step(h=1e-30,自写不复用库)总 rel-err 1.55e-13;**非空消融:删时间链→dT 4.6e-1、跨段换"只 i-1"陷阱→2.1e-1**。
- **连续时间证书重推**:per-控制点时空下"单段一个 halfspace 罩整段凸包"失效(每点对应不同时刻人位置)。改**膨胀半径保守单 halfspace**:a_i=unit(centroid−c_h(t_i^0)),R_i=r+d_safe+||vel||·T_i,margin=min_k(a_i^T(P_ik−c_h)−R_i)。S4.5 验 cert≥0 ⇒ 密采无漏(sound 不假安全),S4.6 验未膨胀的 per-node 会漏(+1.21)→ 膨胀必要。膨胀=保守(可后续收紧)。
- **静态人(vel=0)经 `_is_moving` 门走 Stage 3 老路径,逐位一致** → 所有旧测试不回归(additive)。dgdt 在 vel=0 恒 0。
- **非空动机**:5m/s 的人沿走廊,旧"段末时间"式在 t=1.667s 看人(人已移开)→ 漏判,实际无人机钻进人体 dense clearance −0.30;per-控制点时空在 t=1.167s 抓住。
- **测试**:`stage3_minco_perclass_grad.py` 加 S4.0–S4.6(17/17);workflow 自报全套 142/142、0 回归;我独立复跑梯度门禁 17/17 + realtime 10/10 + perclass 38/38 数字逐位吻合。
- **改/新文件**:minco.py(control_point_times)、local_opt.py(_is_moving/_trust_mask/_certificate_margin_spacetime 新;_seg_normals→(M,6,H,3)/_alm_constraints 加 dgdt+trust/_alm_term Source 2/_alm_solve 冻 trust mask/_build_certificates 加 trust 字段;OptParams tau_trust+spacetime_hard)。
- **挂账**:① 证书保守(膨胀半径);② tau_trust=0.75 是占位,真值要按 EKF 可信窗标定;③ 真 EKF(CA 9D quintic)适配 = Stage 5;④ AABB 硬墙仍 per-控制点 surrogate;⑤ 闭环 realtime 测试用的还是匀速预测(S4 让"硬约束"内部变时空精确,但 harness 的人仍 CV)。

### Stage 5 核心:MINCO 接进 planner ✅ 2026-05-29(workflow w1eemyec9,7 agent ~28min;独立复核 pass)
- **背景**:`planner.py`(64KB)是 SANDO 基线的 Python 移植——局部求解用"凸走廊 SFC(cvx_ellipsoid_decomp)+ factor 扫 + SolverGurobi 每 factor + 选 winner"。**纯 Python、0 rclpy**(ROS 耦合只在 `nodes/sando_node.py`),所以可用合成输入离线端到端测。
- **做法(additive,沿 plan/plan_minco 老套路)**:加 `plan_local_trajectory_minco` + `Parameters.local_solver`(默认 `'minco'`,`'gurobi'` 留旧路径作 oracle);`plan_local_trajectory` 顶部 dispatch。**没删任何文件**,Gurobi 体原封不动在下面可达。
- **关键 mismatch + 解法**:`types.PieceWisePol` 是**三次/降幂/局部时间**,`MinjerkTraj` 是**五次/升幂** → 不能直接复用。新 `QuinticPieceWisePol(PieceWisePol)` 只重写 `_eval_axis` 用五次升幂求值器(和 `MinjerkTraj._basis` 同 `**` 幂,bit-parity);适配器 `_minjerk_to_pwp`:`times=A_time+mj._cum`、`coeffs=mj.c.reshape(M,6,3)` 升幂复制(A_time 在局部时间抵消)。**eval 等价 1.78e-15**(<<1e-9 门)。
- **执飞契约**:真正飞的是 `self.goal_setpoints`(List[RobotState],按 `par.dc` 采样,t=A_time+t_local,填 pos/vel/accel/jerk)——`append_to_plan` 只读它;`pwp_to_share`(发布/可视化/安全检查)+ `cps` 也填。warm-start `v0=A.vel, a0=A.accel`。honest `info['trajectory_valid']` 门;plan_minco 抛 'all seeds failed' → 计失败返 False。
- **class 标签源(占位钩子 `_obstacles_from_snapshot`)**:今天 snapshot(obst_pos 中心 + obst_bbox 全尺寸)= 动态追踪集 → 全部 `SphereObstacle(class='human', HARD, radius=0.5*max(extent), vel=0)`;wall/AABB 分支**注释保留**(待静态体素源)。真分类器换这个 body(单一可换钩子)。
- **测试**:新 `test/stage5_planner_minco.py` **28/28**(合成输入离线集成:适配器等价、执飞==发布、避人 dense clearance 1.66≥d_safe(原直线路 -0.30 故判据非空)、**0 次 Gurobi 调用**(spy)、dispatch 双向)。回归 132/132 7 套件 0 回归。我独立复跑 stage5 28/28 + perclass 38/38 + realtime 10/10 + grad 17/17 吻合。
- **挂账(Stage 5 收尾要做)**:① class 源是占位(全当人;墙分支待**静态体素源**——hgp voxel_map 的占据格 → AABB 软墙);② **障碍 vel=0**:snapshot 没带速度 → 集成 planner 暂时没喂动态速度给 plan_minco,S4a 时空在集成层"没料可吃"(realtime 测试直接喂动态人已验算法,缺的是 planner←tracker 的速度接线,即 S4d EKF 适配);③ **删 `solver_gurobi.py` 待塔菲大人确认**(销毁性);④ GUI 真仿真(Gazebo/RViz)本环境跑不了,只验了 planner 逻辑层;⑤ 还没接 `nodes/sando_node.py` 的 ROS 层(订阅/发布)+ EKFState 加 class field。

### Stage 5 RViz demo + 实时提速 ✅ 2026-06-01(workflow w7384qz2j 卡死后我手动打捞完成)
- **RViz demo 建成**:`launch/perclass_demo.launch.py`(一键起 sando_node-MINCO + fake_sim + 移动 per-class 障碍 + 自动 goal + RViz)。新节点 `nodes/perclass_obstacle_pub.py`(人 id=100 横穿=硬 / 墙 id=200 静止=软,发 `predicted_trajs` + 自带 `perclass_obstacles` MarkerArray)、`nodes/auto_goal_pub.py`(延迟发 term_goal)。新 `config/perclass.rviz`。跑法:`source /opt/ros/humble + ws/install`,`ros2 launch sando_py perclass_demo.launch.py`(rviz:=false 无头)。**我 headless 看不到 RViz 画面,只能验管道**(节点起、topic、飞行)。
- **per-class class 源接通(挂账②清)**:DynTraj 无 class 字段、且 `_compute_obst_pos_and_traj_max_time` 原本只留 pos/bbox。改:按 `traj.id`(200≤id<300=wall/SOFT,否则 human/HARD)建并行 `obst_class`+`obst_vel` 快照,`_obstacles_from_snapshot` 据此建 SphereObstacle(硬,带速度)/AABBObstacle(软)。人现在带真速度进 plan_minco(动态反应性在集成层真成立)。
- **headless 跑揪出并修的真 bug**:① `setup.py` 的 `packages` 漏了 `sando_py.local`(minco/local_opt 子包从没装→sando_node import 崩,老 bug);② Marker 的 `pose.position` 必须 `Point` 不是 `Vector3`;③ install 是 5/11 旧拷贝,**改 Python 必须 `colcon build --packages-select sando_py` 再 source**(非 symlink-install)。
- **★ 实时提速(用户报"计算太慢",实测根因)**:单次 replan **1347ms→203ms(~6.6x,~5Hz)**。两根因都不在 MINCO 求解:① **集成路径 plan_local_trajectory_minco 调 plan_minco 时开着默认 detour 多起点**(我只在测试/demo 关过)→ 关掉(全局 A* 已给好种子;multi-start 留离线/Stage 2):236ms→23ms;② **全局 heat-A* 的 `_compose_dynamic_heat` 每拍 Python 三重循环扫每个障碍的可达区域×~10 预测时刻**(大墙盒覆盖几乎全网格)→ 向量化成 numpy 整块(同 int_to_float/同公式,数值等价):空地 88ms / 人+墙 **2055ms→109ms**。全局段测试(stage1_map_heat 16、dynamic_astar 4、stage_stress 8、stage2 11)全绿验等价。**剩 ~88ms 是占据/静态热度重建,可继续向量化压。**
- **坑**:demo 节点退出不干净(zombie,memory [[sando-run-sim-zombie-pattern]]),同名空间多 sando_node 抢 replan 会混淆计时;沙箱拦 pkill/后台`&`(exit 144/1),清理用 dangerouslyDisableSandbox 或 TaskStop;headless 测 `ros2 launch` 必须带 timeout/run_in_background(直接跑会挂——workflow implement agent 就是这么卡死 2 天的)。
- **★ commit-horizon 根因(用户报"轨迹只第一秒被影响、后续不follow"+"狂抖动")**:`find_A_and_Atime` 的拼接点 A 在无人机前方 `commit = k_value_factor × est_comp_time` 处(append_to_plan 保留前 commit 段旧轨迹、其余换新)。`k_value_factor=5`(SANDO 为慢求解器调的):提速前 est_comp_time≈1.3s → commit≈6.5s≈整条 → 只跟第一条(撞 0.19);提速后 est≈0.2s → commit≈1s → A 在 ~2m 前、执飞落后避让仍撞。砍到 `k_value_factor=1.5`(commit≈0.3s)→ 跟得上但**每拍重写近期段→狂抖**。**RT 经典权衡:commit 太长落后、太短抖**。折中 `k_value_factor=3.0`(commit≈0.6s):平滑 + 跟得上(commanded y 单调无抖、drone-hum 跟 minclr)。残留:人摆太快(amp2.5/period6→峰值 vy 2.6m/s)时 0.6s commit 期间人窜 1.6m → 最坏对齐 drone-hum 0.36 还擦;**把人调真实步速(amp2.0/period10→1.26m/s)→ drone-hum 1.52(出身体、擦 d_safe ~0.18)**。根治要 replan 更快(剩 88ms 静态热度向量化→commit 更小不抖)。
- **detour 回落**:RT 路径 `plan_local_trajectory_minco` 先单种子(detour off,~30ms);**若 trajectory_valid=False 再回落 detour 多起点**(慢但稀有,救难场景)——既保 demo 快又过 stage5 回归(纯关 detour 会让某些场景返 False)。
- **commit 调参在 launch**:`k_value_factor`/`default_k_value`;人速在 `perclass_obstacle_pub` 的 `human_amp`/`human_period`。改 Python 必 `colcon build --packages-select sando_py` 再 source。

### Learning 提速定位(2026-05-29 用户拍板纳入路线图,deploy 阶段可选 Stage)
- **做法**:learned **warm-start**——网络输入(粗路+障碍+类别)→ 输出近最优 `q,T` 初值,优化器从好起点出发收敛步数砍 5-10×。**优化器(MINCO+ALM)一字不改、当 certifier 保留**:还输出连续时间证书/λ/STOP,安全完全不依赖网络。零可解释损失。可选再加"学往哪边绕(拓扑/seed 选择)"省多 seed ~5×(略灰但仍可解释)。
- **不做**:端到端神经网络直接吐轨迹(丢 λ/证书/STOP = 丢可解释卖点+人硬约束安全保证,❌不当主线)。
- **时机=核心算法 Stage 3/4 跑通之后 + 和上机 C++/numba 一起**。硬理由:① 训练数据要从优化器自己跑出来(先有正确 teacher 才能 imitate);② formulation 还在变,现在训白训;③ 研究阶段速度已够,ms 级硬需求在 deploy;④ 先做零可解释代价的确定性提速(numba/向量化掉 cost+grad 的 over-M Python loop)。
- **paper framing**:"learned warm-start with retained continuous-time safety certificate"——学习加速 + 保留保证,与 per-class 硬约束贡献**协同非替代**(顶会爱的故事)。

## 进度
- **阶段1(地图+热度)✅ 完成 2026-05-25**。测试:`test/stage1_map_heat.py`(独立脚本,不依赖 ROS/gurobi),**16/16 通过 + 1 NOTE**。
  - 全对:坐标 round-trip、空地图=内部 UNKNOWN + y 边界墙、点云障碍占据+1格膨胀、动态障碍 AABB 占据+dyn_mask、热度(障碍处高/随距离降/远处 0/受 Hmax 限)、点云障碍周边静态热度。
  - NOTE:占据"写"用 floor、"读" float_to_int 用 -0.5,差 ~1 格。已对 C++ 核实:**C++ 也这样(floor 写、-0.5 读),是忠实复刻、非 bug**,被 inflation≥1格 掩盖。
  - 读 C++ 时另发现一个真分歧(测试未直接覆盖):Python 动态 AABB 栅格化用 float_to_int(-0.5),C++ 用 floor → 动态障碍落格差半格,要修改 `voxel_map._rasterize_aabb`。对新方案影响小(地图只喂全局 heat-A* 向导,精度本就粗)。
- **2026-05-25 修复**:把 `voxel_map.float_to_int` 改成 floor(去掉 -0.5)。一处改动让 写/读/动态AABB 三套口径统一为"点属于包含它的格";重跑 16/16,"写=读"一致。上面两条 NOTE/分歧一并解决。**刻意偏离 C++ 的 -0.5 截断——这是新算法,以"干净"为准,不追求复刻。** 阶段1 彻底完成。
- **阶段2(全局 A* 核心 graph_search)✅ 完成 2026-05-25**。测试:`test/stage2_astar.py`(独立,全 3D 场景,手搭地图),**11/11 通过**。
  - 空 3D→直线(cost≈euclid 仅 4% 余量)、墙留高 z 缝(路爬上去穿)、完全堵死→报无路不崩、热度→路绕开热团(权重 5 已够,50 不更近,因团半径固定绕到边即可)、**最优性:6 张随机 3D 图 A* cost == Dijkstra**。
  - **还没测**:`hgp_planner` 的后处理(视线直连捷径 + 路径简化),它把粗路整理成 local_opt 用的拐点。风险低,单列。
- **阶段2b(hgp_planner 后处理)✅ 完成 2026-05-25**。测试:`test/stage2b_hgp_postprocess.py`,**12/12 通过**。
  - **抓到并修复一个真安全 bug**:`angle_spacing_filter` 删点只看几何、不查障碍 → 在"窄缝墙"场景把贴安全侧的 approach 点删了,简化后引导线直奔缝口、从缝下一格穿墙。碰撞检查(LOS/is_blocked)本身没错,是过滤步骤没去问它们。
  - 修法:给 `angle_spacing_filter` + `collapse_short_edges` 加碰撞检查(删点前若会穿障碍就不删);`hgp_planner` 末尾加 `_repair_blocked_segments` 安全复查(有穿墙段就用原始安全细路补回)。改了 utils.py + hgp_planner.py。
  - 验证:窄缝墙 + 3D 障碍块,简化路全程不碰障碍;A* 核心(stage2)无回归 11/11。
- **压力/真实环境测试 ✅ 完成 2026-05-25**(`test/stage_stress.py`,**8/8**)。把抽查升级成随机批量 + 真实地图,又揪出 **3 个更深的 bug 并修复**(抽查全漏了,验证了"简单测试没意义"):
  1. **A* 斜穿墙角(corner-cutting)**:26 连通允许对角步擦过被占的角格 → 引导线擦障碍,后处理修不回(原始路自己就不安全)。修:A* 加 no-corner-cutting(`graph_search._corner_cut`),测试 Dijkstra 同步加同规则。
  2. **碰撞检查点采样会漏格**:`is_blocked`/`line_of_sight_capsule` 用点采样,薄格/掠过漏判 → 简化路仍可能穿障碍。修:换成**精确体素遍历 DDA(Amanatides-Woo)**,不漏线段经过的任何格。
  3. **静态热度在真实(几乎全 UNKNOWN)地图上是死的**:默认不往 UNKNOWN 写,Python 又不扫空地 → 墙的软清晰度等于关掉。修:`static_heat_apply_on_unknown` 默认改 True。
  - 覆盖:随机 200 图后处理安全 0 穿障碍 / 带 UNKNOWN+w_unknown 的 A* vs Dijkstra 最优性 / 真 read_map 地图上热度真把路推开(on<off)/ 动态障碍热度跟着未来走。
  - 改的文件:graph_search.py、voxel_map.py、utils.py、hgp_planner.py。5 个测试脚本都在 `test/`。
- **动态障碍 + A* 绕行时机 ✅ 验证 2026-05-25**(`test/dynamic_astar.py`,4/4):
  - 有预测 → A* 绕开障碍**未来要扫过的整条走廊**(不只躲当前点;偏移 0.70 vs 无预测 0.10)。A* 会选 z 方向往上绕。
  - **时机生效**(低 heat_weight 区间):快要穿过的强制大绕(0.70),很久才穿过的几乎不绕(0.10);高 heat_weight 时两者饱和。
  - **重要认知**:A* **没有时间轴**,绕的是"按多久到加权的静态热度管";真正的空间+时间避让(在正确时刻躲)是 **local_opt 的活**,不是全局向导的。
- **全局段(1+2+2b+stress+dynamic)= 真·扎实。**
- **阶段3 起步:`local/bspline.py` ✅ 完成 2026-05-28**。3D uniform B-spline(默认 quintic),手写 De Boor + nonzero_basis,LSQ 从折线拟控制点 + clamp_endpoints。
  - **5 个测试文件 84 项全绿**(沿全局段测试风格:functional / invariants / boundary / real / perf):
    1. `stage3_bspline.py` 33 项:De Boor vs scipy(err ~1e-15)、批量/边界、partition of unity、导数 vs 中央差分、lock_endpoint、LSQ 直线/圆弧/折线、LSQ+clamp、30 配置压力扫。
    2. `stage3_bspline_invariants.py` 9 项:平移不变、ctrl 线性、凸包、C^{p-1} 连续性、scipy.BSpline.derivative oracle(orders 1..p)、basis 非负+和为 1、200 随机 LSQ+clamp、refit 自一致(spline→采样→LSQ 还原)。
    3. `stage3_bspline_boundary.py` 28 项:最小 num_ctrl=p+1、太少报错、degree=1 线性插值、degree=7、极端 dt、常数/共线/M=2/under-determined path、lock_endpoint 错 side、d=2、ndim 校验、构造参数校验(dt≤0/degree<1/ndim=1)、eval(±inf) clip、clamp 回归。
    4. `stage3_bspline_real.py` 9 项:接全局段 HGPPlanner,2 个手搭 3D 场景 + 24 个随机 3D 地图(seed=2026),验端点锁、chord 距离 < 容忍、4 阶导有限。
    5. `stage3_bspline_perf.py` 5 项:N=500 spline、M=2000 LSQ、p=7、所有阶导、10k 采样,30s 阈。
  - **review 揪到 1 个真 bug + 修**:`fit_path(clamp_endpoints=True)` 当 `num_ctrl < 2*(p+1)` 时,首 p+1 锁和末 p+1 锁会重叠,end 锁覆盖 start 锁导致起点不被锁。修:加 check,要求 `num_ctrl ≥ 2*(degree+1)`。钉 B.14 回归。
  - **设计微调**:`__init__` 加 dt>0 / degree≥1 数学合法性检查;`lock_endpoint` 删掉 level 参数(level=0 时位置实际不锁,陷阱)。
  - **认知点**:LSQ 用均匀 t 参数化,不按弧长。当 A* 路段长不均匀时,"按索引对齐"误差大但 chord 距离仍小(spline 形状没问题,只是 t 走得快慢和原 path 不同步)。real 测试用 chord 距离作判据,这才是几何意义上的"接近 polyline"。
  - 全局段 11 个测试 91 项,跑完无回归。
- **阶段3 第二块:`local/avoid_config.py` + `obstacles.py` + `cost.py` ✅ 完成 2026-05-28**。
  - **设计**:open schema(`AvoidParams`:name/mode/d_safe/weight),wall=AABB+soft,human=球+hard,EGO 三次罚分 `(d_safe-d)^3 if d<d_safe`,动态障碍带 `.predict(t)` 接口(`SphereObstacle.centre0 + vel*t`)。
  - **5 个测试文件 52 项全绿**:
    1. `stage3_cost.py` 17 项:空 list、远离=0、公式对照手算、AABB、多障碍可加、hard/soft 比 = weight 比 1000x、动态 vs 静态 sphere、C^1 边界平滑、类独立、K 收敛。
    2. `stage3_cost_invariants.py` 9 项:平移不变、weight 线性、d_safe 单调、类加性、非负(200 批量)、远处=0(50 批量)、1st-order Taylor 一致性(梯度 oracle)、K=1000 vs K=4000 < 5%。
    3. `stage3_cost_boundary.py` 14 项:K=1/10000、d_safe=0(只罚穿透)、weight=0、负 d_safe=0、radius=0 点障碍、AABB lo==hi、AABB hi<lo 报错、巨型障碍 finite、未知类报 KeyError、混类、最小 spline、K override。
    4. `stage3_cost_real.py` 8 项:HGPPlanner 真实 A* + 障碍,空=0、近障碍>0、远=0、滑近单调增、hard/soft 100x、30 随机 3D 场景。
    5. `stage3_cost_perf.py` 4 项:K=500 31ms、K=2000+50 障碍 527ms、200 次调用 4.9s、finite-diff 梯度 90 维 1.17s。
  - **review 时改进**:初版 cost 是 `Σ` 不是 mean,K 跟 weight 强耦合(K↑4 倍 cost↑4 倍)— I.7 K-收敛测试抓到。改成 `Σ/K`,K 解耦,优化器友好。
  - **性能尺度**:LBFGS 内部 finite-diff 梯度(90 维 ctrl)约 1s/iter,100 iter ~2 分钟。够研究/调试用,上机要手算梯度或 jit。
  - 全套阶段3(10 文件 136 项)+ 全局段(11 文件 91 项)= 21 文件 227 项全绿,无回归。
- **阶段3 第三块:`local/local_opt.py` ✅ 完成 2026-05-28**。scipy L-BFGS-B 主流程。
  - **cost**:`obstacle_cost`(EGO 三次)+ `w_smooth·mean(||snap||²)` + `w_vel·mean(max(0,|v|-vmax))²` + `w_accel·mean(max(0,|a|-amax))²` + `w_time·((T-T_target)/T_target)²`。
  - **优化变量**:中间 ctrl(`n_ctrl - 2(p+1)`)+ dt 单标量。首末 p+1 个 ctrl 锁端点。dt 加 lower bound = `dt_min`。
  - **时间锚**(关键):不加 anchor 时 LBFGS 会把 dt 拉爆(smooth/vel/accel cost ~1/dt²,作弊式拉时间换形状),T 从 3.3s 涨到 71s。加 `w_time·((T-T_target)/T_target)²` 拉 T 到 `path_len/vmax` 修了。
  - **接口**:`plan(astar_path, obstacles, avoid_cfg, *, num_ctrl=None, opt_params=None) → (spline, info)`。info 含 converged/init_cost/final_cost/iter/dt0/dt_final/T_target/T_final/num_ctrl/message。
  - **不收敛策略**:正常 return + `info.converged=False`,上层看 info 决定下一步。
  - **5 个测试文件 52 项全绿**(maxiter=30-80, K=80-200 lean 预算):functional 17、invariants 5、boundary 15、real 8、perf 7。
  - **已知限制**:LBFGS + 非凸 obstacle penalty 卡局部最优(简单场景 16 iter 后 `converged=True` 但 obstacle 没完全绕开,加 maxiter 没用因 gtol 已触发)。第一版预期,下一版需 multi-start / 更好初值 / Adam。
  - **性能**:dim=10 finite-diff LBFGS,lean 5s/default 15s/iter ~30。研究够,上机需手算梯度或 jit。
- **下一步**:接 ROS nodes / 接 controller / obstacle_tracker 加 class,或先做 local_opt 二代(multi-start / Adam)解 LBFGS 局部最优。
- **全套阶段3(15 文件 188 项)+ 全局段(11 文件 91 项)= 26 文件 279 项全绿**。
