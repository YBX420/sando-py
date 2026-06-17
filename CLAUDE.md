# CLAUDE.md — sando-py(给 Claude Code 自动读的项目入口)

> ⚠️ **2026-06-11 已 pivot(方案 B)。** 主论文 = **planner 无关的认证语义风险安全层**(投 RA-L,~9/15);原 per-class MINCO 规划器降级为 side paper(9/15 之后写,10-11 月)。下面正文已按 pivot 重写;旧的 planner-主线叙事见 `docs/research-direction.md`(已标注过时)。

Claude:在本仓库工作前,**先读权威蓝图,再读上下文快照**:
- **权威方向(以这两份为准,先读)**:`docs/safety-layer-spec.md`(修正版 spec)+ `docs/safety-layer-plan.md`(13 周施工计划);原始档 `docs/safety-layer-dossier.json`、环境/移植 `docs/UBUNTU22_PORT.md`。
- `.claude/CLAUDE.user.md` —— 用户(塔菲大人)个人偏好:**称呼他「塔菲大人」**、默认中文、简洁、少术语、改 bug 别顺手重构、销毁性操作先确认。
- `.claude/CLAUDE.workspace.md` —— `~/code` colcon 工作区技术细节、构建/跑仿真命令、代码架构(注:`~/code/sando_ws` 只在带 display 的仿真机上存在,这台移植笔记本没有)。
- `.claude/memory/` —— **多为 pivot 前快照(framing 已过时)**:feedback-*(工作风格,仍有效)、sando-rgbd-*(C++ 测试清单/坑,仍有效)、sando-py-core-idea/conformal-cert/defense-map/sim2real-fakes(pivot 前的 planner-内置证书前身,顶部已加横幅)、MEMORY.md(索引)。conformal 证书数学被 pivot 继承,但「planner 内置 / per-class 为主轴」的定位已过时。

## 这个仓库是什么
`sando_py`:在 **SANDO 基线**(MIT-ACL,ROS 2 Humble,C++)之上开发的、承载**两条论文线**的代码仓库。
- **主线(博士主论文,投 RA-L ~9/15)= planner 无关的认证语义风险安全层**:无人机在行人附近飞行时,对**每个被检测到的行人轨迹、每个规划回合**,保证 `P(撞该人) ≤ ε_cls + ε_pred`(ε=0.1,拆 0.02+0.08)。三个感知分支:① 检测+分类的 agent → conformal 分类集合(含 human→硬约束)+ per-agent conformal tube;② 未观测/遮挡空间 → 确定性遮挡阴影 `r_occ + v_max·t`;③ 看见但没识别成 agent → depth→occupancy body-clearance 门 + ~0.4m 延迟膨胀。保证在类别×密度 Mondrian 分层(3 类×3 密度=9 格)内成立;计分用整段 sup/max-over-horizon 时间整形分数(避免 per-step union)。证书在 **tracker 输出**上、用 **Isaac 机载渲染**标定(绝不用 SDD 标定证书);**学习型预测器是成败手**(W4 gate:tracker 输出 3s q95 ≤ 0.6-0.7m)。RTA = 监视器独立节点(不站策略推理路径)+ 最小修正 QP + 垂直爬升 backup。
- **side paper(9/15 之后)= per-class 差异化 MINCO 规划器**:人 = 硬约束(凸包 + ALM + 连续时间证书 + Stage-4 时空避让),墙 = 软场(EGO),由障碍类别决定用哪套机制;骨架 = 全局 heat-A* 向导 → 局部 **MINCO**(min-jerk 五次,banded `M(T)c=b`)+ 解析梯度优化。spec §5 判定 **planner 基本可冻结**,它在主论文里当载体、不前置。**原「双 planner 即插即用」已砍。**

## 进度速览(细节见 `docs/safety-layer-plan.md` §1 时间线)
- **W1 闸门 = Boyle 签字**(弱化定理一页纸;不同意则回退 C = planner 论文为主),并行:笔记本移植验收(ctest 19/19 + 闭环 PNG)、Isaac 版本钉死、IRA 5/15/40 三档密度跑通、SDD 预处理启动、DynTraj label-set ABI 设计稿、飞行笼审批提交。
- **关键 gate**:W4 末预测器 q95 达标、W7 中总 go/no-go。三条铁律见 plan §0(9/15 前不碰 side paper / FN 三分支+预测器+tracker 标定永不砍 / 摘要防雷句式)。
- 规划器侧(side paper 载体)地基已成:M0/M1/M2、Stage 3 per-class、Stage 4 时空 ALM、Stage 5 MINCO 接进 planner、**ctest 19/19 全绿**;基本冻结,只在安全层需要时动(如 DynTraj label 字段、committed-traj getter)。

## 跑 demo / 测试

**本机(移植笔记本)算法 + C++ 测试 = conda env `sando`**(权威指南 `docs/UBUNTU22_PORT.md`):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sando
cd cpp && cmake --build build -j && (cd build && ctest)        # C++ golden:19/19 全过
# ★ capi 共享库不在 CMake 构建图里,改 C++ 后要手动重编(否则 python 桥 / Isaac 闭环全断):
g++ -O2 -shared -fPIC -std=c++17 -o capi/sando_capi.so capi/sando_capi.cpp -Iinclude -Ithird_party/eigen -Ithird_party
PYTHONPATH=python python python/test/<name>.py                 # python 测试是独立脚本,直接跑
```

**ROS demo(需带 display 的仿真机 + `~/code/sando_ws`,本机没有)**:
```bash
cd ~/code/sando_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch sando_py perclass_demo.launch.py          # RViz 实时演示
colcon build --packages-select sando_py && source install/setup.bash   # 改 Python 后必须重建
python3 src/sando_py/test/stage3_minco_perclass.py    # 测试是独立脚本,直接 python3 跑
```
进程清理(跑仿真前后):`pkill -9 -f 'sando_py/lib/sando_py'; pkill -9 -f 'ros2 launch sando_py'`

> 注:`.claude/CLAUDE.user.md` / `CLAUDE.workspace.md` / `memory/` 是从 `~/.claude` 和 `~/code/CLAUDE.md` 拷来的快照(方便跨电脑)。要让 Claude 的「自动记忆」也生效,见 `.claude/README.md` 的还原说明。

## Git 提交署名(强制)
- 禁止在 commit message 中添加 `Co-Authored-By:` 行 —— 不要把 Claude(或任何 AI)添加为 co-author。
- 禁止在 commit / PR 描述中添加 "Generated with Claude Code"、"🤖 ..." 等 AI 署名。
- 不要把 Claude 列入 contributor。
