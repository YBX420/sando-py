# CLAUDE.md — sando-py(给 Claude Code 自动读的项目入口)

Claude:在本仓库工作前,**先读 `.claude/` 里打包的完整上下文**(换电脑/换会话时这就是你的全部背景):
- `.claude/CLAUDE.user.md` —— 用户(塔菲大人)的个人偏好:**称呼他「塔菲大人」**、默认中文、简洁、少术语、改 bug 别顺手重构、销毁性操作先确认、**当前研究方向**。
- `.claude/CLAUDE.workspace.md` —— `~/code` 三个 colcon 工作区的技术细节、构建/跑仿真/benchmark 命令、代码架构。
- `.claude/memory/sando-rgbd-plan.md` —— **当前算法方向、决策、分阶段进度的权威记录(以它为准,先读这个)**;同目录还有 feedback-*(工作风格)、sando-rgbd-*(测试清单/已知坑/规格)、MEMORY.md(索引)。

## 这个仓库是什么
`sando_py`:在 **SANDO 基线**(MIT-ACL,ROS 2 Humble,C++)之上开发的**新无人机轨迹规划器(博士主线,顶会目标)**。
核心创新 = **per-class 差异化避障**:人 = 硬约束(凸包 + ALM 连续时间证书 + Stage-4 时空避让),墙 = 软场(EGO),由障碍类别决定用哪套机制;骨架 = 全局 heat-A* 向导 → 局部 **MINCO**(min-jerk 五次,banded `M(T)c=b`)+ 解析梯度优化。**不是 patch SANDO,是基于它做新框架。**

四个优化指标:① 动态反应性 ② 可解释性(当 paper 卖点)③ 安全裕度/成功率 ④ 轨迹质量。

## 进度速览(细节见 `.claude/memory/sando-rgbd-plan.md`)
- 地基 M0/M1/M2 ✅、Stage 3 per-class ✅、Stage 4 时空 ALM ✅、Stage 5 MINCO 接进 planner ✅、RViz demo ✅、提速(向量化/detour-off/commit 调参)✅。
- 当前痛点:RViz demo 效果待调(commit 拼接滞后、感知 class 占位、全局向导粗)。

## 跑 demo / 测试
```bash
cd ~/code/sando_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch sando_py perclass_demo.launch.py          # RViz 实时演示
colcon build --packages-select sando_py && source install/setup.bash   # 改 Python 后必须重建
python3 src/sando_py/test/stage3_minco_perclass.py    # 测试是独立脚本,直接 python3 跑
```
进程清理(跑仿真前后):`pkill -9 -f 'sando_py/lib/sando_py'; pkill -9 -f 'ros2 launch sando_py'`

> 注:`.claude/CLAUDE.user.md` / `CLAUDE.workspace.md` / `memory/` 是从 `~/.claude` 和 `~/code/CLAUDE.md` 拷来的快照(方便跨电脑)。要让 Claude 的「自动记忆」也生效,见 `.claude/README.md` 的还原说明。
