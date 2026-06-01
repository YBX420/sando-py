"""Stage 3 interpretability demo: per-class HARD(human)/SOFT(wall) avoidance.

中文说明(这个文件是干嘛的):
  「可解释性」演示脚本(画静态 PNG,无需 GUI)。它在一个固定场景里,把规划器「为什么这么躲」
  画清楚:MINCO 轨迹 vs A* 全局向导、人(HARD 硬约束)和它的安全圈 d_safe、墙(SOFT 软约束、
  允许蹭)、轨迹到人的「连续时间净空」曲线,以及一个对照实验(把所有障碍都当软 / per-class /
  全当硬)看最小净空。卖点就是「每一步为什么都看得见」。
  它不是 ROS 节点,直接调底层的 plan_minco() 规划一次再用 matplotlib 出图。
  关键术语:HARD=必须满足、绝不能进安全圈;SOFT=尽量躲、可以轻微蹭;
  ALM 的「force / shadow price(影子价格)」=该约束有多「顶手」,箭头越长说明这个点被推得越凶。

Renders, for one scene, why the planner avoids the way it does:
  - the MINCO trajectory vs the A* guide,
  - the human (hard) with its d_safe ring + the ALM "force" arrows (shadow price),
  - the wall (soft) it is allowed to graze,
  - the continuous-time clearance vs d_safe, contrasted soft-only / per-class / all-hard,
  - the ablation min-clearance bars (the paper's soft-breaches-into-the-person figure).
"""
import os
import sys

import matplotlib
# 中文:Agg 是非交互后端,只画到文件不弹窗——这样在没有显示器的机器(headless)上也能出图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

# 中文:把本文件所在目录加进搜索路径,这样不装包也能直接 import 到 sando_py 的本地模块。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig  # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle    # noqa: E402
from sando_py.local.avoid_config import AvoidParams                  # noqa: E402

# ---------------- scene (z fixed at 1.5 -> top-down XY view) ----------------
# 中文:整个场景 z 固定 1.5,等于俯视 XY 平面看。下面构造的是一个「窄通道」:
#       人在下面把路顶上去(硬约束、起作用),墙在上面封顶(软约束、被蹭),
#       于是轨迹被夹在一条细缝里——这样最能看出 per-class 的区别。
start = np.array([0.0, 0.0, 1.5])
goal = np.array([8.0, 0.0, 1.5])
astar = np.linspace(start, goal, 9)                    # straight guide through the gap
# channel: the human below forces the path up to its 1.2 m berth (hard, BINDING);
# a soft wall caps the top and is grazed -> the path is pinned in a narrow lane.
human = SphereObstacle(centre0=[4.0, -0.80, 1.5], radius=0.4, class_name="human")
wall = AABBObstacle(lo=[3.0, 0.60, 0.0], hi=[5.0, 3.0, 3.0], class_name="wall")
# 中文:per-class 配置表,按障碍类别名给一套避障参数:
#       human=hard 用大权重(必须躲开,d_safe=0.8 米安全圈);wall=soft 用小权重(轻轻躲一下,可蹭)。
cfg = {
    "human": AvoidParams("human", "hard", d_safe=0.8, weight=1.0e4),
    "wall": AvoidParams("wall", "soft", d_safe=0.4, weight=1.0e1),
}
obstacles = [human, wall]
HUMAN_DSAFE = cfg["human"].d_safe


# 中文:跑一次规划。override=None 表示按 per-class 各自的硬/软来;传 "soft"/"hard" 则把
#       所有障碍统一强制成软/硬,用来做对照实验。detour 关掉,只看纯优化的避让效果。
def run(override):
    sp, info = plan_minco(astar, obstacles, cfg, opt_params=OptParams(avoid_override=override),
                          detour_cfg=DetourConfig(enabled=False))
    return sp, info


# 中文:沿轨迹密集采样,算每个时刻无人机到「人」的净空(到人中心距离减去人半径)。
#       返回 (沿轨迹累计弧长 s, 净空 d, 采样点 pts)。注意用 human.predict(t) 取人在该时刻的位置,
#       所以即使人会动,这里比的也是「同一时刻」的距离。
def human_clearance(sp, n=600):
    ts = np.linspace(sp.t_start, sp.t_end, n)
    pts = sp.eval(ts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    c = np.array([human.predict(float(t)) for t in ts])
    d = np.linalg.norm(pts - c, axis=1) - human.radius
    return s, d, pts


# 中文:三个对照组——per-class(各按本类硬/软)、all-soft(全软)、all-hard(全硬)。
#       一次性都跑出来,后面三张子图都基于这个 res。
arms = {"per-class": None, "all-soft": "soft", "all-hard": "hard"}
res = {name: run(ov) for name, ov in arms.items()}

fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0])
axT = fig.add_subplot(gs[:, 0])     # top-down trajectory (per-class)
axC = fig.add_subplot(gs[0, 1])     # clearance vs arc-length
axB = fig.add_subplot(gs[1, 1])     # ablation bars

# ---- panel 1: top-down per-class scene ----
sp, info = res["per-class"]
axT.plot(astar[:, 0], astar[:, 1], "--", color="0.6", lw=1.5, label="A* guide")
ts = np.linspace(sp.t_start, sp.t_end, 400)
P = sp.eval(ts)
axT.plot(P[:, 0], P[:, 1], "-", color="C0", lw=2.5, label="MINCO trajectory")
# human: body + d_safe ring
axT.add_patch(Circle(human.centre0[:2], human.radius, color="C3", alpha=0.55, zorder=3))
axT.add_patch(Circle(human.centre0[:2], human.radius + HUMAN_DSAFE, fill=False,
                     ec="C3", ls="--", lw=1.5, zorder=3))
axT.text(human.centre0[0], human.centre0[1], "human\n(HARD)", ha="center", va="center",
         fontsize=8, color="white", zorder=4)
# wall (soft)
axT.add_patch(Rectangle(wall.lo[:2], *(wall.hi[:2] - wall.lo[:2]), color="0.4", alpha=0.6))
axT.text(0.5 * (wall.lo[0] + wall.hi[0]), 0.5 * (wall.lo[1] + wall.hi[1]), "wall\n(soft)",
         ha="center", va="center", fontsize=8, color="white")
# ALM force arrows (shadow price) at active hard control points
# 中文:画 ALM「受力箭头」。只画那些真正起作用(active)的硬约束控制点;lambda 是该约束的
#       拉格朗日乘子(影子价格),越大说明这个点被顶得越狠。下面用 lambda 归一化来定箭头长短。
certs = [c for c in info["hard_certificates"] if c["active"]]
lam_max = max((c["lambda"] for c in certs), default=1.0) or 1.0
for c in certs:
    Pk = c["P"]
    f = c["force"]
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        continue
    u = f / nf
    # 中文:箭头长度按 lambda 缩放——越「顶手」的约束箭头越长,直观看出哪个点最吃紧。
    L = 0.25 + 0.9 * (c["lambda"] / lam_max)        # arrow length encodes lambda
    axT.annotate("", xy=(Pk[0] + u[0] * L, Pk[1] + u[1] * L), xytext=(Pk[0], Pk[1]),
                 arrowprops=dict(arrowstyle="->", color="C1", lw=1.8), zorder=5)
    axT.plot(Pk[0], Pk[1], "o", color="C1", ms=3, zorder=5)
axT.plot([], [], "o-", color="C1", label=r"hard force $\lambda\cdot\hat n$ (shadow price)")
axT.set_aspect("equal")
axT.set_xlabel("x [m]"); axT.set_ylabel("y [m]")
axT.set_title(f"per-class: human=HARD, wall=soft\nvalid={info['trajectory_valid']}, "
              f"min human clearance={info['continuous_min_clearance']:.3f} m "
              f"(d_safe={HUMAN_DSAFE})")
axT.legend(loc="upper left", fontsize=8); axT.grid(alpha=0.3)

# ---- panel 2: clearance vs arc-length ----
# 中文:第二张图,三种策略各画一条「净空随弧长变化」的曲线,并画出 d_safe 水平线。
#       低于这条线就是侵入了人的安全圈(危险)。
colors = {"per-class": "C0", "all-soft": "C3", "all-hard": "C2"}
for name in arms:
    sp_a, info_a = res[name]
    s, d, _ = human_clearance(sp_a)
    axC.plot(s, d, color=colors[name], lw=2,
             label=f"{name} (min={d.min():.3f})")
axC.axhline(HUMAN_DSAFE, color="k", ls="--", lw=1, label=f"d_safe={HUMAN_DSAFE}")
axC.set_xlabel("arc length [m]"); axC.set_ylabel("clearance to human [m]")
axC.set_title("continuous-time clearance (densely sampled)")
axC.legend(fontsize=8); axC.grid(alpha=0.3)

# ---- panel 3: ablation min-clearance bars ----
# 中文:第三张图,对照实验的柱状图——每种策略的「最小净空」一根柱子,并标 VALID/BREACH。
#       要点:三组只差「类别->硬/软 这个开关」,其余完全一样,所以差异纯粹来自 per-class 机制。
#       预期 all-soft 会侵入人(BREACH),per-class 和 all-hard 不会。
names = list(arms)
mins = [human_clearance(res[n][0])[1].min() for n in names]
valids = [res[n][1]["trajectory_valid"] for n in names]
bars = axB.bar(names, mins, color=[colors[n] for n in names])
axB.axhline(HUMAN_DSAFE, color="k", ls="--", lw=1)
axB.text(2.5, HUMAN_DSAFE + 0.02, f"d_safe={HUMAN_DSAFE}", ha="right", fontsize=8)
for b, m, v in zip(bars, mins, valids):
    axB.text(b.get_x() + b.get_width() / 2, m + 0.03,
             f"{m:.3f}\n{'VALID' if v else 'BREACH'}", ha="center", fontsize=8)
axB.set_ylabel("min human clearance [m]")
axB.set_title("ablation: only the class->mechanism switch changes")
axB.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_stage3_perclass.png")
fig.savefig(out, dpi=130)
print("saved", out)
for name in arms:
    info_a = res[name][1]
    print(f"{name:10s} valid={info_a['trajectory_valid']!s:5s} "
          f"min_clr={info_a['continuous_min_clearance']:.4f} "
          f"n_hard={info_a['n_hard']} n_soft={info_a['n_soft']} "
          f"cert_margin={info_a['certificate_margin']:.4f}")
