"""Stage 3 interpretability demo: per-class HARD(human)/SOFT(wall) avoidance.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig  # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle    # noqa: E402
from sando_py.local.avoid_config import AvoidParams                  # noqa: E402

# ---------------- scene (z fixed at 1.5 -> top-down XY view) ----------------
start = np.array([0.0, 0.0, 1.5])
goal = np.array([8.0, 0.0, 1.5])
astar = np.linspace(start, goal, 9)                    # straight guide through the gap
# channel: the human below forces the path up to its 1.2 m berth (hard, BINDING);
# a soft wall caps the top and is grazed -> the path is pinned in a narrow lane.
human = SphereObstacle(centre0=[4.0, -0.80, 1.5], radius=0.4, class_name="human")
wall = AABBObstacle(lo=[3.0, 0.60, 0.0], hi=[5.0, 3.0, 3.0], class_name="wall")
cfg = {
    "human": AvoidParams("human", "hard", d_safe=0.8, weight=1.0e4),
    "wall": AvoidParams("wall", "soft", d_safe=0.4, weight=1.0e1),
}
obstacles = [human, wall]
HUMAN_DSAFE = cfg["human"].d_safe


def run(override):
    sp, info = plan_minco(astar, obstacles, cfg, opt_params=OptParams(avoid_override=override),
                          detour_cfg=DetourConfig(enabled=False))
    return sp, info


def human_clearance(sp, n=600):
    ts = np.linspace(sp.t_start, sp.t_end, n)
    pts = sp.eval(ts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    c = np.array([human.predict(float(t)) for t in ts])
    d = np.linalg.norm(pts - c, axis=1) - human.radius
    return s, d, pts


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
certs = [c for c in info["hard_certificates"] if c["active"]]
lam_max = max((c["lambda"] for c in certs), default=1.0) or 1.0
for c in certs:
    Pk = c["P"]
    f = c["force"]
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        continue
    u = f / nf
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
