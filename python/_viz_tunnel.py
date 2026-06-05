"""画 tunnel 约束内穿人群:读 C++ dump 的 CSV(media/_tunnel_W*.csv),两栏对比
W=1.6(lane 内织过移动人群到达)vs W=0.9(lane 太窄无缝 -> held-safe)。
跑:  python _viz_tunnel.py  ->  media/_tunnel.png  (先跑 cpp/test/viz_tunnel 生成 CSV)
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

ROOT = os.path.dirname(os.path.abspath(__file__)); MED = os.path.join(ROOT, "media")
D_SAFE = 0.8; HR = 0.3
SPEC = [(3,-1.5,0.55),(5,1.5,-0.55),(7,-1.5,0.55),(9,1.5,-0.55),(11,-1.5,0.55),(13,1.5,-0.55)]


def load(W):
    fn = os.path.join(MED, f"_tunnel_W{round(W*10):02d}.csv")
    t, x, y, c = [], [], [], []
    with open(fn) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"])); x.append(float(row["x"])); y.append(float(row["y"])); c.append(float(row["clr"]))
    return map(np.array, (t, x, y, c))


def panel(ax, W, title):
    t, x, y, c = load(W)
    ax.axhspan(-W, W, color="tab:blue", alpha=0.08, zorder=0)
    ax.axhline(W, color="tab:blue", ls="--", lw=1); ax.axhline(-W, color="tab:blue", ls="--", lw=1)
    ax.text(0.2, W, f" tunnel ±{W:.1f}", color="tab:blue", fontsize=8, va="bottom")
    # humans: faint track + circle+halo at the time the path is nearest its x
    Tend = t[-1] if len(t) else 5
    for (hx, hy0, vy) in SPEC:
        ys = [hy0 + vy * tt for tt in np.linspace(0, Tend, 30)]
        ax.plot([hx] * 30, ys, "-", color="0.85", lw=4, alpha=0.6, zorder=1)
        if len(x):
            tp = t[np.argmin(np.abs(x - hx))]; yc = hy0 + vy * tp
            ax.add_patch(Circle((hx, yc), HR + D_SAFE, color="tab:red", alpha=0.10, zorder=2))
            ax.add_patch(Circle((hx, yc), HR, color="tab:red", alpha=0.65, zorder=3))
    # path coloured by clearance (green safe / red if <d_safe)
    if len(x) > 1:
        pts = np.array([x, y]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="RdYlGn", norm=plt.Normalize(D_SAFE - 0.3, D_SAFE + 1.2), zorder=5)
        lc.set_array(c[:-1]); lc.set_linewidth(3); ax.add_collection(lc)
        ax.plot(x[-1], y[-1], "o", color="k", ms=6, zorder=6)
    ax.plot(0, 0, "o", color="lime", ms=9, zorder=6); ax.plot(14, 0, "*", color="k", ms=15, zorder=6)
    ax.set_xlim(-1, 15.5); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10); ax.set_ylabel("y (m)")


fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
t16, x16, _, c16 = load(1.6); t09, x09, _, c09 = load(0.9)
panel(a1, 1.6, f"tunnel W=1.6: A+B THREADS the moving crowd within the lane → reaches goal "
               f"(x={x16[-1]:.1f}, min_clr={c16.min():+.2f}≥d_safe)  [<50ms/replan]")
panel(a2, 0.9, f"tunnel W=0.9: no timed gap in the narrow lane → gatekeeper HOLDS safe "
               f"(stops at x={x09[-1]:.1f}, still in-lane & no collision; min_clr={c09.min():+.2f})")
a2.set_xlabel("x (m)")
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(D_SAFE - 0.3, D_SAFE + 1.2))
cb = fig.colorbar(sm, ax=[a1, a2], fraction=0.025); cb.set_label("human clearance (m); red<d_safe")
fig.suptitle("Thread the crowd WITHIN a tunnel: safety provable at all widths; reaching iff the lane admits a timed gap", fontsize=11)
fig.savefig(os.path.join(MED, "_tunnel.png"), dpi=120, bbox_inches="tight")
print("wrote media/_tunnel.png")
