"""画 gif:高复杂度(12 个密集移动硬行人,SANDO dynamic-hard 强度)+ tunnel,无人机在 lane 内织过去。
读 C++ dump 的 media/_gif_path.csv / _gif_hum.csv / _gif_meta.csv。
跑:  python _viz_tunnel_gif.py  ->  media/_tunnel_crowd.gif
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as anim

ROOT = os.path.dirname(os.path.abspath(__file__)); MED = os.path.join(ROOT, "media")
meta = list(csv.DictReader(open(os.path.join(MED, "_gif_meta.csv"))))[0]
W = float(meta["W"]); D_SAFE = float(meta["d_safe"]); GOALX = float(meta["goalx"])

path = list(csv.DictReader(open(os.path.join(MED, "_gif_path.csv"))))
T = np.array([float(r["t"]) for r in path]); X = np.array([float(r["x"]) for r in path])
Y = np.array([float(r["y"]) for r in path]); C = np.array([float(r["clr"]) for r in path])
MS = np.array([float(r["ms"]) for r in path])

hum_by_t = {}
for r in csv.DictReader(open(os.path.join(MED, "_gif_hum.csv"))):
    hum_by_t.setdefault(round(float(r["t"]), 3), []).append((float(r["x"]), float(r["y"]), float(r["r"])))

frames = list(range(0, len(T), 2))   # subsample for gif size

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(-1, GOALX + 1.5); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal")
ax.axhspan(-W, W, color="tab:blue", alpha=0.08)
ax.axhline(W, color="tab:blue", ls="--", lw=1); ax.axhline(-W, color="tab:blue", ls="--", lw=1)
ax.text(0.0, W + 0.05, f"tunnel ±{W:.1f}", color="tab:blue", fontsize=8, va="bottom")
ax.plot(0, 0, "o", color="lime", ms=9); ax.plot(GOALX, 0, "*", color="k", ms=15)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
halos = [Circle((0, 0), 0.1, color="tab:red", alpha=0.10) for _ in range(20)]
bodies = [Circle((0, 0), 0.1, color="tab:red", alpha=0.7) for _ in range(20)]
for h in halos + bodies: ax.add_patch(h); h.set_visible(False)
trail, = ax.plot([], [], "-", color="tab:green", lw=2.2)
drone, = ax.plot([], [], "o", color="yellow", mec="k", ms=10, zorder=8)
txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle="round", fc="white", alpha=0.7))


def upd(fi):
    i = frames[fi]; tt = round(T[i], 3)
    hs = hum_by_t.get(tt, [])
    for j, h in enumerate(halos):
        if j < len(hs):
            hx, hy, r = hs[j]
            halos[j].center = (hx, hy); halos[j].set_radius(r + D_SAFE); halos[j].set_visible(True)
            bodies[j].center = (hx, hy); bodies[j].set_radius(r); bodies[j].set_visible(True)
        else:
            halos[j].set_visible(False); bodies[j].set_visible(False)
    trail.set_data(X[:i + 1], Y[:i + 1]); drone.set_data([X[i]], [Y[i]])
    safe = C[i] >= D_SAFE - 1e-3
    txt.set_text(f"t={T[i]:4.1f}s  x={X[i]:4.1f}  min human clr={C[i]:+.2f} "
                 f"({'SAFE' if safe else 'tight'})  replan {MS[i]:.0f}ms (<50)")
    drone.set_color("yellow" if safe else "orange")
    return halos + bodies + [trail, drone, txt]


ani = anim.FuncAnimation(fig, upd, frames=len(frames), interval=60, blit=False)
fig.suptitle(f"A+B threads a DENSE moving crowd (12 dynamic-hard humans) WITHIN a tunnel (±{W:.1f}), <50ms/replan, human-safe", fontsize=10)
fig.tight_layout()
ani.save(os.path.join(MED, "_tunnel_crowd.gif"), writer=anim.PillowWriter(fps=16))
print("wrote media/_tunnel_crowd.gif")
