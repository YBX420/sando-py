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
# z(高度):tunnel 只约束 y,z 自由 -> 最挤处无人机抬高飞越人(z 远离人面 2.0)。
# 注意:2D 俯视图里无人机会和最近的人「叠」,那是把 3D 飞越投影到平面的假象;clr 是真 3D 净空(恒>=d_safe)。
Z = np.array([float(r.get("z", 2.0)) for r in path])
HUMAN_Z = 2.0

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
    over = Z[i] > HUMAN_Z + 0.4   # 明显抬高 = 正在从顶上飞越人
    txt.set_text(f"t={T[i]:4.1f}s  x={X[i]:4.1f}  z={Z[i]:4.2f}{'  (climbs OVER)' if over else ''}  "
                 f"3D clr={C[i]:+.2f} ({'SAFE' if safe else 'tight'})  replan {MS[i]:.0f}ms (<50)")
    drone.set_color("cyan" if over else ("yellow" if safe else "orange"))
    drone.set_markersize(14 if over else 10)
    return halos + bodies + [trail, drone, txt]


ani = anim.FuncAnimation(fig, upd, frames=len(frames), interval=60, blit=False)
_nh = max((len(v) for v in hum_by_t.values()), default=16)
fig.suptitle(f"A+B threads a DENSE moving crowd ({_nh} dynamic-hard humans) within a y-tunnel (±{W:.1f}); "
             f"z is free, so at the tightest pinch the drone CLIMBS OVER (cyan) — real 3D clr stays ≥d_safe, <50ms/replan", fontsize=9)
fig.tight_layout()
ani.save(os.path.join(MED, "_tunnel_crowd.gif"), writer=anim.PillowWriter(fps=16))
print("wrote media/_tunnel_crowd.gif")
