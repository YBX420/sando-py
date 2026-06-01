"""SANDO-level REAL-TIME 3D demo (animated GIF, headless via pillow).

A genuine closed-loop / receding-horizon flight in 3D:
  - every tick the planner RE-PLANS from the drone's current state (real-time),
  - the human MOVES between replans (dynamic obstacle),
  - the drone reacts in 3D, keeping the HARD human outside its d_safe shell while
    being free to graze the SOFT wall,
  - shown in a rotating 3D view with the executed trail + current MINCO plan.
Output: demo_anim_perclass.gif
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402

ND = DetourConfig(enabled=False)
START = np.array([0.0, 0.0, 1.0]); GOAL = np.array([8.0, 0.0, 2.0])        # z rises -> 3D
CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4)}
HR, DSAFE = 0.4, 0.8
H0 = np.array([4.0, 0.0, 1.5]); HVEL = np.array([0.0, 0.0, 0.0])           # blocks the path -> forces a 3D arc
DT, SUB = 0.22, 3


def box_faces(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [v[idx] for idx in f]


def sphere_xyz(c, r, n=16):
    u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
    return (c[0] + r * np.outer(np.cos(u), np.sin(v)),
            c[1] + r * np.outer(np.sin(u), np.sin(v)),
            c[2] + r * np.outer(np.ones_like(u), np.cos(v)))


# ---------------- REAL-TIME closed-loop rollout (re-plan every tick) --------
drone = START.copy(); dvel = np.zeros(3); hc = H0.copy(); tg = 0.0
trail = [drone.copy()]; frames = []
for tick in range(48):
    mover = SphereObstacle(hc.copy(), HR, vel=HVEL, class_name="human")
    tr, info = plan_minco(np.linspace(drone, GOAL, 8), [mover], CFG,
                          opt_params=OptParams(), detour_cfg=ND, v0=dvel)
    Pplan = tr.eval(np.linspace(tr.t_start, tr.t_end, 60))
    te = min(DT, tr.t_end)
    for a in np.linspace(0.0, te, SUB, endpoint=False):
        dp = tr.eval(float(a)); hp = hc + HVEL * float(a)
        trail.append(dp.copy())
        frames.append(dict(drone=dp.copy(), human=hp.copy(), planned=Pplan.copy(),
                           trail=np.array(trail), clr=float(np.linalg.norm(dp - hp) - HR),
                           t=tg + float(a)))
    drone = tr.eval(te); dvel = tr.eval_deriv(te, 1); hc = hc + HVEL * te; tg += te
    if np.linalg.norm(drone[:2] - GOAL[:2]) < 0.3:
        break

min_run = [min(f["clr"] for f in frames[:i + 1]) for i in range(len(frames))]
print(f"frames={len(frames)} ticks~{len(frames)//SUB} min_clr={min(min_run):.3f} "
      f"reached={np.linalg.norm(frames[-1]['drone'][:2]-GOAL[:2])<0.4}")

# ---------------- 3D animation ----------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")


def draw(i):
    f = frames[i]
    ax.clear()
    ax.set_xlim(0, 8); ax.set_ylim(-2.2, 2.2); ax.set_zlim(0.4, 2.6)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    # human (HARD): body + translucent d_safe shell (MOVING)
    hx, hy, hz = sphere_xyz(f["human"], HR)
    ax.plot_surface(hx, hy, hz, color="C3", alpha=0.95, linewidth=0)
    sx, sy, sz = sphere_xyz(f["human"], HR + DSAFE)
    ax.plot_surface(sx, sy, sz, color="C3", alpha=0.10, linewidth=0)
    # current MINCO plan + executed trail + drone
    Pp = f["planned"]
    ax.plot(Pp[:, 0], Pp[:, 1], Pp[:, 2], "-", color="C0", alpha=0.5, lw=1.6)
    tr_ = f["trail"]
    ax.plot(tr_[:, 0], tr_[:, 1], tr_[:, 2], "-", color="C1", lw=2.6)
    ax.scatter(*f["drone"], color="k", s=45)
    ax.scatter(*GOAL, color="g", marker="*", s=160)
    ax.plot([], [], color="C1", lw=2.6, label="executed trail")
    ax.plot([], [], color="C0", lw=1.6, label="current MINCO plan (replanned each tick)")
    ax.plot([], [], color="C3", lw=6, alpha=0.4, label="human (HARD) + d_safe")
    ax.set_title(f"REAL-TIME 3D: replan every tick, drone arcs around the HARD obstacle   "
                 f"t={f['t']:.1f}s   clearance={f['clr']:.2f} m (d_safe {DSAFE})")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-60 + 0.35 * i)        # slow rotation


anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_anim_perclass.gif")
anim.save(out, writer=PillowWriter(fps=12))
print("saved", out)
