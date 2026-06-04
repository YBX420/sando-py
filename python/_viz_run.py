"""可视化当前 isaac_sando.yaml 场景下的一次飞行(只读场景,绝不写回)。真 C++ 引擎闭环,
障碍按 DynTraj 解析运动 = Isaac 背后那套物理。输出:
  _viz_run.gif  俯视(x-y)动图:移动障碍(灰框)+ 无人机(速度着色轨迹)
  _viz_run.png  速度-x 剖面 + 俯视全程轨迹
跑:  python _viz_run.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.animation as anim

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO


def record():
    par, START, GOAL, OBS, LOOP = B.build()
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)

    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3)
    t = 0.0; nr = 0.0; last_rt = 0.0; reached = False; nstep = 0
    traj = []           # (x, y, speed)
    frames = []         # (t, drone_xy, [obstacle_centers_xy])
    while t < T_MAX and not reached and nstep < 8000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy()
            sando.update_state(st)
            for j, dt in enumerate(DTS):
                if np.linalg.norm(dt.eval(t) - p) <= cull:
                    sando.add_traj(dt, t)
            import time as _tm
            a0 = _tm.perf_counter(); sando.replan(last_rt, t); last_rt = _tm.perf_counter() - a0
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg:
            p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        sp = float(np.linalg.norm(v))
        traj.append((p[0], p[1], sp))
        if nstep % 5 == 0:   # subsample frames for the gif (every 0.1 s)
            cen = [(DTS[j].eval(t)[0], DTS[j].eval(t)[1]) for j in range(len(OBS))]
            frames.append((t, (p[0], p[1]), cen))
        t += DT; nstep += 1
        if sando.get_drone_status() and float(np.linalg.norm(p - GOAL)) < float(par.goal_radius):
            reached = True
    return np.array(traj), frames, SZ, START, GOAL, reached, t


def main():
    traj, frames, SZ, START, GOAL, reached, t_end = record()
    xs, ys, sp = traj[:, 0], traj[:, 1], traj[:, 2]
    vmax_c = max(8.0, float(sp.max()))
    print(f"recorded {len(traj)} ticks, {len(frames)} frames, reached={reached}, t={t_end:.2f}s, "
          f"mean_speed={sp.mean():.2f} max={sp.max():.2f}")

    # ---- static PNG: top-down path (speed-coloured) + speed-vs-x ----
    fig, (axp, axs) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(0, vmax_c))
    lc.set_array(sp[:-1]); lc.set_linewidth(2.5)
    axp.add_collection(lc)
    # obstacles at their START positions (faint) for context
    t0, _, cen0 = frames[0]
    for (cx, cy), s in zip(cen0, SZ):
        axp.add_patch(Rectangle((cx - s[0] / 2, cy - s[1] / 2), s[0], s[1],
                                facecolor="0.7", edgecolor="0.5", alpha=0.35, lw=0.4))
    axp.plot(START[0], START[1], "o", color="lime", ms=9, label="start")
    axp.plot(GOAL[0], GOAL[1], "*", color="red", ms=15, label="goal")
    axp.set_xlim(-2, 62); axp.set_ylim(-6, 6); axp.set_aspect("equal")
    axp.set_title(f"flown path (speed-coloured)  |  cruise sustained, reached={reached}")
    axp.legend(loc="upper right"); axp.set_ylabel("y (m)")
    cb = fig.colorbar(lc, ax=axp, fraction=0.025); cb.set_label("speed (m/s)")
    axs.plot(xs, sp, "-", color="tab:blue", lw=1.5)
    axs.axhline(8.0, color="red", ls="--", lw=1, label="8 m/s target")
    axs.set_xlim(-2, 62); axs.set_ylim(0, vmax_c + 1)
    axs.set_xlabel("x (m)"); axs.set_ylabel("speed (m/s)"); axs.legend(loc="lower right"); axs.grid(alpha=0.3)
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "_viz_run.png"), dpi=110)
    print("wrote _viz_run.png")

    # ---- animated GIF: top-down, obstacles move, drone flies with a fading trail ----
    figa, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(-2, 62); ax.set_ylim(-6, 6); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.plot(GOAL[0], GOAL[1], "*", color="red", ms=14, zorder=5)
    boxes = PatchCollection([], facecolor="0.6", edgecolor="0.4", alpha=0.6)
    ax.add_collection(boxes)
    trail, = ax.plot([], [], "-", color="deepskyblue", lw=2)
    drone, = ax.plot([], [], "o", color="yellow", mec="k", ms=9, zorder=6)
    txt = ax.text(0.5, 5.0, "", fontsize=10)
    tx = [f[1][0] for f in frames]; ty = [f[1][1] for f in frames]

    def upd(k):
        tt, (dx, dy), cen = frames[k]
        rects = [Rectangle((cx - s[0] / 2, cy - s[1] / 2), s[0], s[1]) for (cx, cy), s in zip(cen, SZ)]
        boxes.set_paths(rects)
        trail.set_data(tx[:k + 1], ty[:k + 1])
        drone.set_data([dx], [dy])
        txt.set_text(f"t={tt:4.1f}s   x={dx:4.1f}m")
        return boxes, trail, drone, txt

    ani = anim.FuncAnimation(figa, upd, frames=len(frames), interval=50, blit=False)
    figa.tight_layout()
    ani.save(os.path.join(ROOT, "media", "_viz_run.gif"), writer=anim.PillowWriter(fps=20))
    print("wrote _viz_run.gif")


if __name__ == "__main__":
    main()
