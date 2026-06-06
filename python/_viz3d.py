"""LOW-COST 3D animated gif (no Isaac): headless C++-engine rollout, then a matplotlib-3D animation of the
drone (+trail) threading the MOVING obstacle boxes in 3D. Shows the z-climb / flyover the top-down view hides.
Obstacles culled to a window around the drone for speed. Body-collisions (if any) flash the drone red.

  STC=1 corridor  STG=1 ST-graph timing  STDSD=0.25 seed keep-out
Run:  python _viz3d.py            (writes media/_viz3d.gif)
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def box_faces(c, s):                                  # 6 quad faces of the AABB (center c, full size s)
    h = 0.5 * np.asarray(s); c = np.asarray(c)
    x0, y0, z0 = c - h; x1, y1, z1 = c + h
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[V[i] for i in f] for f in idx]


def record():
    os.environ.setdefault("INFL", "0")
    par, START, GOAL, OBS, LOOP = B.build()
    par.dyn_base_inflation_m = float(os.environ.get("INFL", 0.0))
    par.use_spacetime_corridor = os.environ.get("STC", "0") == "1"
    par.use_st_graph = os.environ.get("STG", "0") == "1"
    if os.environ.get("STDSD"): par.stc_d_safe_dyn = float(os.environ["STDSD"])
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0)); DR = float(par.drone_radius)
    WIN = 9.0                                          # x-window of obstacles to draw (speed)
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0; frames = []
    while t < T_MAX and not reached and nstep < 9000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            sando.replan(last_rt, t); nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        cmin = min(signed_box(p, DTS[j].eval(t), SZ[j]) - DR for j in range(len(OBS)))
        if nstep % 7 == 0:                             # sample frames (~0.14 s)
            boxes = [(DTS[j].eval(t).copy(), SZ[j]) for j in range(len(OBS))
                     if abs(DTS[j].eval(t)[0] - p[0]) < WIN]
            frames.append((t, p.copy(), boxes, cmin))
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    return frames, START, GOAL, reached, t


def main():
    frames, START, GOAL, reached, t_end = record()
    stg = os.environ.get("STG", "0") == "1"
    print(f"frames={len(frames)} reached={reached} t={t_end:.2f}s STG={int(stg)}")
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection="3d")
    tx = [f[1][0] for f in frames]; ty = [f[1][1] for f in frames]; tz = [f[1][2] for f in frames]

    def upd(k):
        ax.cla()
        ax.set_xlim(0, 60); ax.set_ylim(-4, 4); ax.set_zlim(0.5, 5.0)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        tt, p, boxes, cmin = frames[k]
        polys = []
        for c, s in boxes: polys += box_faces(c, s)
        if polys:
            pc = Poly3DCollection(polys, alpha=0.18, facecolor="0.5", edgecolor="0.35", linewidths=0.2)
            ax.add_collection3d(pc)
        ax.plot(tx[:k + 1], ty[:k + 1], tz[:k + 1], color="deepskyblue", lw=2)
        col = "red" if cmin < 0 else "yellow"
        ax.scatter([p[0]], [p[1]], [p[2]], color=col, edgecolor="k", s=60)
        ax.scatter([GOAL[0]], [GOAL[1]], [GOAL[2]], color="red", marker="*", s=120)
        ax.view_init(elev=22, azim=-72)
        ax.set_title(f"t={tt:4.1f}s  x={p[0]:4.1f}  z={p[2]:4.2f}  body_clr={cmin:+.2f}m  STG={int(stg)}")
        return []

    ani = anim.FuncAnimation(fig, upd, frames=len(frames), interval=70, blit=False)
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    ani.save(os.path.join(ROOT, "media", "_viz3d.gif"), writer=anim.PillowWriter(fps=14))
    print("wrote media/_viz3d.gif")


if __name__ == "__main__":
    main()
