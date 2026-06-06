"""LIGHT visualization (no Isaac GUI needed): headless C++-engine rollout (same algorithm + same
analytic obstacle motion Isaac drives), then plot with matplotlib (Agg). Compute-then-draw = fast,
repeatable, cheap. Shows the dynamic-obstacle INFLATION HALO (orange) so you can see route-around
vs thread.

  media/_viz_infl<I>.png   top-down flown path (speed-coloured) + obstacle boxes + inflation halos
  media/_viz_infl<I>.gif   top-down animation: obstacles + halos move, drone flies

dyn_base_inflation_m comes from isaac_sando.yaml; override it to test threading:
  INFL=0   python _viz_run.py     # inflation OFF  -> global tries to thread the gaps
  INFL=1.2 python _viz_run.py     # current        -> halos merge into a wall -> routes around
"""
import os, sys, time
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
    infl = float(os.environ.get("INFL", par.dyn_base_inflation_m))
    par.dyn_base_inflation_m = infl                       # toggle the dynamic-obstacle inflation
    par.use_spacetime_corridor = os.environ.get("STC", "0") == "1"   # draw the space-time corridor
    par.use_st_graph = os.environ.get("STG", "0") == "1"             # ST-graph timing front-end
    if os.environ.get("STDSD"): par.stc_d_safe_dyn = float(os.environ["STDSD"])  # ST-graph seed keep-out
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)

    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3)
    t = 0.0; nr = 0.0; last_rt = 0.0; reached = False; nstep = 0; nfail = 0
    traj = []; frames = []
    while t < T_MAX and not reached and nstep < 8000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy()
            sando.update_state(st)
            for j, dt in enumerate(DTS):
                if np.linalg.norm(dt.eval(t) - p) <= cull:
                    sando.add_traj(dt, t)
            a0 = time.perf_counter(); ret = sando.replan(last_rt, t); last_rt = time.perf_counter() - a0
            if not bool(ret[0] if isinstance(ret, tuple) else ret): nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg:
            p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        traj.append((p[0], p[1], float(np.linalg.norm(v))))
        if nstep % 5 == 0:
            cen = [(DTS[j].eval(t)[0], DTS[j].eval(t)[1]) for j in range(len(OBS))]
            cor = sando.get_corridor()                    # space-time corridor (empty when STC off)
            frames.append((t, (p[0], p[1]), cen, cor))
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius):
            reached = True
    return np.array(traj), frames, SZ, START, GOAL, reached, t, infl, nfail


def main():
    traj, frames, SZ, START, GOAL, reached, t_end, infl, nfail = record()
    xs, ys, sp = traj[:, 0], traj[:, 1], traj[:, 2]
    vmax_c = max(8.0, float(sp.max()))
    tag = f"infl{infl:.2f}".replace(".", "p")
    print(f"INFL={infl:.2f}: {len(traj)} ticks, reached={reached}, t={t_end:.2f}s, "
          f"max|y|={np.abs(ys).max():.2f}, replan_fail={nfail}")

    def halos(ax, centers):                              # orange inflated boxes (the routed footprint)
        if infl <= 0: return
        for (cx, cy), s in zip(centers, SZ):
            ax.add_patch(Rectangle((cx - (s[0] + 2 * infl) / 2, cy - (s[1] + 2 * infl) / 2),
                                   s[0] + 2 * infl, s[1] + 2 * infl,
                                   facecolor="orange", edgecolor="none", alpha=0.12))

    # ---- static PNG ----
    fig, (axp, axs) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(0, vmax_c))
    lc.set_array(sp[:-1]); lc.set_linewidth(2.5); axp.add_collection(lc)
    _, _, cen0, _cor0 = frames[0]
    halos(axp, cen0)
    for (cx, cy), s in zip(cen0, SZ):
        axp.add_patch(Rectangle((cx - s[0] / 2, cy - s[1] / 2), s[0], s[1],
                                facecolor="0.7", edgecolor="0.5", alpha=0.45, lw=0.4))
    axp.plot(START[0], START[1], "o", color="lime", ms=9, label="start")
    axp.plot(GOAL[0], GOAL[1], "*", color="red", ms=15, label="goal")
    axp.set_xlim(-2, 62); axp.set_ylim(-8, 8); axp.set_aspect("equal")
    axp.set_title(f"dyn_inflation={infl:.2f}  reached={reached}  max|y|={np.abs(ys).max():.2f}  "
                  f"replan_fail={nfail}  (orange=inflation halo, gray=real box)")
    axp.legend(loc="upper right"); axp.set_ylabel("y (m)")
    cb = fig.colorbar(lc, ax=axp, fraction=0.025); cb.set_label("speed (m/s)")
    axs.plot(xs, sp, "-", color="tab:blue", lw=1.5); axs.axhline(8.0, color="red", ls="--", lw=1)
    axs.set_xlim(-2, 62); axs.set_ylim(0, vmax_c + 1)
    axs.set_xlabel("x (m)"); axs.set_ylabel("speed (m/s)"); axs.grid(alpha=0.3)
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", f"_viz_{tag}.png"), dpi=110)
    print(f"wrote media/_viz_{tag}.png")

    # ---- animated GIF ----
    figa, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(-2, 62); ax.set_ylim(-8, 8); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.plot(GOAL[0], GOAL[1], "*", color="red", ms=14, zorder=5)
    halo_pc = PatchCollection([], facecolor="orange", edgecolor="none", alpha=0.12); ax.add_collection(halo_pc)
    boxes = PatchCollection([], facecolor="0.6", edgecolor="0.4", alpha=0.7); ax.add_collection(boxes)
    corr_pc = PatchCollection([], facecolor="none", edgecolor="limegreen", lw=1.4); ax.add_collection(corr_pc)
    trail, = ax.plot([], [], "-", color="deepskyblue", lw=2)
    drone, = ax.plot([], [], "o", color="yellow", mec="k", ms=9, zorder=6)
    txt = ax.text(0.5, 6.6, "", fontsize=10)
    tx = [f[1][0] for f in frames]; ty = [f[1][1] for f in frames]

    def upd(k):
        tt, (dx, dy), cen, cor = frames[k]
        boxes.set_paths([Rectangle((cx - s[0] / 2, cy - s[1] / 2), s[0], s[1]) for (cx, cy), s in zip(cen, SZ)])
        if infl > 0:
            halo_pc.set_paths([Rectangle((cx - (s[0] + 2 * infl) / 2, cy - (s[1] + 2 * infl) / 2),
                                         s[0] + 2 * infl, s[1] + 2 * infl) for (cx, cy), s in zip(cen, SZ)])
        # space-time corridor cuboids (x-y face); window-gated: bright when active at tt, dim otherwise
        cps, ecs = [], []
        for cb in cor:
            lo, hi = cb["lo"], cb["hi"]
            cps.append(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1]))
            active = cb["t_l"] <= tt <= cb["t_u"]
            ecs.append((0.0, 0.9, 0.0, 0.95) if active else (0.0, 0.6, 0.0, 0.25))
        corr_pc.set_paths(cps); corr_pc.set_edgecolor(ecs)
        trail.set_data(tx[:k + 1], ty[:k + 1]); drone.set_data([dx], [dy])
        txt.set_text(f"t={tt:4.1f}s  x={dx:4.1f}m  STC boxes={len(cor)}")
        return halo_pc, boxes, corr_pc, trail, drone, txt

    ani = anim.FuncAnimation(figa, upd, frames=len(frames), interval=50, blit=False)
    figa.tight_layout()
    ani.save(os.path.join(ROOT, "media", f"_viz_{tag}.gif"), writer=anim.PillowWriter(fps=20))
    print(f"wrote media/_viz_{tag}.gif")


if __name__ == "__main__":
    main()
