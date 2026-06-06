"""LOW-COST multi-view (no Isaac): headless C++-engine rollout, then 4 matplotlib panels —
TOP (x-y), SIDE (x-z), FRONT (y-z), and speed(x). Body-COLLISIONS are marked RED on every panel
(signed_box(center) < drone_radius), with the worst hit annotated. This is the cheap way to see the
side/front the Isaac GUI shows, and to check the "early avoids, later rams" claim.

  STC=1  ST-graph corridor on   STG=1  ST-graph timing on   STDSD=0.25  seed keep-out   VMAX=..  cap
Run:  python _viz3.py            (writes media/_viz3.png)
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


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
    p = START.copy(); v = np.zeros(3); a = np.zeros(3)
    t = 0.0; nr = 0.0; last_rt = 0.0; reached = False; nstep = 0; nfail = 0
    traj = []; hits = []                     # hits: (x,y,z, obstacle box center, sz) where body collides
    while t < T_MAX and not reached and nstep < 9000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t)
            if not int(ret[0] if isinstance(ret, tuple) else ret): nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        cmin = np.inf; chit = None
        for j in range(len(OBS)):
            d = signed_box(p, DTS[j].eval(t), SZ[j]) - DR
            if d < cmin: cmin = d; chit = j
        traj.append((p[0], p[1], p[2], float(np.linalg.norm(v)), cmin))
        if cmin < 0.0 and chit is not None:
            hits.append((p[0], p[1], p[2], DTS[chit].eval(t).copy(), SZ[chit]))
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    return np.array(traj), hits, SZ, START, GOAL, reached, t, nfail, par


def main():
    traj, hits, SZ, START, GOAL, reached, t_end, nfail, par = record()
    x, y, z, sp, clr = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4]
    stg = os.environ.get("STG", "0") == "1"
    print(f"reached={reached} t={t_end:.2f}s nfail={nfail} min_body_clr={clr.min():+.3f} "
          f"collisions={len(hits)} STG={int(stg)}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    def path(axp, ix, iy, xl, yl, title):
        pts = np.array([traj[:, ix], traj[:, iy]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", norm=plt.Normalize(0, max(8.0, sp.max())))
        lc.set_array(sp[:-1]); lc.set_linewidth(2.0); axp.add_collection(lc)
        for h in hits:                                    # red collision markers
            axp.plot(h[ix], h[iy], "x", color="red", ms=9, mew=2)
        axp.plot(START[ix], START[iy], "o", color="lime", ms=8)
        axp.plot(GOAL[ix], GOAL[iy], "*", color="red", ms=13)
        axp.set_xlabel(xl); axp.set_ylabel(yl); axp.set_title(title); axp.grid(alpha=0.3)
        axp.autoscale()
    path(ax[0, 0], 0, 1, "x (m)", "y (m)", "TOP (x-y)")
    path(ax[0, 1], 0, 2, "x (m)", "z (m)", "SIDE (x-z)")
    path(ax[1, 0], 1, 2, "y (m)", "z (m)", "FRONT (y-z)")
    ax[1, 1].plot(x, sp, "-", color="tab:blue", lw=1.3)
    for h in hits: ax[1, 1].axvline(h[0], color="red", alpha=0.25)
    ax[1, 1].set_xlabel("x (m)"); ax[1, 1].set_ylabel("speed (m/s)")
    ax[1, 1].set_title(f"speed  reached={reached} t={t_end:.1f}s nfail={nfail} collisions={len(hits)}")
    ax[1, 1].grid(alpha=0.3)
    fig.suptitle(f"multi-view  STG={int(stg)}  min_body_clr={clr.min():+.3f}m  "
                 f"(red x / red lines = body collision)", fontsize=12)
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "_viz3.png"), dpi=110)
    print("wrote media/_viz3.png")


if __name__ == "__main__":
    main()
