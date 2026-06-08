"""FORCED-vs-MISSED probe. Replays the rollout to the WORST body-collision frame, then at that exact
frame+time asks: was there collision-free space the drone could have been in instead (a MISSED decision),
or was it boxed in (a person on the only escape side -> FORCED to graze the wall)?

At the worst frame it reports:
  (1) census of every box within 1.5 m body-clearance (static wall vs moving 'person');
  (2) clearance achievable by a pure +z / -z / +y / -y / +x / -x nudge (how far to get the body clear);
  (3) the smallest-norm offset that makes the body clear of ALL boxes (>=0) and SAFE (>=0.15), and whether
      that offset stays clear over a +-0.3 s window (so it is not just an instantaneous gap);
  (4) on that escape side, is any MOVING box within 1.5 m (i.e. would escaping hit a person)?
Env knobs match _viz3d/_diag: STG STC STDSD VMAX INFL.
Run:  python _probe_freespace.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def main():
    os.environ.setdefault("INFL", "0")
    par, START, GOAL, OBS, LOOP = B.build()
    par.dyn_base_inflation_m = float(os.environ.get("INFL", 0.0))
    par.use_spacetime_corridor = os.environ.get("STC", "0") == "1"
    par.use_st_graph = os.environ.get("STG", "0") == "1"
    if os.environ.get("STDSD"): par.stc_d_safe_dyn = float(os.environ["STDSD"])
    thresh = float(getattr(par, "dynamic_speed_thresh", 0.5))
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    DR = float(par.drone_radius)
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0))

    def box_speed(j, t):
        h = 1e-3
        return float(np.linalg.norm((DTS[j].eval(t + h) - DTS[j].eval(t - h)) / (2 * h)))

    def body_min_at(q, t):                            # min body clearance over ALL boxes at time t
        return min(signed_box(q, DTS[j].eval(t), SZ[j]) - DR for j in range(len(OBS)))

    # --- pass 1: find worst body-collision frame ---
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0
    worst = np.inf; worst_state = None
    while t < T_MAX and not reached and nstep < 9000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            sando.replan(last_rt, t); nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        b = body_min_at(p, t)
        if b < worst: worst = b; worst_state = (t, p.copy(), v.copy())
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True

    tw, pw, vw = worst_state
    print(f"=== WORST body frame: t={tw:.2f}s  drone=({pw[0]:.2f},{pw[1]:.2f},{pw[2]:.2f}) "
          f"vel=({vw[0]:.2f},{vw[1]:.2f},{vw[2]:.2f}) |v|={np.linalg.norm(vw):.2f}  body_clr={worst:+.3f} ===")
    print(f"    z bounds [{par.z_min},{par.z_max}]  y bounds [{par.y_min},{par.y_max}]  drone_radius={DR}\n")

    # --- (1) census of nearby boxes ---
    print("(1) boxes within 1.5 m body-clearance of the drone at this instant:")
    rows = []
    for j in range(len(OBS)):
        bc = signed_box(pw, DTS[j].eval(tw), SZ[j]) - DR
        if bc < 1.5:
            spd = box_speed(j, tw); cen = DTS[j].eval(tw)
            rows.append((bc, j, spd, cen, SZ[j]))
    for bc, j, spd, cen, sz in sorted(rows):
        kind = "MOVER" if spd >= thresh else "static"
        print(f"    box#{j:3d} body_clr={bc:+.3f} spd={spd:4.2f} {kind:6s} "
              f"c=({cen[0]:5.1f},{cen[1]:+.2f},{cen[2]:.2f}) sz={sz}")

    # --- (2) pure-axis nudge clearances ---
    print("\n(2) how far a pure nudge must go to clear the body (min over all boxes >= 0):")
    for name, d in [("+z", np.array([0, 0, 1.])), ("-z", np.array([0, 0, -1.])),
                    ("+y", np.array([0, 1., 0])), ("-y", np.array([0, -1., 0])),
                    ("+x", np.array([1., 0, 0])), ("-x", np.array([-1., 0, 0]))]:
        hit = None
        for s in np.arange(0.0, 1.21, 0.05):
            q = pw + s * d
            if name in ("+z", "-z") and not (par.z_min <= q[2] <= par.z_max): continue
            if body_min_at(q, tw) >= 0.0:
                hit = s; break
        print(f"    {name}: clear at {hit:.2f} m" if hit is not None else f"    {name}: NOT clear within 1.2 m")

    # --- (3) smallest-norm clear+safe offset, with time-window robustness ---
    print("\n(3) smallest offset making the body SAFE (>=0.15) of all boxes, +-0.3s robust:")
    best = None
    for dz in np.arange(-0.6, 0.61, 0.1):
        for dy in np.arange(-0.9, 0.91, 0.1):
            for dx in (-0.3, 0.0, 0.3):
                off = np.array([dx, dy, dz]); q = pw + off
                if not (par.z_min <= q[2] <= par.z_max): continue
                cl = body_min_at(q, tw)
                if cl < 0.15: continue
                win = min(body_min_at(q, tw + s) for s in (-0.3, -0.15, 0.0, 0.15, 0.3))
                if win < 0.0: continue
                nrm = float(np.linalg.norm(off))
                if best is None or nrm < best[0]: best = (nrm, off.copy(), cl, win)
    if best is not None:
        nrm, off, cl, win = best
        side = "MOVER-side" if False else None
        print(f"    YES: offset ({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f}) |off|={nrm:.2f} m -> "
              f"body_clr={cl:+.3f} (window-min {win:+.3f})")
        # --- (4) is a mover blocking that escape side? ---
        edir = off / (np.linalg.norm(off) + 1e-9)
        blocked = []
        for bc, j, spd, cen, sz in rows:
            if spd < thresh: continue
            to_box = (cen - pw); proj = float(to_box @ edir)
            if proj > 0 and np.linalg.norm(to_box) < 1.5:
                blocked.append((j, np.linalg.norm(to_box), proj))
        print(f"    escape direction ~({edir[0]:+.2f},{edir[1]:+.2f},{edir[2]:+.2f}); "
              f"movers blocking that side within 1.5 m: {blocked if blocked else 'NONE'}")
    else:
        print("    NONE found within +-0.6z/+-0.9y/+-0.3x -> genuinely boxed in at this instant")

    print(f"\n--- reached={reached} t_end={t:.2f}s worst_body={worst:+.3f} ---")


if __name__ == "__main__":
    main()
