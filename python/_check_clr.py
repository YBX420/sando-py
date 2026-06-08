"""Per-STEP body-clearance audit of the same rollout _viz3d animates (NOT the 7-step-subsampled gif title).
Honest answer to "are there real collisions?": body_clr = signed_dist(center,box) - drone_radius; <0 = body
touches a box. Reports min over EVERY step, # of breach steps, and where/when the worst breach is.
Respects the same env knobs as _viz3d: STG STC STDSD VMAX INFL.
Run:  python _check_clr.py            (uses isaac_sando.yaml; set VMAX/STG/... to match a gif)
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
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    DR = float(par.drone_radius)
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0
    min_body = np.inf; min_at = None; n_breach = 0   # body_clr<0 steps
    min_dsafe = np.inf                                # center-to-surface (no body radius), for context
    while t < T_MAX and not reached and nstep < 9000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            sando.replan(last_rt, t); nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        sd = min(signed_box(p, DTS[j].eval(t), SZ[j]) for j in range(len(OBS)))
        body = sd - DR
        min_dsafe = min(min_dsafe, sd)
        if body < min_body: min_body = body; min_at = (t, p.copy())
        if body < 0.0: n_breach += 1
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    tag = f"STG={int(par.use_st_graph)} STC={int(par.use_spacetime_corridor)} vmax={par.v_max:.0f}"
    print(f"[{tag}] reached={reached} t={t:.2f}s steps={nstep}")
    print(f"    min_body_clr = {min_body:+.3f} m   (drone surface to nearest box; <0 = real body collision)")
    print(f"    min_center_clr = {min_dsafe:+.3f} m  (center to surface, no body radius)")
    print(f"    breach_steps = {n_breach} / {nstep}", end="")
    if min_at is not None:
        print(f"   worst @ t={min_at[0]:.2f}s pos=({min_at[1][0]:.1f},{min_at[1][1]:.2f},{min_at[1][2]:.2f})")
    else:
        print()
    print("    VERDICT:", "*** REAL BODY COLLISION ***" if min_body < 0 else "no body collision (clear)")


if __name__ == "__main__":
    main()
