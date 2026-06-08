"""Forensics on the BODY-COLLISION the rollout produces: at every breach step dump WHICH box is hit, its
SPEED at that instant (finite-diff of the analytic traj -> is it >= dynamic_speed_thresh, i.e. should it be
HARD-certified, or slow/static -> SOFT-by-design?), the body clearance, and whether the most-recent replan
SUCCEEDED (ok) or the drone is flying a held/old commit. Mirrors _viz3d/_check_clr rollout exactly.
Env: STG STC STDSD VMAX INFL  (set to match the colliding gif).
Run:  python _diag_collision.py
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

    def box_speed(j, t):                              # |d/dt center| via central finite diff
        h = 1e-3
        return float(np.linalg.norm((DTS[j].eval(t + h) - DTS[j].eval(t - h)) / (2 * h)))

    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0; last_ok = None; n_breach = 0
    while t < T_MAX and not reached and nstep < 9000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t); last_ok = bool(ret[0] if isinstance(ret, tuple) else ret)
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        # nearest box (by signed body clearance)
        jbest, sbest = -1, np.inf
        for j in range(len(OBS)):
            d = signed_box(p, DTS[j].eval(t), SZ[j])
            if d < sbest: sbest, jbest = d, j
        body = sbest - DR
        if body < 0.0:
            n_breach += 1
            spd = box_speed(jbest, t)
            cls = "HARD(moving)" if spd >= thresh else "soft(slow/static)"
            bc = DTS[jbest].eval(t)
            print(f"BREACH t={t:5.2f} body={body:+.3f} | box#{jbest} spd={spd:5.2f}->{cls} "
                  f"boxc=({bc[0]:5.1f},{bc[1]:+.2f},{bc[2]:.2f}) sz={SZ[jbest]} "
                  f"drone=({p[0]:5.1f},{p[1]:+.2f},{p[2]:.2f}) | last_replan_ok={last_ok}")
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    print(f"--- total breach steps={n_breach} reached={reached} t={t:.2f} thresh={thresh} ---")


if __name__ == "__main__":
    main()
