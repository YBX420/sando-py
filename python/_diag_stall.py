"""WHY does it stall at INFL=0? Diagnosis: global path is fine (gN>0), drone has clearance, but
local plan_minco fails every tick at full 200ms budget. Soft walls don't gate clearance -> the
failure must be velocity/acceleration overshoot (huge minco_w_time=2500 pushes the threading
trajectory past v_max/a_max). minco_retime_overshoot would DILATE time (slow down) instead of
failing - but the yaml never enables it. Test: retime OFF vs ON at INFL=0.
Run: python _diag_stall.py"""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO


def run(retime, infl=0.0, t_max=26.0):
    par, START, GOAL, OBS, LOOP = B.build()
    par.dyn_base_inflation_m = float(infl)
    par.minco_retime_overshoot = bool(retime)
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    nstep = 0; nfail = 0; reached = False; maxy = 0.0
    while t < t_max and nstep < 6000 and not reached:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for d in DTS:
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t)
            if not int(ret[0] if isinstance(ret, tuple) else ret): nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        maxy = max(maxy, abs(p[1])); t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    return dict(retime=retime, reached=reached, final_x=float(p[0]), maxy=maxy, nfail=nfail, t=t)


print("=== INFL=0 (try to thread). Does enabling minco_retime_overshoot un-stall it? ===")
for rt in (False, True):
    r = run(rt)
    print(f"  retime={str(rt):5s} | reached={str(r['reached']):5s} | final_x={r['final_x']:5.1f} | "
          f"max|y|={r['maxy']:.2f} | replan_fail={r['nfail']:4d} | t={r['t']:.1f}")
print("\nif retime=True reaches (vs retime=False stuck) -> the stall is vel/accel overshoot, fixed by retiming.")
