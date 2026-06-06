"""Inflation-vs-thread tradeoff curve. Now that INFL=0 THREADS (max|y|~0.5, reaches), how much
dynamic inflation can we afford before the global guide routes wide / the local solve stalls?
Sweet spot = largest dyn_base_inflation_m that still threads (low max|y|) AND reaches fast.
Run: python _infl_sweep.py"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO


def run(infl, t_max=26.0):
    par, START, GOAL, OBS, LOOP = B.build()
    par.dyn_base_inflation_m = float(infl)
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
    return dict(infl=infl, reached=reached, final_x=float(p[0]), maxy=maxy, nfail=nfail, t=t)


print("=== dyn_base_inflation_m sweep: thread (low max|y|) vs route-around / stall ===")
print(f"  {'infl':>5} | {'reached':>7} | {'final_x':>7} | {'max|y|':>6} | {'fails':>5} | {'t(s)':>5}")
for infl in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    r = run(infl)
    print(f"  {r['infl']:5.1f} | {str(r['reached']):>7} | {r['final_x']:7.1f} | "
          f"{r['maxy']:6.2f} | {r['nfail']:5d} | {r['t']:5.1f}")
print("\nlow max|y| = threads straight through; high max|y| = swings wide (route-around).")
