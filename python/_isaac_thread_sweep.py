"""Why does it route AROUND instead of THREADING? Sweep dyn_base_inflation_m on the production
engine (headless, same scene as Isaac). High inflation -> global heat-A* treats the moving-box
cluster as one big blocked blob -> routes around the edge (huge max|y|). Lower it -> the global
path should find gaps BETWEEN boxes -> thread (low max|y|), if the corridor+local stay safe.

Run:  python _isaac_thread_sweep.py
"""
import os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from sando_cpp_bridge import (Parameters, RobotState, DynTraj, SANDO,
                              DroneStatus_GOAL_REACHED as GOAL_REACHED)
CFG = yaml.safe_load(open(os.path.join(ROOT, "isaac_sando.yaml"), encoding="utf-8"))
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)
OBS = SCENE["obstacles"]; IS_HUMAN = [o.get("class", "wall") != "wall" for o in OBS]


def make_dt(i, o):
    dt = DynTraj(); dt.id = (200 + i) if o.get("class", "wall") == "wall" else i; dt.mode = "Analytic"
    dt.bbox = np.array(o["size"], float)
    if "traj" in o:
        dt.traj_x, dt.traj_y, dt.traj_z = [str(e) for e in o["traj"]]
        if "vel" in o: dt.traj_vx, dt.traj_vy, dt.traj_vz = [str(e) for e in o["vel"]]
    dt.compile_analytic(); return dt


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def run(infl, t_max=28.0):
    par = Parameters()
    for k, v in PLN.items():
        if k == "sando_map_res": par.res = float(v); continue
        if hasattr(par, k): setattr(par, k, v)
    par.dyn_base_inflation_m = float(infl)
    DTS = [make_dt(i, o) for i, o in enumerate(OBS)]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"])
    p = START.copy(); v = np.zeros(3); a = np.zeros(3)
    t = 0.0; last_rt = 0.0; nr = 0.0; nfail = 0; reached = False
    mn = np.inf; maxy = 0.0; nstep = 0
    while t < t_max and not reached and nstep < 6000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy()
            sando.update_state(st)
            for d in DTS:
                if np.linalg.norm(d.eval(t) - p) <= 40.0: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t); ok = bool(ret[0] if isinstance(ret, tuple) else ret)
            if not ok: nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        for i, o in enumerate(OBS):
            mn = min(mn, signed_box(p, DTS[i].eval(t), np.array(o["size"], float)))
        maxy = max(maxy, abs(p[1])); t += DT; nstep += 1
        if float(np.linalg.norm(p[:2] - GOAL[:2])) < float(par.goal_radius): reached = True
    return dict(infl=infl, reached=reached, final_x=float(p[0]), maxy=maxy, mn=mn, nfail=nfail, t=t)


print("=== thread vs route-around: sweep dyn_base_inflation_m (lower = thread through gaps) ===")
print(f"{'infl':>5} | {'reached':7} | {'final_x':7} | {'max|y|':6} | {'min_clr':7} | {'fails':5} | t")
for infl in (1.2, 0.6, 0.3, 0.1):
    r = run(infl)
    print(f"{r['infl']:5.1f} | {str(r['reached']):7} | {r['final_x']:7.1f} | {r['maxy']:6.2f} | "
          f"{r['mn']:+7.3f} | {r['nfail']:5d} | {r['t']:.1f}")
print("\nlower max|y| at small inflation = it threads through the cluster instead of going around it")
print("(watch min_clr stays >= 0 = still safe, and reached/fails = no new stall).")
