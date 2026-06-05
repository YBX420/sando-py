"""SAFE-half recovery verification through the REAL production C++ engine (sando_capi.dll),
the same pipeline Isaac Sim drives. A blocking corridor of ONCOMING HARD HUMANS makes
plan_minco fail (no forward path) -> the drone must wait. A/B over Parameters.recovery_enabled:

  OFF (today's freeze) : drone freezes on the stale plan -> an oncoming human walks into it (collision).
  ON  (recovery yield) : drone actively yields to keep clearance, then proceeds when the lane clears.

Run:  python _isaac_recovery_test.py     (uses the C++ engine via sando_cpp_bridge)
"""
import os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from sando_cpp_bridge import (Parameters, RobotState, DynTraj, SANDO,
                              DroneStatus_GOAL_REACHED as GOAL_REACHED)
PLN = yaml.safe_load(open(os.path.join(ROOT, "isaac_sando.yaml"), encoding="utf-8"))["planner"]

Z = 3.0
START = np.array([0.0, 0.0, Z]); GOAL = np.array([12.0, 0.0, Z])
# scene: a soft-wall lane |y|<=1.3 (x in [2,10]) + 4 ONCOMING hard humans marching -x inside it
SCENE = [
    dict(cls="wall",  size=[8.0, 0.6, 6.0], traj=["6", "-1.6", str(Z)],          vel=["0", "0", "0"]),
    dict(cls="wall",  size=[8.0, 0.6, 6.0], traj=["6",  "1.6", str(Z)],          vel=["0", "0", "0"]),
    dict(cls="human", size=[0.6, 0.6, 0.6], traj=["8 - 1.0*t",  "0.0", str(Z)],  vel=["-1.0", "0", "0"]),
    dict(cls="human", size=[0.6, 0.6, 0.6], traj=["9 - 1.0*t",  "0.6", str(Z)],  vel=["-1.0", "0", "0"]),
    dict(cls="human", size=[0.6, 0.6, 0.6], traj=["10 - 1.0*t", "-0.6", str(Z)], vel=["-1.0", "0", "0"]),
    dict(cls="human", size=[0.6, 0.6, 0.6], traj=["11 - 1.0*t", "0.3", str(Z)],  vel=["-1.0", "0", "0"]),
]
IS_HUMAN = [o["cls"] != "wall" for o in SCENE]


def make_dt(i, o):
    dt = DynTraj(); dt.id = (200 + i) if o["cls"] == "wall" else i; dt.mode = "Analytic"
    dt.bbox = np.array(o["size"], float)
    dt.traj_x, dt.traj_y, dt.traj_z = o["traj"]
    dt.traj_vx, dt.traj_vy, dt.traj_vz = o["vel"]
    dt.compile_analytic(); return dt


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def run(recovery):
    par = Parameters()
    for k, v in PLN.items():
        if k == "sando_map_res": par.res = float(v); continue
        if hasattr(par, k): setattr(par, k, v)
    par.recovery_enabled = bool(recovery)
    par.force_goal_z = False                         # our scene fixes z explicitly
    DTS = [make_dt(i, o) for i, o in enumerate(SCENE)]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)

    DT = float(par.dc); RD = 0.1; T_MAX = 30.0
    p_d = START.copy(); v_d = np.zeros(3); a_d = np.zeros(3)
    t = 0.0; last_rt = 0.0; nr = 0.0; nfail = 0; reached = False
    min_hum = np.inf; collided = False; maxy = 0.0; nstep = 0
    while t < T_MAX and not reached and nstep < 12000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p_d.copy(); st.vel = v_d.copy(); st.accel = a_d.copy()
            sando.update_state(st)
            for dt_ in DTS:
                if np.linalg.norm(dt_.eval(t) - p_d) <= 40.0: sando.add_traj(dt_, t)
            ret = sando.replan(last_rt, t)
            ok = bool(ret[0] if isinstance(ret, tuple) else ret)
            if not ok: nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg:
            p_d = np.asarray(ng.pos, float); v_d = np.asarray(ng.vel, float); a_d = np.asarray(ng.accel, float)
        for i, o in enumerate(SCENE):
            if IS_HUMAN[i]:
                d = signed_box(p_d, DTS[i].eval(t), np.array(o["size"], float))
                min_hum = min(min_hum, d)
                if d < 0.0: collided = True
        maxy = max(maxy, abs(p_d[1]))
        t += DT; nstep += 1
        if float(np.linalg.norm(p_d[:2] - GOAL[:2])) < float(par.goal_radius): reached = True
    return dict(reached=reached, min_hum=min_hum, collided=collided, nfail=nfail,
                final_x=float(p_d[0]), maxy=maxy)


print("=== SAFE-half recovery A/B (production C++ engine via sando_capi.dll) ===")
print("blocking corridor: soft-wall lane |y|<=1.3 + 4 oncoming HARD humans marching -x\n")
for rec in (False, True):
    r = run(rec)
    tag = "recovery ON " if rec else "recovery OFF"
    print(f"  {tag} | reached={str(r['reached']):5s} | min_human={r['min_hum']:+.3f} "
          f"{'COLLIDED' if r['collided'] else 'no-collision'} | replan_fail={r['nfail']:4d} "
          f"| final_x={r['final_x']:.1f} max|y|={r['maxy']:.2f}")
print("\nOFF should COLLIDE (human walks into the frozen drone); ON should stay collision-free (active yield).")
