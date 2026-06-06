"""AUDIT: prove every collision/avoidance pathology in the dense moving-crowd scene, in one table.

Runs the SAME scene (pure agile, no route-around inflation) under many configs and measures, per run,
the worst BODY clearance (center-to-box minus drone_radius), whether it COLLIDED, whether it reached,
and the time the drone spends inside the densest cluster (exposure). NON-raising: it tabulates.

What the table proves (cross-checked by the 5 code audits):
 1. worst clearance is INVARIANT to corridor weight / body inflation / motion / soft-vs-hard
    -> local avoidance does not shape the executed path (commit-lock + inert soft + motion-blind hard).
 2. slower COLLIDES (longer exposure) and faster COLLIDES (overshoot) -> "fast=safe" is exposure
    gambling, not a computed solution.
 3. worst clearance is ALWAYS far below d_safe (0.4 center / +0.2 body) -> no real margin is ever
    enforced, even when hard is forced.

Run:  python _audit_collisions.py
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


def run(vmax=12.0, wcor=None, body=True, avoid="", dyn=0.5, t_max=20.0):
    os.environ["DYNINFL"] = "0"                              # pure agile, no route-around inflation
    par, START, GOAL, OBS, LOOP = B.build()
    par.v_max = float(vmax)
    par.inflate_walls_by_body = bool(body)
    par.dynamic_speed_thresh = float(dyn)                    # >0 -> fast movers become hard "dynamic" class
    par.avoid_override = str(avoid)                          # "" per-class; "hard"/"soft" force
    if wcor is not None: par.minco_w_corridor = float(wcor)
    DR = float(par.drone_radius)
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); p_prev = START.copy()
    t = 0.0; nr = 0.0; last_rt = 0.0; reached = False; nstep = 0; nfail = 0
    worst = np.inf; collided = False; expo = 0.0; last_ok = True; breach_after_fail = None
    while t < t_max and not reached and nstep < 8000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t)
            last_ok = bool(int(ret[0] if isinstance(ret, tuple) else ret))
            if not last_ok: nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        if 14.0 <= p[0] <= 18.0: expo += DT                 # time inside the densest cluster
        for s in np.linspace(0.0, 1.0, 5):
            ps = p_prev + s * (p - p_prev)
            for j in range(len(OBS)):
                bc = signed_box(ps, DTS[j].eval(t), SZ[j]) - DR
                if bc < worst: worst = bc
                if bc < 0.0:
                    if not collided: breach_after_fail = (not last_ok)   # was the last replan a FAILURE?
                    collided = True
        p_prev = p.copy(); t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    return dict(reached=reached, final_x=float(p[0]), worst=worst, collided=collided, nfail=nfail,
                expo=expo, breach_after_fail=breach_after_fail)


CONFIGS = [
    # label,                         vmax, wcor,  body, avoid, dyn
    ("vmax=12 OLD (soft walls)",     12.0, None,  True, "",   0.0),   # before: dynamic reclass OFF
    ("vmax=12 NEW (dyn->hard)",      12.0, None,  True, "",   0.5),   # after: fast movers -> hard cert
    ("vmax=6  NEW",                  6.0,  None,  True, "",   0.5),
    ("vmax=9  NEW",                  9.0,  None,  True, "",   0.5),
    ("vmax=16 NEW",                  16.0, None,  True, "",   0.5),
    ("vmax=12 NEW corridor=0",       12.0, 0.0,   True, "",   0.5),   # should now DIFFER from corridor=100
    ("vmax=12 NEW corridor=100",     12.0, 100.0, True, "",   0.5),   # (commit-lock broken => local bites)
    ("vmax=12 NEW body=OFF",         12.0, None,  False, "",  0.5),
]

if __name__ == "__main__":
    print("=== COLLISION AUDIT: dense moving crowd, pure agile, no route-around inflation ===")
    print("   d_safe(wall)=0.4 center -> +0.2 body would be a real margin. drone_radius=0.2.\n")
    print(f"   {'config':<26} {'reached':>7} {'collide':>7} {'worstBody':>9} {'final_x':>7} {'exposure':>8} {'fails':>5}")
    for label, vm, wc, bd, av, dy in CONFIGS:
        r = run(vmax=vm, wcor=wc, body=bd, avoid=av, dyn=dy)
        tag = "  <-- COLLISION" if r["collided"] else ("" if r["reached"] else "  <-- HOLD/no-reach (honest)")
        print(f"   {label:<26} {str(r['reached']):>7} {str(r['collided']):>7} "
              f"{r['worst']:>+9.3f} {r['final_x']:>7.1f} {r['expo']:>7.2f}s {r['nfail']:>5d}{tag}")
    print("\n   If the 4 'vmax=12 *' variants share one worstBody value -> local avoidance is INERT")
    print("   (commit-lock + dominated soft + motion-blind hard). worstBody << +0.2 everywhere -> no")
    print("   real margin is ever computed; collision-free runs are exposure luck, not a solution.")
