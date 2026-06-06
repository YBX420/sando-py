"""HARD collision gate for the generated-image scene, pure agile, NO inflation (DYNINFL=0).

The viz/bench only flagged collision when the drone CENTER entered a box (point model). A real
quad has a body radius (par.drone_radius=0.2). This rollout treats every obstacle as a hard wall:
if the drone BODY ever touches a box -> signed_box(center, box(t)) < drone_radius -> RAISE
AssertionError naming the obstacle, time, x, and penetration depth. No collision -> report the
worst body clearance over the whole flight. This is the ground-truth "did it really thread" test.

Run:  python _thread_assert.py
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


def run():
    os.environ.setdefault("DYNINFL", "0")                  # pure agile, no route-around inflation
    par, START, GOAL, OBS, LOOP = B.build()
    par.inflate_walls_by_body = os.environ.get("BODY", "1") == "1"   # body-aware C-space (default ON)
    DR = float(par.drone_radius)                           # body radius -> the hard-collision threshold
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); p_prev = START.copy()
    t = 0.0; nr = 0.0; last_rt = 0.0; reached = False; nstep = 0; nfail = 0
    worst = (np.inf, -1, 0.0, 0.0)                         # (body_clear, obs_id, t, x)
    while t < T_MAX and not reached and nstep < 8000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t)
            if not int(ret[0] if isinstance(ret, tuple) else ret):
                nfail += 1
                print(f"    [replan FAIL @ t={t:.2f} x={p[0]:.1f}]")
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        # dense check: 4 sub-samples on the segment p_prev->p, obstacles at current t
        for s in np.linspace(0.0, 1.0, 5):
            ps = p_prev + s * (p - p_prev)
            for j in range(len(OBS)):
                bc = signed_box(ps, DTS[j].eval(t), SZ[j]) - DR     # BODY clearance
                if bc < worst[0]: worst = (bc, int(DTS[j].id), t, float(ps[0]))
                if bc < 0.0:
                    raise AssertionError(
                        f"COLLISION: drone body hit obstacle id={DTS[j].id} at t={t:.2f}s x={ps[0]:.1f}m "
                        f"penetration={-bc:.3f}m (center_clear={bc + DR:.3f} < drone_radius={DR})")
        p_prev = p.copy()
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    return dict(reached=reached, final_x=float(p[0]), worst=worst, nfail=nfail, t=t, DR=DR)


if __name__ == "__main__":
    r = run()
    bc, oid, bt, bx = r["worst"]
    print("=== HARD body-collision gate (pure agile, no inflation) ===")
    print(f"  reached={r['reached']}  final_x={r['final_x']:.1f}  t={r['t']:.2f}s  replan_fail={r['nfail']}")
    print(f"  drone_radius={r['DR']}  worst BODY clearance={bc:+.3f}m  (obstacle id={oid}, t={bt:.2f}s, x={bx:.1f}m)")
    print("  PASS: no body collision." if bc >= 0 else "  FAIL.")
