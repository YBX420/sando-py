"""Diagnose WHY given random scenes (e.g. 05, 12) stall: instrumented closed-loop rollout (no render).
Reports per scene: reached, final_x, where forward progress stalls, replan ok/fail counts, and AT the stall
whether there is escape space (a +z/+y/-y nudge that clears the body) -> distinguishes a genuine block
(no gap) from a FALSE HOLD (free space exists but the planner froze). Uses the same scenes as _rand_scenes.
Run:  python _rand_diag.py 5 12 [...seeds]
"""
import os, sys
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX"):
    os.environ.pop(k, None)
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
import _rand_scenes as R
from sando_cpp_bridge import RobotState, SANDO


def diag(seed):
    par, _, _, _, LOOP = B.build()
    par.v_max = R.VMAX; par.use_st_graph = False; par.use_spacetime_corridor = False; par.force_goal_z = False
    START = np.array([0.0, 0.0, 3.0]); GOAL = np.array([R.GOAL_X, 0.0, 3.0])
    OBS = R.make_scene(seed)
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    DR = float(par.drone_radius)
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0; nrep = 0; nfail = 0
    max_x = -1e9; stall_t0 = 0.0; stall = None
    while t < R.T_MAX and not reached and nstep < 6000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            vh = np.array([v[0], v[1], 0.0]); vhn = float(np.linalg.norm(vh))
            hd = (vh / vhn) if vhn > 0.5 else np.array([1.0, 0.0, 0.0])
            for j, d in enumerate(DTS):
                if R.visible(p, hd, d.eval(t), SZ[j]): sando.add_traj(d, t)   # D435i FOV
            ret = sando.replan(last_rt, t); nrep += 1
            if not bool(ret[0] if isinstance(ret, tuple) else ret): nfail += 1
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p, v, a = R.step_drone(p, v, ng, float(par.a_max), R.VMAX, DT)   # physics
        if p[0] > max_x + 0.3: max_x = p[0]; stall_t0 = t; stall = (t, p.copy())
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    # at the last forward-progress point, probe escape space
    ts, ps = stall if stall else (t, p)
    def bmin(q, tt): return min(R.signed_box(q, DTS[j].eval(tt), SZ[j]) - DR for j in range(len(OBS)))
    here = bmin(ps, ts)
    esc = {}
    for nm, dvec in [("+z", [0,0,1.]), ("+y", [0,1.,0]), ("-y", [0,-1.,0]), ("+x", [1.,0,0])]:
        best = -9.0
        for s in np.arange(0.0, 1.21, 0.1):
            q = ps + s*np.array(dvec)
            if not (par.z_min <= q[2] <= par.z_max): continue
            best = max(best, bmin(q, ts))
        esc[nm] = best
    print(f"scene {seed:02d}: reached={int(reached)} final_x={p[0]:5.1f} STALLED@x={ps[0]:5.1f}(t={ts:4.1f}) "
          f"| replans {nrep} fails {nfail} | nObs={len(OBS)}")
    print(f"   at stall: body_clr_here={here:+.2f}  escape best within 1.2m: "
          + "  ".join(f"{k}={v:+.2f}" for k, v in esc.items()))
    # list nearby obstacles
    near = []
    for j in range(len(OBS)):
        bc = R.signed_box(ps, DTS[j].eval(ts), SZ[j]) - DR
        if bc < 2.5:
            cls = "WALL" if 200 <= DTS[j].id < 300 else "human"
            c = DTS[j].eval(ts); near.append((bc, j, cls, c, SZ[j]))
    for bc, j, cls, c, sz in sorted(near):
        print(f"     obs#{j} {cls} clr={bc:+.2f} c=({c[0]:4.1f},{c[1]:+.2f},{c[2]:.2f}) sz={list(np.round(sz,2))}")


def main():
    seeds = [int(x) for x in sys.argv[1:]] or [5, 12]
    for s in seeds:
        diag(s)


if __name__ == "__main__":
    main()
