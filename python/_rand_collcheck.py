"""Collision forensics for random scenes: find the worst BODY-collision frame and report WHICH obstacle is
hit (WALL vs human), its speed, the depth, the drone pos, and whether a human was within slow_near (i.e. the
drone was in gentle-slow mode). Distinguishes wall vs human collisions and the slow<->wall coupling.
Run:  python _rand_collcheck.py 14 21 41 55
"""
import os, sys
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX", "RETIME"):
    os.environ.pop(k, None)
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _speed_bench as B
import _rand_scenes as R
from sando_cpp_bridge import RobotState, SANDO


def run(seed):
    par, _, _, _, LOOP = B.build()
    par.v_max = R.VMAX; par.use_st_graph = False; par.use_spacetime_corridor = False; par.force_goal_z = False
    START = np.array([0.0, 0.0, 3.0]); GOAL = np.array([R.GOAL_X, 0.0, 3.0])
    OBS = R.make_scene(seed)
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    DR = float(par.drone_radius)
    is_wall = [200 <= DTS[j].id < 300 for j in range(len(OBS))]
    slow_near = float(getattr(par, "minco_human_slow_near", 3.0))
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); pp = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0
    worst = np.inf; wj = -1; wstate = None
    while t < R.T_MAX and not reached and nstep < 6000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            vh = np.array([v[0], v[1], 0.0]); vhn = float(np.linalg.norm(vh))
            hd = (vh / vhn) if vhn > 0.5 else np.array([1.0, 0.0, 0.0])
            for j, d in enumerate(DTS):
                if R.visible(p, hd, d.eval(t), SZ[j]): sando.add_traj(d, t)   # D435i FOV
            sando.replan(last_rt, t); nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p, v, a = R.step_drone(p, v, ng, float(par.a_max), R.VMAX, DT)   # physics
        # densified body clearance between pp and p (catches grazes the per-step misses)
        for s in np.linspace(0.0, 1.0, 5):
            ps = pp + s * (p - pp)
            for j in range(len(OBS)):
                bc = R.signed_box(ps, DTS[j].eval(t), SZ[j]) - DR
                if bc < worst:
                    worst = bc; wj = j; wstate = (t, ps.copy())
        pp = p.copy(); t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    tw, pw = wstate
    cls = "WALL" if is_wall[wj] else "human"
    spd = float(np.linalg.norm((DTS[wj].eval(tw + 1e-3) - DTS[wj].eval(tw - 1e-3)) / 2e-3))
    bc = DTS[wj].eval(tw)
    # nearest human distance at that moment (was the drone in slow mode?)
    hd = min([np.linalg.norm(DTS[j].eval(tw) - pw) for j in range(len(OBS)) if not is_wall[j]] or [9.9])
    print(f"scene {seed:02d}: reached={int(reached)} worst_body={worst:+.3f} -> HIT {cls}#{wj} (spd={spd:.2f}) "
          f"t={tw:.2f} drone=({pw[0]:.1f},{pw[1]:+.2f},{pw[2]:.2f})")
    print(f"   {cls} center=({bc[0]:.1f},{bc[1]:+.2f},{bc[2]:.2f}) size={list(np.round(SZ[wj],2))} | "
          f"nearest human dist={hd:.2f} (slow_near={slow_near} -> {'SLOW-MODE' if hd<=slow_near else 'full-speed'})")


if __name__ == "__main__":
    for s in ([int(x) for x in sys.argv[1:]] or [14, 21, 41, 55]):
        run(s)
