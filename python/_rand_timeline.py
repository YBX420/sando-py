"""Timeline of a stuck scene: per replan tick print drone pos/vel/status, replan ok, and the nearest
HUMAN (clr + its y) and nearest WALL (clr). Shows WHEN it starts failing, whether the crossing human
actually leaves, and whether replan ever RECOVERS after it leaves. No C++ changes (uses bridge accessors).
Run:  python _rand_timeline.py 5 [12 ...]
"""
import os, sys
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX"):
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
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); cull = float(LOOP.get("sense_cull_r", 40.0))
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0
    print(f"=== scene {seed:02d} ({sum(is_wall)} walls, {len(OBS)-sum(is_wall)} humans) ===")
    while t < R.T_MAX and not reached and nstep < 6000:
        did_replan = False; ok = None
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            for j, d in enumerate(DTS):
                if np.linalg.norm(d.eval(t) - p) <= cull: sando.add_traj(d, t)
            ret = sando.replan(last_rt, t); ok = bool(ret[0] if isinstance(ret, tuple) else ret)
            nr = t + RD; did_replan = True
        okg, ng = sando.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        if did_replan and (nstep % 5 == 0 or True):   # print every replan tick (~0.1s)
            hum = [(R.signed_box(p, DTS[j].eval(t), SZ[j]) - DR, DTS[j].eval(t)[1]) for j in range(len(OBS)) if not is_wall[j]]
            wal = [R.signed_box(p, DTS[j].eval(t), SZ[j]) - DR for j in range(len(OBS)) if is_wall[j]]
            hclr, hy = min(hum, key=lambda x: x[0]) if hum else (9.9, 0.0)
            wclr = min(wal) if wal else 9.9
            stat = sando.get_drone_status()
            print(f" t={t:5.2f} x={p[0]:5.1f} y={p[1]:+.2f} z={p[2]:.2f} |v|={np.linalg.norm(v):.2f} "
                  f"stat={stat} replan_ok={int(ok)} | nearHuman clr={hclr:+.2f}(y={hy:+.1f}) nearWall clr={wclr:+.2f}",
                  flush=True)
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    print(f" --> reached={reached} final x={p[0]:.1f} t={t:.1f}")


if __name__ == "__main__":
    for s in ([int(x) for x in sys.argv[1:]] or [5]):
        run(s)
