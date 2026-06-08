"""Measure real replan() solve time on the RGBD-FOV map (with physics). Confirms whether the visual
stutter is compute (it is NOT, in sim) -- and what the real-time solve budget would be on hardware."""
import os, sys, time
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX", "RETIME", "GOALX", "VM", "CAMR", "CAMFOV"):
    os.environ.pop(k, None)
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import _rand_scenes as R, _speed_bench as B
from sando_cpp_bridge import RobotState, SANDO

ms_all = []
for seed in (int(x) for x in (sys.argv[1:] or ["1", "7", "13", "19"])):
    par, _, _, _, LOOP = B.build()
    par.v_max = R.VMAX; par.use_st_graph = False; par.use_spacetime_corridor = False; par.force_goal_z = False
    START = np.array([0., 0, 3.]); GOAL = np.array([R.GOAL_X, 0, 3.]); OBS = R.make_scene(seed)
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]; SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par); st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); AMAX = float(par.a_max)
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.; nr = 0.; reached = False; nstep = 0; ms = []
    while t < R.T_MAX and not reached and nstep < 6000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            hd = v / max(np.linalg.norm(v), 1e-6) if np.linalg.norm(v) > 0.5 else np.array([1., 0, 0])
            for j, d in enumerate(DTS):
                if R.visible(p, hd, d.eval(t), SZ[j]): sando.add_traj(d, t)
            a0 = time.perf_counter(); sando.replan(0., t); ms.append((time.perf_counter() - a0) * 1000.0); nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg: p, v, a = R.step_drone(p, v, ng, AMAX, R.VMAX, DT)
        t += DT; nstep += 1
        if np.linalg.norm(p - GOAL) < float(par.goal_radius): reached = True
    m = np.array(ms); ms_all += ms
    print(f"seed {seed}: replans={len(ms)} ms mean={m.mean():.0f} p99={np.percentile(m,99):.0f} max={m.max():.0f}")
m = np.array(ms_all)
print(f"=== ALL: replans={len(m)} ms mean={m.mean():.1f} p50={np.percentile(m,50):.0f} p99={np.percentile(m,99):.0f} "
      f"max={m.max():.0f} | replan_dt budget={int(1000*float(B.build()[4]['replan_dt']))}ms ===")
