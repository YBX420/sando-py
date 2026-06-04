"""无头验证(含运动障碍): 真 sando_py 管线驱动, 障碍按 DynTraj 解析式运动。
跑: python _isaac_pipeline_headless_test.py"""
import os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
CFG = yaml.safe_load(open(os.path.join(ROOT, "isaac_sando.yaml"), encoding="utf-8"))
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]
SANDO_ENGINE = os.environ.get("SANDO_ENGINE", CFG.get("engine", "cpp")).lower()
if SANDO_ENGINE == "cpp":
    from sando_cpp_bridge import (Parameters, RobotState, DynTraj, SANDO,
                                  DroneStatus_GOAL_REACHED as GOAL_REACHED)
    print("[headless] engine = C++ (sando_cpp_bridge)")
else:
    from sando_py.types import Parameters, RobotState, DynTraj
    from sando_py.planner import SANDO
    try:
        from sando_py.types import DroneStatus
    except Exception:
        from sando_py.planner import DroneStatus
    GOAL_REACHED = int(DroneStatus.GOAL_REACHED)
    print("[headless] engine = Python (sando_py)")

par = Parameters()
for k, v in PLN.items():
    if k == "sando_map_res": par.res = float(v); continue
    if hasattr(par, k): setattr(par, k, v)
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)
if bool(getattr(par, "force_goal_z", False)): GOAL[2] = float(par.default_goal_z)
OBS = SCENE["obstacles"]


def make_dt(i, o):
    dt = DynTraj()
    dt.id = (200 + i) if o.get("class", "wall") == "wall" else i
    dt.mode = "Analytic"
    dt.bbox = np.array(o["size"], float)
    if "traj" in o:
        dt.traj_x, dt.traj_y, dt.traj_z = [str(e) for e in o["traj"]]
        if "vel" in o:
            dt.traj_vx, dt.traj_vy, dt.traj_vz = [str(e) for e in o["vel"]]
    else:
        c = o["center"]
        dt.traj_x, dt.traj_y, dt.traj_z = f"{c[0]}", f"{c[1]}", f"{c[2]}"
        dt.traj_vx = dt.traj_vy = dt.traj_vz = "0.0"
    dt.compile_analytic()
    return dt


DTS = [make_dt(i, o) for i, o in enumerate(OBS)]
IS_HUMAN = [o.get("class", "wall") != "wall" for o in OBS]

sando = SANDO(par)
_st = RobotState(); _st.pos = START.copy(); sando.update_state(_st)
sando.update_occupancy_map_ptr(np.zeros((0, 3)))
_G = RobotState(); _G.pos = GOAL.copy(); sando.set_terminal_goal(_G)
print("startup ok. status=", sando.get_drone_status(), "start", START, "goal", GOAL,
      "| obstacles:", len(OBS), "(", sum(IS_HUMAN), "human movers )")


def push(t, p_d):
    cull = float(LOOP.get("sense_cull_r", 40.0))
    for dt in DTS:
        if np.linalg.norm(dt.eval(t) - p_d) <= cull:
            sando.add_traj(dt, t)


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def clearances(p, t):
    """返回 (对所有障碍最小净空, 对运动行人最小净空)"""
    call = chum = np.inf
    for i, o in enumerate(OBS):
        c = DTS[i].eval(t); sz = np.array(o["size"], float)
        d = signed_box(p, c, sz)
        call = min(call, d)
        if IS_HUMAN[i]:
            chum = min(chum, d)
    return call, chum


DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
p_d = START.copy(); v_d = np.zeros(3); a_d = np.zeros(3)
t = 0.0; last_rt = 0.0; nr = 0.0; ninv = 0; reached = False
min_all = min_hum = np.inf; maxy = 0.0; collided = False; nstep = 0
while t < T_MAX and not reached and nstep < 8000:
    if t >= nr - 1e-9:
        st = RobotState(); st.pos = p_d.copy(); st.vel = v_d.copy(); st.accel = a_d.copy()
        sando.update_state(st); push(t, p_d)
        a = time.perf_counter(); ret = sando.replan(last_rt, t); last_rt = time.perf_counter() - a
        ok = bool(ret[0] if isinstance(ret, tuple) else ret)
        if not ok: ninv += 1
        gp = sando.get_global_path(); gy = max((abs(float(q[1])) for q in gp), default=0.0)
        ca, ch = clearances(p_d, t)
        if t < 4.0 or nstep % 40 == 0:
            print(f"t={t:5.2f} ok={int(ok)} st={sando.get_drone_status()} gN={len(gp):2d} "
                  f"g|y|={gy:4.1f} x={p_d[0]:5.1f} y={p_d[1]:+5.2f} clr_all={ca:+5.2f} clr_hum={ch:+5.2f} ms={last_rt*1000:4.0f}")
        nr = t + RD
    okg, ng = sando.get_next_goal()
    if okg:
        p_d = np.asarray(ng.pos, float); v_d = np.asarray(ng.vel, float); a_d = np.asarray(ng.accel, float)
    ca, ch = clearances(p_d, t)
    min_all = min(min_all, ca); min_hum = min(min_hum, ch); maxy = max(maxy, abs(p_d[1]))
    if ca < 0.0: collided = True
    t += DT; nstep += 1
    if sando.get_drone_status() == GOAL_REACHED and float(np.linalg.norm(p_d - GOAL)) < float(par.goal_radius):
        reached = True

print("\n=== RESULT (complex moving scene) ===")
print(f"reached={reached} final_x={p_d[0]:.1f} max|y|={maxy:.2f}")
print(f"min_clear_all={min_all:.3f}  min_clear_human={min_hum:.3f}  collided={collided}  replan_fail={ninv}")
print("VERDICT:",
      ("REACHED" if reached else "DID NOT REACH"),
      "| routes" if maxy > 0.5 else "| straight?",
      "| HUMAN-SAFE" if min_hum >= 0.0 else "| HUMAN COLLISION",
      "| wall-clear" if min_all >= -0.01 else "| hit-wall")
