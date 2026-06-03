"""无头验证: 用真 sando_py 管线驱动(去掉 Isaac),确认能绕开障碍、API 正确。
等价于 isaac_sando_loop.py 但不渲染。跑: python _isaac_pipeline_headless_test.py"""
import os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CFG = yaml.safe_load(open(os.path.join(ROOT, "isaac_sando.yaml"), encoding="utf-8"))
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]

from sando_py.types import Parameters, RobotState, DynTraj
from sando_py.planner import SANDO
try:
    from sando_py.types import DroneStatus
except Exception:
    from sando_py.planner import DroneStatus
GOAL_REACHED = int(DroneStatus.GOAL_REACHED)

par = Parameters()
skipped = []
for k, v in PLN.items():
    if k == "sando_map_res":
        par.res = float(v); continue
    if hasattr(par, k):
        setattr(par, k, v)
    else:
        skipped.append(k)
print("skipped yaml keys (not in Parameters):", skipped)

START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)
if bool(getattr(par, "force_goal_z", False)):
    GOAL[2] = float(getattr(par, "default_goal_z", GOAL[2]))
OBS = SCENE["obstacles"]

sando = SANDO(par)
_st = RobotState(); _st.pos = START.copy(); sando.update_state(_st)
sando.update_occupancy_map_ptr(np.zeros((0, 3)))
_G = RobotState(); _G.pos = GOAL.copy(); sando.set_terminal_goal(_G)
print("startup ok. status=", sando.get_drone_status(), "start", START, "goal", GOAL)


def push(t, p_d):
    for i, o in enumerate(OBS):
        c = np.array(o["center"], float)
        if np.linalg.norm(c - p_d) > float(LOOP.get("sense_cull_r", 40.0)):
            continue
        dt = DynTraj(); cls = o.get("class", "wall")
        dt.id = (200 + i) if cls == "wall" else i
        dt.mode = "Analytic"
        dt.traj_x = f"{float(c[0])}"; dt.traj_y = f"{float(c[1])}"; dt.traj_z = f"{float(c[2])}"
        dt.traj_vx = "0.0"; dt.traj_vy = "0.0"; dt.traj_vz = "0.0"
        dt.bbox = np.array(o["size"], float)
        sando.add_traj(dt, t)


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def clr(p):
    return min(signed_box(p, np.array(o["center"], float), np.array(o["size"], float)) for o in OBS)


DT = float(par.dc); REPLAN_DT = float(LOOP.get("replan_dt", 0.1)); T_MAX = float(LOOP.get("t_max", 60.0))
p_d = START.copy(); v_d = np.zeros(3); a_d = np.zeros(3)
t = 0.0; last_rt = 0.0; nr = 0.0; ninv = 0; reached = False
min_clr = np.inf; maxy = 0.0; nstep = 0
while t < T_MAX and not reached and nstep < 6000:
    if t >= nr - 1e-9:
        st = RobotState(); st.pos = p_d.copy(); st.vel = v_d.copy(); st.accel = a_d.copy()
        sando.update_state(st); push(t, p_d)
        t0 = time.perf_counter(); ret = sando.replan(last_rt, t); last_rt = time.perf_counter() - t0
        ok = bool(ret[0] if isinstance(ret, tuple) else ret)
        if not ok: ninv += 1
        gp = sando.get_global_path(); gy = max((abs(float(p[1])) for p in gp), default=0.0)
        if t < 6.0 or nstep % 50 == 0:
            print(f"t={t:5.2f} ok={int(ok)} st={sando.get_drone_status()} gN={len(gp):2d} "
                  f"g|y|={gy:5.2f} x={p_d[0]:6.2f} y={p_d[1]:+.2f} clr={clr(p_d):+.2f} ms={last_rt*1000:4.0f}")
        nr = t + REPLAN_DT
    okg, ng = sando.get_next_goal()
    if okg:
        p_d = np.asarray(ng.pos, float); v_d = np.asarray(ng.vel, float); a_d = np.asarray(ng.accel, float)
    min_clr = min(min_clr, clr(p_d)); maxy = max(maxy, abs(p_d[1]))
    t += DT; nstep += 1
    if sando.get_drone_status() == GOAL_REACHED and float(np.linalg.norm(p_d - GOAL)) < float(par.goal_radius):
        reached = True

print("\n=== RESULT ===")
print(f"reached={reached} final_x={p_d[0]:.2f} max|y|={maxy:.2f} min_clr={min_clr:.3f} replan_fail={ninv}")
print("VERDICT:", "ROUTES AROUND (y left 0)" if maxy > 0.5 else "STILL STRAIGHT (y~0) -- problem",
      "| clearance OK" if min_clr >= 0.2 else "| BREACH")
