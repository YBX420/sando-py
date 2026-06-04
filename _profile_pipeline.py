"""分项 profile: 把每拍 replan 拆成 全局(HGP 建图+heat-A*) vs 局部(MINCO),
看瓶颈在哪。再 cProfile 出最热的函数。跑: python _profile_pipeline.py"""
import os, sys, time, cProfile, pstats, io
import numpy as np, yaml

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
CFG = yaml.safe_load(open(os.path.join(ROOT, "isaac_sando.yaml"), encoding="utf-8"))
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]
from sando_py.types import Parameters, RobotState, DynTraj
from sando_py.planner import SANDO

par = Parameters()
for k, v in PLN.items():
    if k == "sando_map_res": par.res = float(v); continue
    if hasattr(par, k): setattr(par, k, v)
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)
if getattr(par, "force_goal_z", False): GOAL[2] = par.default_goal_z
OBS = SCENE["obstacles"]

sando = SANDO(par)
_st = RobotState(); _st.pos = START.copy(); sando.update_state(_st)
sando.update_occupancy_map_ptr(np.zeros((0, 3)))
_G = RobotState(); _G.pos = GOAL.copy(); sando.set_terminal_goal(_G)

def push(t, p):
    for i, o in enumerate(OBS):
        dt = DynTraj(); dt.id = (200 + i) if o.get("class","wall")=="wall" else i
        c = np.array(o["center"], float); dt.mode="Analytic"
        dt.traj_x=f"{c[0]}"; dt.traj_y=f"{c[1]}"; dt.traj_z=f"{c[2]}"
        dt.traj_vx="0.0"; dt.traj_vy="0.0"; dt.traj_vz="0.0"; dt.bbox=np.array(o["size"],float)
        sando.add_traj(dt, t)

# 把无人机摆到障碍区中间(x~20)再测,避免起点段障碍稀疏不真实
p_d = np.array([20.0, 1.5, 2.0]); t = 5.0
st = RobotState(); st.pos = p_d.copy(); sando.update_state(st); push(t, p_d)

# warm up
for _ in range(2):
    sando.generate_global_path(t, 0.0)

N = 20
tg = []; tl = []
for _ in range(N):
    st = RobotState(); st.pos = p_d.copy(); sando.update_state(st); push(t, p_d)
    a = time.perf_counter(); okg, gp = sando.generate_global_path(t, 0.0); b = time.perf_counter()
    okl = sando.plan_local_trajectory_minco(gp, 0.0); c = time.perf_counter()
    tg.append((b - a) * 1000); tl.append((c - b) * 1000)

print(f"\n=== 分项耗时 (N={N}, 障碍区中段) ===")
print(f"全局 generate_global_path (HGP 建图+heat-A*): mean={np.mean(tg):6.1f} ms  median={np.median(tg):6.1f}")
print(f"局部 plan_local_trajectory_minco (MINCO):     mean={np.mean(tl):6.1f} ms  median={np.median(tl):6.1f}")
print(f"合计每拍: {np.mean(tg)+np.mean(tl):6.1f} ms  ->  {1000/(np.mean(tg)+np.mean(tl)):.1f} Hz")

# cProfile 全局段(瓶颈大头),找最热函数
print("\n=== cProfile generate_global_path 最热 18 个 (cumtime) ===")
pr = cProfile.Profile(); pr.enable()
for _ in range(10):
    sando.generate_global_path(t, 0.0)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
print("\n".join(s.getvalue().splitlines()[:30]))
