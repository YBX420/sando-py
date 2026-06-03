"""cProfile 真实闭环(从起点跑 ~30 replan),按 self-time 看时间真正花哪 —— 含 MINCO 内部。"""
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
        dt = DynTraj(); dt.id=(200+i) if o.get("class","wall")=="wall" else i
        c=np.array(o["center"],float); dt.mode="Analytic"
        dt.traj_x=f"{c[0]}";dt.traj_y=f"{c[1]}";dt.traj_z=f"{c[2]}"
        dt.traj_vx="0.0";dt.traj_vy="0.0";dt.traj_vz="0.0";dt.bbox=np.array(o["size"],float)
        sando.add_traj(dt,t)
DT=float(par.dc); RD=float(LOOP["replan_dt"])
p_d=START.copy(); v_d=np.zeros(3); a_d=np.zeros(3); t=0.0; nr=0.0; rt=0.0
per=[]
def step_once():
    global p_d,v_d,a_d,t,nr,rt
    if t>=nr-1e-9:
        st=RobotState();st.pos=p_d.copy();st.vel=v_d.copy();st.accel=a_d.copy()
        sando.update_state(st); push(t,p_d)
        a=time.perf_counter(); sando.replan(rt,t); rt=time.perf_counter()-a; per.append(rt*1000)
        nr=t+RD
    okg,ng=sando.get_next_goal()
    if okg: p_d=np.asarray(ng.pos,float);v_d=np.asarray(ng.vel,float);a_d=np.asarray(ng.accel,float)
    t+=DT
# warm
for _ in range(60): step_once()
per.clear()
pr=cProfile.Profile(); pr.enable()
for _ in range(700): step_once()
pr.disable()
print(f"\n=== 真实 replan 每拍: mean={np.mean(per):.1f}ms median={np.median(per):.1f} p90={np.percentile(per,90):.1f} (N={len(per)}) ===")
print("=== 按 self-time(tottime)最热 20 ===")
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats("tottime").print_stats(20)
print("\n".join(s.getvalue().splitlines()[:32]))
