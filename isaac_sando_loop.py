"""Isaac Sim <-> 真 sando_py 管线(零 hack)。

驱动你自己的 SANDO 编排器,完全照 sando_node 的方式:
  每拍: update_state(当前位姿) -> add_traj(per-class DynTraj 障碍) -> replan()
        replan 内部跑你真管线: 全局 HGP heat-A*(绕开障碍) -> 局部 per-class MINCO -> append_to_plan
        get_next_goal() 从 plan 队列消费 setpoint 驱动无人机。
所有参数在 isaac_sando.yaml 里,改那个调参,本脚本不用动。

⚠️ 已知你代码里的两处不一致(在此如实标注,未偷偷绕):
  (1) DynTraj.bbox 约定冲突: voxel_map.read_map 当它是【半尺寸】,
      _obstacles_from_snapshot 当它是【全尺寸】(half=0.5*bbox)。同一个值两边差一倍。
      取舍: 这里填【全尺寸】-> MINCO 避障尺寸正确(安全优先), HGP 体素偏保守(更宽地绕, 安全)。
  (2) plan_local_trajectory_minco 调 plan_minco 时没传 opt_params,
      所以 YAML 里的 v_max/a_max 不会进 MINCO 解(OptParams 默认 vmax=3 在管)。要让 YAML 限速
      生效需在 planner 里把 par.v_max/a_max 透传进 plan_minco —— 这是你代码该补的一处。

跑法:  E:\isssacsim\python.bat D:\Projects\sando_py\sando-py\isaac_sando_loop.py
"""
import os
import sys
import time

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# ---- 先读配置(纯 python, 在 SimulationApp 之前 OK), 拿 headless ----
with open(os.path.join(REPO_ROOT, "isaac_sando.yaml"), "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)
PLN = CFG["planner"]; SCENE = CFG["scene"]; LOOP = CFG["loop"]

# ---- SimulationApp 必须最先创建, 且在任何 omni/isaacsim import 之前 ----
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": bool(LOOP.get("headless", False))})

# ---- Isaac 4.5 core API ----
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import VisualCuboid

# ---- 你的真管线(只经 SANDO, 不直调局部求解器)----
# 引擎选择: "cpp" = C++ 移植(sando_cpp_bridge -> sando_capi.dll, golden 验证过的同一套算法),
#          "py"  = 纯 Python sando_py。默认 cpp;可用环境变量 SANDO_ENGINE 或 YAML 顶层 engine: 覆盖。
SANDO_ENGINE = os.environ.get("SANDO_ENGINE", CFG.get("engine", "cpp")).lower()
if SANDO_ENGINE == "cpp":
    from sando_cpp_bridge import (Parameters, RobotState, DynTraj, SANDO,
                                  DroneStatus_GOAL_REACHED as GOAL_REACHED)
    print(f"[isaac] engine = C++ (sando_cpp_bridge)", flush=True)
else:
    from sando_py.types import Parameters, RobotState, DynTraj
    from sando_py.planner import SANDO
    try:
        from sando_py.types import DroneStatus
    except Exception:
        from sando_py.planner import DroneStatus
    GOAL_REACHED = int(DroneStatus.GOAL_REACHED)
    print("[isaac] engine = Python (sando_py)", flush=True)

# ===========================================================================
# Parameters: 从 YAML 逐字段 setattr(只认 Parameters 真有的字段)
# ===========================================================================
par = Parameters()
for k, v in PLN.items():
    if k == "sando_map_res":
        par.res = float(v); continue
    if hasattr(par, k):
        setattr(par, k, v)
    else:
        print(f"[isaac][warn] Parameters 无字段 '{k}', 跳过", flush=True)

START = np.array(SCENE["start"], dtype=float)
GOAL = np.array(SCENE["goal"], dtype=float)
if bool(getattr(par, "force_goal_z", False)):
    GOAL[2] = float(getattr(par, "default_goal_z", GOAL[2]))

# ===========================================================================
# Isaac 世界: 地板 + 灯 + 无人机 + 障碍盒
# ===========================================================================
world = World(stage_units_in_meters=1.0)
print("[isaac] World 已创建", flush=True)

_gx = float(GOAL[0])
world.scene.add(VisualCuboid(prim_path="/World/floor", name="floor",
    position=np.array([_gx / 2.0, 0.0, 0.0]),
    scale=np.array([abs(_gx) + 20.0, 16.0, 0.02]), color=np.array([0.25, 0.25, 0.28])))

import omni.usd
from pxr import UsdLux, Sdf
_stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(_stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1000.0)
UsdLux.DistantLight.Define(_stage, Sdf.Path("/World/SunLight")).CreateIntensityAttr(3000.0)
print("[isaac] 地板+灯已建", flush=True)

drone = world.scene.add(VisualCuboid(prim_path="/World/drone", name="drone",
    position=START.copy(), scale=np.array([0.35, 0.35, 0.35]), color=np.array([0.1, 0.4, 1.0])))

OBS = SCENE["obstacles"]


def make_dt(i, o):
    """从场景 spec 造 per-class DynTraj(静态: center; 运动: traj/vel 表达式)。"""
    dt = DynTraj()
    dt.id = (200 + i) if o.get("class", "wall") == "wall" else i   # 200<=id<300 -> wall/软, 否则 human/硬
    dt.mode = "Analytic"
    dt.bbox = np.array(o["size"], float)                           # 全尺寸(见文件头注 1)
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
obs_prims = []
for i, o in enumerate(OBS):
    p0 = DTS[i].eval(0.0); sz = np.array(o["size"], float)
    col = np.array([1.0, 0.25, 0.25]) if IS_HUMAN[i] else np.array([0.55, 0.55, 0.6])
    obs_prims.append(world.scene.add(VisualCuboid(prim_path=f"/World/obs_{i}", name=f"obs_{i}",
        position=p0.copy(), scale=sz.copy(), color=col)))

# 面包屑轨迹
N_TRAIL = 500
trail = [world.scene.add(VisualCuboid(prim_path=f"/World/trail_{k}", name=f"trail_{k}",
         position=np.array([0.0, 0.0, -5.0]), scale=np.array([0.08, 0.08, 0.08]),
         color=np.array([0.1, 1.0, 0.2]))) for k in range(N_TRAIL)]

print(f"[isaac] {len(OBS)} 障碍盒已加入, reset()...", flush=True)
world.reset()
print("[isaac] world.reset() 完成", flush=True)

# ===========================================================================
# 建你的 planner + 启动序列(照 blueprint / sando_node)
# ===========================================================================
sando = SANDO(par)

# 1) 先喂初始状态 -> state_initialized=True
_st = RobotState(); _st.pos = START.copy(); _st.vel = np.zeros(3)
sando.update_state(_st)
# 2) 初始化占据图一次 -> map_initialized=True(否则 check_ready_to_replan 在首次 replan 前就 False)
sando.update_occupancy_map_ptr(np.zeros((0, 3)))
# 3) 设终点 -> terminal_goal_initialized + (skip_initial_yawing=true) 直接 TRAVELING
_G = RobotState(); _G.pos = GOAL.copy()
sando.set_terminal_goal(_G)
print(f"[isaac] 启动完毕. start={START} goal={GOAL} status={sando.get_drone_status()}", flush=True)


def push_obstacles(t, p_d):
    """把 SENSE_R 内的障碍作为 per-class DynTraj 推给 planner(add_traj 按 id 去重/更新)。
       DynTraj.traj_* 是预测层 —— 将来 NN 预测就换这里, planner/证书不动。"""
    cull = float(LOOP.get("sense_cull_r", 40.0))
    for dt in DTS:
        if np.linalg.norm(dt.eval(t) - p_d) <= cull:
            sando.add_traj(dt, t)


def signed_box_dist(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    outside = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    if np.any(outside > 0.0):
        return float(np.linalg.norm(outside))
    return float(np.max(np.maximum(lo - p, p - hi)))   # <=0 内部


def clearance_all(p, t):
    cmin = np.inf
    for i, o in enumerate(OBS):
        cmin = min(cmin, signed_box_dist(p, DTS[i].eval(t), np.array(o["size"], float)))
    return cmin


# ===========================================================================
# 闭环: replan(你真管线) + get_next_goal 消费
# ===========================================================================
DT = float(par.dc)                       # 控制 tick = setpoint 间隔
REPLAN_DT = float(LOOP.get("replan_dt", 0.1))
T_MAX = float(LOOP.get("t_max", 60.0))

p_d = START.copy(); v_d = np.zeros(3); a_d = np.zeros(3)
t = 0.0; last_rt = 0.0; next_replan = 0.0
n_invalid = 0; reached = False; collided = False
min_clear = np.inf
flown = []; trail_idx = 0; TRAIL_EVERY = 3

print("[isaac] start loop. 列: t | replan_ok | status | globalN | global|y|max | drone_y | solve_ms", flush=True)

while simulation_app.is_running() and t < T_MAX and not reached:
    if t >= next_replan - 1e-9:
        # 1) 刷新当前位姿
        st = RobotState(); st.pos = p_d.copy(); st.vel = v_d.copy(); st.accel = a_d.copy()
        sando.update_state(st)
        # 2) 推 per-class 障碍
        push_obstacles(t, p_d)
        # 3) 跑一拍你真管线(HGP heat-A* -> per-class MINCO -> append_to_plan)
        t0 = time.perf_counter()
        ret = sando.replan(last_rt, t)
        last_rt = time.perf_counter() - t0
        ok_plan = bool(ret[0] if isinstance(ret, tuple) else ret)
        if not ok_plan:
            n_invalid += 1
        gp = sando.get_global_path()
        gmaxy = max((abs(float(p[1])) for p in gp), default=0.0)
        print(f"  t={t:5.2f} | ok={int(ok_plan)} | status={sando.get_drone_status()} | "
              f"gN={len(gp):2d} | g|y|max={gmaxy:5.2f} | drone_y={p_d[1]:+.2f} | "
              f"ms={last_rt*1000:5.0f} | clr={clearance_all(p_d, t):+.2f}", flush=True)
        next_replan = t + REPLAN_DT

    # 消费 plan 队列推进无人机(你真管线的输出, 不直接 eval mj)
    ok_g, ng = sando.get_next_goal()
    if ok_g:
        p_d = np.asarray(ng.pos, dtype=float)
        v_d = np.asarray(ng.vel, dtype=float)
        a_d = np.asarray(ng.accel, dtype=float)

    drone.set_world_pose(position=p_d)
    for i in range(len(OBS)):                       # 障碍按 t 运动(可视化跟着动)
        obs_prims[i].set_world_pose(position=DTS[i].eval(t))
    c = clearance_all(p_d, t)
    min_clear = min(min_clear, c)
    if c < 0.0:
        collided = True
    if (len(flown) % TRAIL_EVERY) == 0:
        trail[trail_idx % N_TRAIL].set_world_pose(position=p_d.copy()); trail_idx += 1
    flown.append((t, float(p_d[0]), float(p_d[1]), float(p_d[2]), c, int(sando.get_drone_status())))

    world.step(render=True)
    t += DT
    if sando.get_drone_status() == GOAL_REACHED and float(np.linalg.norm(p_d - GOAL)) < float(par.goal_radius):
        reached = True

# ---- 落盘 ----
csv_path = os.path.join(REPO_ROOT, "isaac_flown.csv")
with open(csv_path, "w") as f:
    f.write("t,x,y,z,exec_clr,status\n")
    for r in flown:
        f.write(",".join(str(v) for v in r) + "\n")
print(f"[isaac] 轨迹存 {csv_path} ({len(flown)} 点)", flush=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array([(r[1], r[2]) for r in flown])
    fig, ax = plt.subplots(figsize=(16, 4))
    for o in OBS:
        c = np.array(o["center"], float); sz = np.array(o["size"], float)
        ax.add_patch(plt.Rectangle((c[0] - 0.5 * sz[0], c[1] - 0.5 * sz[1]), sz[0], sz[1], color="gray"))
    ax.plot(arr[:, 0], arr[:, 1], "-", color="green", lw=2, label="flown")
    ax.scatter([START[0], GOAL[0]], [START[1], GOAL[1]], c="blue", marker="*", s=140)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
    ax.set_title(f"real pipeline (HGP->MINCO)  reached={reached} collided={collided} min_clr={min_clear:.2f}")
    fig.savefig(os.path.join(REPO_ROOT, "isaac_flown.png"), dpi=120, bbox_inches="tight")
    print("[isaac] 俯视图存 isaac_flown.png", flush=True)
except Exception as e:
    print(f"[isaac] (跳过 PNG: {e})", flush=True)

print("\n===========================================================", flush=True)
print(f"[isaac] done. reached={reached} collided={collided} t={t:.2f}s 规划失败={n_invalid}", flush=True)
print(f"[isaac] 最小净空 = {min_clear:.3f} m", flush=True)
print(f"[isaac] 判定: {'COLLIDED 撞了' if collided else ('OK 绕开了' if min_clear >= 0.3 else 'BREACH 擦了')}", flush=True)
print("===========================================================\n", flush=True)

simulation_app.close()
