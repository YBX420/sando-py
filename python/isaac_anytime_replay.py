r"""在 Isaac Sim 里【回放】A+B 在 tunnel 内穿密集动态硬人群的运行(3D 渲染)。
读 media/_gif_path.csv / _gif_hum.csv / _gif_meta.csv(由 cpp/test/viz_tunnel_gif 生成),
spawn 无人机 + 16 行人 + tunnel 两道墙 + 目标,逐帧把它们摆到录制位置并 render。

⚠️ 这是【回放】录制好的 A+B 结果(轨迹/安全/<50ms 已在 C++ 验过),不是 Isaac 里现场跑求解器
   —— 现场 A+B 需把 C++ 求解器接进 C-ABI DLL(生产集成,待办)。回放给你看 3D 画面。

跑(你本机,需 Isaac Sim):
   E:\isssacsim\python.bat D:\Projects\sando_py\sando-py\python\isaac_anytime_replay.py
数据若缺:先在 cpp/ 下编译跑 viz_tunnel_gif 生成 media/_gif_*.csv。
"""
import os, sys, csv, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
MED = os.path.join(ROOT, "media")

# ---- load recorded run ----
def _csv(fn):
    return list(csv.DictReader(open(os.path.join(MED, fn))))

meta = _csv("_gif_meta.csv")[0]
W = float(meta["W"]); D_SAFE = float(meta["d_safe"]); GOALX = float(meta["goalx"])
path = _csv("_gif_path.csv")
hum_by_t = {}
for r in _csv("_gif_hum.csv"):
    hum_by_t.setdefault(round(float(r["t"]), 3), []).append((float(r["x"]), float(r["y"]), float(r["r"])))
NH = max(len(v) for v in hum_by_t.values())
DRONE_Z = 1.5   # 飞行高度(可视化)

# ---- Isaac (SimulationApp 必须最先) ----
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import VisualCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add(VisualCuboid(prim_path="/World/floor", name="floor",
    position=np.array([GOALX / 2, 0, 0]), scale=np.array([GOALX + 8, 12, 0.02]),
    color=np.array([0.22, 0.22, 0.26])))
import omni.usd
from pxr import UsdLux, Sdf
_st = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(_st, Sdf.Path("/World/Dome")).CreateIntensityAttr(1000.0)
UsdLux.DistantLight.Define(_st, Sdf.Path("/World/Sun")).CreateIntensityAttr(3000.0)

# tunnel 两道墙(y = ±W),长在 x、薄在 y、立在 z
for s, yy in [("hi", W), ("lo", -W)]:
    world.scene.add(VisualCuboid(prim_path=f"/World/tunnel_{s}", name=f"tunnel_{s}",
        position=np.array([GOALX / 2, yy, 1.25]), scale=np.array([GOALX + 2, 0.05, 2.5]),
        color=np.array([0.2, 0.45, 0.9])))

# 无人机
drone = world.scene.add(VisualCuboid(prim_path="/World/drone", name="drone",
    position=np.array([0, 0, DRONE_Z]), scale=np.array([0.4, 0.4, 0.4]), color=np.array([1.0, 0.9, 0.1])))
# 目标
world.scene.add(VisualCuboid(prim_path="/World/goal", name="goal",
    position=np.array([GOALX, 0, DRONE_Z]), scale=np.array([0.3, 0.3, 0.3]), color=np.array([0.1, 0.9, 0.2])))
# 16 行人(人形盒,站立,红色)
humans = [world.scene.add(VisualCuboid(prim_path=f"/World/hum_{i}", name=f"hum_{i}",
    position=np.array([0, 0, 0.9]), scale=np.array([0.55, 0.55, 1.7]), color=np.array([0.9, 0.2, 0.2])))
    for i in range(NH)]
# 面包屑
N_TRAIL = 400
trail = [world.scene.add(VisualCuboid(prim_path=f"/World/tr_{k}", name=f"tr_{k}",
    position=np.array([0, 0, -5]), scale=np.array([0.09, 0.09, 0.09]), color=np.array([0.1, 1.0, 0.3])))
    for k in range(N_TRAIL)]

world.reset()
print(f"[replay] {NH} humans, tunnel ±{W}, {len(path)} frames. 回放 A+B 穿密集人群...", flush=True)

ti = 0
while simulation_app.is_running():
    row = path[ti % len(path)]
    t = round(float(row["t"]), 3)
    dpos = np.array([float(row["x"]), float(row["y"]), DRONE_Z])
    drone.set_world_pose(position=dpos)
    hs = hum_by_t.get(t, [])
    for i, h in enumerate(humans):
        if i < len(hs):
            hx, hy, r = hs[i]
            h.set_world_pose(position=np.array([hx, hy, 0.9]))
        else:
            h.set_world_pose(position=np.array([0, 0, -5]))
    trail[ti % N_TRAIL].set_world_pose(position=dpos.copy())
    world.step(render=True)
    ti += 1
    if ti % len(path) == 0:   # 一轮放完,清面包屑重放
        for tr in trail:
            tr.set_world_pose(position=np.array([0, 0, -5]))
    time.sleep(0.03)

simulation_app.close()
