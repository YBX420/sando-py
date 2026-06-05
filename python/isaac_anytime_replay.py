r"""在 Isaac Sim 里【回放】A+B 在 tunnel 内穿密集动态硬人群的运行(真实 3D)。
读 media/_gif_path.csv (t,x,y,z,clr,ms) / _gif_hum.csv / _gif_meta.csv(由 cpp/test/viz_tunnel_gif 生成)。

关键:这是【真 3D】避障 —— tunnel 只约束 y(|y|≤W),z 自由,所以在 y-lane 挤不过去的最密处,
无人机会【抬高 z 飞过】最近的人(z 最高到 ~3.2,人在 z=2.0),全程 3D 净空 0.927≥d_safe,真实无碰撞。
之前「回放撞了」是因为旧回放把 z 拍平到同一平面、丢了高度差;这版按真实 z 渲染,会看到它真的飞高越过人群。

⚠️ 这是【回放】录制好的 A+B 结果(轨迹/3D 安全/<50ms 已在 C++ 验过),不是 Isaac 里现场跑求解器
   —— 现场 A+B 需把 C++ 求解器接进 C-ABI DLL(生产集成,待办)。

跑(你本机,需 Isaac Sim):
   E:\isssacsim\python.bat D:\Projects\sando_py\sando-py\python\isaac_anytime_replay.py
数据若缺:在 cpp/ 下 `c++ -O2 -std=c++17 -I include -I third_party -I third_party/eigen test/viz_tunnel_gif.cpp -o build/viz_tunnel_gif.exe` 再从 cpp/ 跑 build/viz_tunnel_gif.exe。
"""
import os, sys, csv, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
MED = os.path.join(ROOT, "media")
HUMAN_Z = 2.0   # 规划里行人(障碍球)所在平面;无人机巡航也在 ~2.0,挤时抬到 ~3.2 飞过

def _csv(fn):
    return list(csv.DictReader(open(os.path.join(MED, fn))))

meta = _csv("_gif_meta.csv")[0]
W = float(meta["W"]); D_SAFE = float(meta["d_safe"]); GOALX = float(meta["goalx"])
path = _csv("_gif_path.csv")
hum_by_t = {}
for r in _csv("_gif_hum.csv"):
    hum_by_t.setdefault(r["t"], []).append((float(r["x"]), float(r["y"]), float(r["r"])))
t0 = path[0]["t"]
R_HUM = [h[2] for h in hum_by_t[t0]]
NH = len(R_HUM)

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

# tunnel 两道墙(y = ±W),长在 x、薄在 y、高在 z(只约束 y,所以无人机能从顶上 z 飞越)
for s, yy in [("hi", W), ("lo", -W)]:
    world.scene.add(VisualCuboid(prim_path=f"/World/tunnel_{s}", name=f"tunnel_{s}",
        position=np.array([GOALX / 2, yy, HUMAN_Z]), scale=np.array([GOALX + 2, 0.05, 3.0]),
        color=np.array([0.2, 0.45, 0.9])))

# 无人机(真实 3D 位置,会看到它在最挤处爬升)
drone = world.scene.add(VisualCuboid(prim_path="/World/drone", name="drone",
    position=np.array([0, 0, HUMAN_Z]), scale=np.array([0.4, 0.4, 0.4]), color=np.array([1.0, 0.9, 0.1])))
# 目标
world.scene.add(VisualCuboid(prim_path="/World/goal", name="goal",
    position=np.array([GOALX, 0, HUMAN_Z]), scale=np.array([0.3, 0.3, 0.3]), color=np.array([0.1, 0.9, 0.2])))
# 16 行人:真实障碍尺寸(直径 2r)的红盒,在飞行平面 z=2.0(无人机从顶上飞越时高度差看得见)
humans = [world.scene.add(VisualCuboid(prim_path=f"/World/hum_{i}", name=f"hum_{i}",
    position=np.array([0, 0, HUMAN_Z]),
    scale=np.array([2 * R_HUM[i], 2 * R_HUM[i], 2 * R_HUM[i]]), color=np.array([0.9, 0.2, 0.2])))
    for i in range(NH)]
# 面包屑(真实 3D,显示爬升轨迹)
N_TRAIL = 400
trail = [world.scene.add(VisualCuboid(prim_path=f"/World/tr_{k}", name=f"tr_{k}",
    position=np.array([0, 0, -5]), scale=np.array([0.09, 0.09, 0.09]), color=np.array([0.1, 1.0, 0.3])))
    for k in range(N_TRAIL)]

world.reset()
print(f"[replay] {NH} humans, tunnel ±{W}, {len(path)} frames. 真 3D 回放:无人机在最挤处抬高飞越人群...", flush=True)

ti = 0
while simulation_app.is_running():
    row = path[ti % len(path)]
    t = row["t"]
    dpos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])  # 真实 z
    drone.set_world_pose(position=dpos)
    hs = hum_by_t.get(t, [])
    for i, h in enumerate(humans):
        if i < len(hs):
            hx, hy, r = hs[i]
            h.set_world_pose(position=np.array([hx, hy, HUMAN_Z]))
        else:
            h.set_world_pose(position=np.array([0, 0, -5]))
    trail[ti % N_TRAIL].set_world_pose(position=dpos.copy())
    world.step(render=True)
    ti += 1
    if ti % len(path) == 0:
        for tr in trail:
            tr.set_world_pose(position=np.array([0, 0, -5]))
    time.sleep(0.03)

simulation_app.close()
