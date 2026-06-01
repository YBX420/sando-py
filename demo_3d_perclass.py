"""3D visualization of the MINCO per-class planner.

中文说明(这个文件是干嘛的):
  MINCO per-class 规划器的 3D 可视化脚本(headless 出 PNG)。画两张图:
  (a) 纯空间 3D:z 会变化的真 3D 场景,MINCO 轨迹 vs A* 向导,人(HARD)画成实心球+半透明
      安全壳(d_safe),墙(SOFT)画成盒子,并在起作用的硬约束控制点上画 ALM 受力箭头(影子价格)。
  (b) 空间-时间 (x,y,t):把时间当成第三个轴。一个「横穿」的人在这个空间里是一条斜着的管子,
      无人机轨迹(按自己的时间抬升)绕开这条管子——这就是「在正确的时刻躲」的可视化故事。
      (b) 还是闭环的:每个 tick 从当前状态重规划、提交一小段、人继续移动,画的是真实执行出来的路径。
  不是 ROS 节点,直接调 plan_minco() 画图。术语同 stage3:HARD=硬约束必须躲、SOFT=软约束可蹭、
  ALM force/shadow price=约束的「顶手程度」。

Two views (headless -> PNG with set camera angles):
  (a) SPATIAL 3D: a true 3D scene (z varies) -- MINCO trajectory vs A* guide,
      the human (HARD) as a body + translucent d_safe shell, the wall (SOFT) as
      a box, and the ALM "force" arrows (lambda * normal, the shadow price) at
      the active hard control points.
  (b) SPACE-TIME (x, y, t): a CROSSING human is a slanted tube in space-time;
      the drone trajectory (lifted by its own time) routes around that tube --
      the Stage 4 "avoid at the right moment" story, made visible.
"""
import os
import sys

import matplotlib
# 中文:headless 后端,只出文件不弹窗。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 中文:不装包也能 import 本地模块。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402

# 中文:detour(绕行多起点搜索)关掉,本 demo 只看纯优化的避让。
ND = DetourConfig(enabled=False)


# 中文:给定盒子的对角两点 lo/hi,返回 6 个面的顶点列表,用来画 3D 长方体(墙)。
def box_faces(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
         [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [v[idx] for idx in f]


# 中文:生成一个球面的 x/y/z 网格(圆心 c、半径 r),用来画人的实心球和半透明安全壳。
def sphere_xyz(c, r, n=22):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


fig = plt.figure(figsize=(16, 7))

# ======================= (a) SPATIAL 3D per-class =======================
# 中文:第一张图——纯空间 3D。起终点 z 不同(1.2->1.8)再加场景本身就有高度差,所以是真 3D;
#       人在下方顶路(硬约束起作用),软墙在上方封顶,轨迹被夹出一条 3D 弧线。
axA = fig.add_subplot(1, 2, 1, projection="3d")
# proven binding channel (human below + soft wall capping the top) with a mild
# z tilt so the scene is genuinely 3D while the hard constraint still binds.
start = np.array([0.0, 0.0, 1.2]); goal = np.array([8.0, 0.0, 1.8])
astar = np.linspace(start, goal, 9)
human = SphereObstacle([4.0, -0.8, 1.5], 0.4, class_name="human")
wall = AABBObstacle(lo=[3.0, 0.6, 0.0], hi=[5.0, 3.0, 3.0], class_name="wall")
cfg = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall", "soft", 0.4, 1.0e1)}
sp, info = plan_minco(astar, [human, wall], cfg, opt_params=OptParams(), detour_cfg=ND)
ts = np.linspace(sp.t_start, sp.t_end, 300)
P = sp.eval(ts)

axA.plot(astar[:, 0], astar[:, 1], astar[:, 2], "--", color="0.6", lw=1.5, label="A* guide")
axA.plot(P[:, 0], P[:, 1], P[:, 2], "-", color="C0", lw=2.8, label="MINCO trajectory")
# human body + translucent d_safe shell
hx, hy, hz = sphere_xyz(human.centre0, human.radius)
axA.plot_surface(hx, hy, hz, color="C3", alpha=0.95, linewidth=0)
sx, sy, sz = sphere_xyz(human.centre0, human.radius + cfg["human"].d_safe)
axA.plot_surface(sx, sy, sz, color="C3", alpha=0.12, linewidth=0)
# wall box (soft)
axA.add_collection3d(Poly3DCollection(box_faces(wall.lo, wall.hi), facecolor="0.4",
                                      alpha=0.28, edgecolor="0.3"))
# ALM force arrows at active hard control points (shadow price)
# 中文:同 stage3——在起作用的硬约束控制点画受力箭头,箭头长短由 lambda(影子价格)编码。
certs = [c for c in info["hard_certificates"] if c["active"]]
lam_max = max((c["lambda"] for c in certs), default=1.0) or 1.0
for c in certs:
    Pk = c["P"]; f = c["force"]; nf = np.linalg.norm(f)
    if nf < 1e-9:
        continue
    u = f / nf; L = 0.4 + 1.0 * (c["lambda"] / lam_max)
    axA.quiver(Pk[0], Pk[1], Pk[2], u[0], u[1], u[2], length=L, color="C1", lw=2)
axA.plot([], [], color="C1", lw=2, label=r"hard force $\lambda\hat n$ (shadow price)")
axA.scatter(*start, color="k", s=30); axA.scatter(*goal, color="g", s=30)
axA.set_xlabel("x [m]"); axA.set_ylabel("y [m]"); axA.set_zlabel("z [m]")
axA.set_title(f"(a) spatial 3D per-class  (human=HARD, wall=soft)\n"
              f"valid={info['trajectory_valid']}, min human clr="
              f"{info['continuous_min_clearance']:.2f} m")
axA.legend(loc="upper left", fontsize=8); axA.view_init(elev=22, azim=-60)

# ======================= (b) SPACE-TIME (x, y, t): CLOSED-LOOP =======================
# Receding-horizon rollout (like the realtime test): replan each tick from the
# current state, commit dt, the human MOVES. We plot the EXECUTED path in
# space-time vs the human's moving d_safe tube -> reaction at the right moment.
# 中文:第二张图——空间-时间闭环(滚动时域)。模拟真实运行:每个 tick 从无人机「当前状态」
#       重规划一次,只执行(提交)前面一小段时间 DT,然后人按速度往前走一点,循环。
#       把真正「执行出来」的路径画进 (x,y,t) 空间,和人那条移动的安全管子对比——
#       看无人机是不是在「人正好挡路的那个时刻」才躲,而不是提前/滞后乱躲。
axB = fig.add_subplot(1, 2, 2, projection="3d")
# 中文:人从 y=-3.6 出发以 1.5 m/s 沿 +y 横穿;无人机要从 s2 走到 g2。hc=人当前中心,
#       drone/dvel=无人机当前位置/速度,tg=已经走过的全局时间。
s2 = np.array([0.0, 0.0, 1.5]); g2 = np.array([8.0, 0.0, 1.5])
h0 = np.array([4.0, -3.6, 1.5]); hvel = np.array([0.0, 1.5, 0.0])
wall2 = AABBObstacle([3.0, 1.2, 0.0], [5.0, 3.0, 3.0], class_name="wall")
DT = 0.30
drone = s2.copy(); dvel = np.zeros(3); hc = h0.copy(); tg = 0.0
exec_xyt = []; clrs = []
tr = None
# 中文:滚动时域主循环,最多 40 个 tick(到目标就提前 break)。
for tick in range(40):
    # 中文:用人「此刻」的位置和速度建一个会动的球障碍(规划器内部会用它预测未来位置)。
    mover = SphereObstacle(hc.copy(), 0.4, vel=hvel, class_name="human")
    # 中文:从无人机当前状态(带初速度 v0=dvel)重新规划到目标。
    tr, _ = plan_minco(np.linspace(drone, g2, 8), [mover, wall2], cfg,
                       opt_params=OptParams(), detour_cfg=ND, v0=dvel)
    # 中文:只执行(提交)未来 DT 这一小段;采样这段记录执行轨迹和到人的净空。
    te = min(DT, tr.t_end)
    for a in np.linspace(0.0, te, 8):
        p = tr.eval(float(a)); hp = hc + hvel * float(a)
        exec_xyt.append([p[0], p[1], tg + float(a)])
        clrs.append(float(np.linalg.norm(p - hp) - 0.4))
    # 中文:推进一步——无人机/人都前进 te,全局时间累加,然后进入下一次重规划。
    drone = tr.eval(te); dvel = tr.eval_deriv(te, 1); hc = hc + hvel * te; tg += te
    if np.linalg.norm(drone[:2] - g2[:2]) < 0.3:
        break
E = np.array(exec_xyt)                                   # (N,3) executed (x,y,t)
min_clr_cl = float(np.min(clrs))
T_end = float(E[-1, 2])
tt = np.linspace(0.0, T_end, 200)
hpath = h0[None, :] + hvel[None, :] * tt[:, None]        # human centre over global time

axB.plot(E[:, 0], E[:, 1], E[:, 2], "-", color="C0", lw=2.8, label="drone EXECUTED (x,y,t)")
axB.plot(hpath[:, 0], hpath[:, 1], tt, "-", color="C3", lw=2.0, label="human centre (x,y,t)")
# 中文:R=人半径+安全圈,沿时间每隔一段画一个圆环,叠起来就是那条「移动的安全管子」。
R = 0.4 + cfg["human"].d_safe
ang = np.linspace(0, 2 * np.pi, 24)
for k in range(0, len(tt), 16):
    axB.plot(hpath[k, 0] + R * np.cos(ang), hpath[k, 1] + R * np.sin(ang),
             np.full_like(ang, tt[k]), color="C3", alpha=0.30, lw=0.8)
axB.set_xlabel("x [m]"); axB.set_ylabel("y [m]"); axB.set_zlabel("time [s]")
axB.set_title(f"(b) space-time (x,y,t): CLOSED-LOOP vs crossing human\n"
              f"replan-each-tick stays clear (min clr {min_clr_cl:.2f} m >= d_safe 0.8)")
axB.legend(loc="upper left", fontsize=8); axB.view_init(elev=18, azim=-72)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_3d_perclass.png")
fig.savefig(out, dpi=130)
print("saved", out)
print(f"(a) spatial : valid={info['trajectory_valid']} min_clr={info['continuous_min_clearance']:.3f} "
      f"n_active_force={len(certs)}")
print(f"(b) spacetime closed-loop: min_clr={min_clr_cl:.3f} ticks={len(E)} T_end={T_end:.2f}")
