"""SANDO-level REAL-TIME 3D demo (animated GIF, headless via pillow).

中文说明(这个文件是干嘛的):
  实时 3D 动图演示(headless 用 pillow 存成 GIF)。这是一次真正的闭环/滚动时域飞行:
  每个 tick 从无人机当前状态重新规划,人在两次规划之间会移动,无人机在 3D 里实时反应,
  把 HARD 的人挡在它的安全圈(d_safe)外面,同时可以蹭 SOFT 的墙。
  最后用一个会缓慢旋转的 3D 视角,画出已飞过的轨迹尾迹 + 当前这次的 MINCO 计划。
  输出:demo_anim_perclass.gif。本场景里人是「静止挡在路中间」的(HVEL=0),逼无人机绕一个 3D 弧。

A genuine closed-loop / receding-horizon flight in 3D:
  - every tick the planner RE-PLANS from the drone's current state (real-time),
  - the human MOVES between replans (dynamic obstacle),
  - the drone reacts in 3D, keeping the HARD human outside its d_safe shell while
    being free to graze the SOFT wall,
  - shown in a rotating 3D view with the executed trail + current MINCO plan.
Output: demo_anim_perclass.gif
"""
import os
import sys

import matplotlib
# 中文:headless 后端,只出文件不弹窗。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# 中文:不装包也能 import 本地模块。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402

# 中文:全局场景设置。detour 关掉;起点 z=1.0、终点 z=2.0(z 上升=真 3D);只配一个 human=HARD;
#       HR=人半径,DSAFE=安全圈;H0=人初始位置,HVEL=0 表示人不动、就堵在路中间逼无人机绕弧;
#       DT=每个 tick 提交执行的时长,SUB=每个 tick 在动图里细分成几帧。
ND = DetourConfig(enabled=False)
START = np.array([0.0, 0.0, 1.0]); GOAL = np.array([8.0, 0.0, 2.0])        # z rises -> 3D
CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4)}
HR, DSAFE = 0.4, 0.8
H0 = np.array([4.0, 0.0, 1.5]); HVEL = np.array([0.0, 0.0, 0.0])           # blocks the path -> forces a 3D arc
DT, SUB = 0.22, 3


# 中文:盒子 6 个面的顶点(画长方体用);本 demo 没画墙,留着是通用工具。
def box_faces(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return [v[idx] for idx in f]


# 中文:球面网格(画人的实心球和半透明安全壳)。
def sphere_xyz(c, r, n=16):
    u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
    return (c[0] + r * np.outer(np.cos(u), np.sin(v)),
            c[1] + r * np.outer(np.sin(u), np.sin(v)),
            c[2] + r * np.outer(np.ones_like(u), np.cos(v)))


# ---------------- REAL-TIME closed-loop rollout (re-plan every tick) --------
# 中文:闭环滚动时域主循环,跑在画图之前——先把每一帧要显示的数据都算好存进 frames,
#       后面动画只负责「回放」。drone/dvel=无人机状态,hc=人位置,tg=全局时间,trail=已飞尾迹。
drone = START.copy(); dvel = np.zeros(3); hc = H0.copy(); tg = 0.0
trail = [drone.copy()]; frames = []
for tick in range(48):
    # 中文:用人此刻状态建会动的球障碍,从无人机当前状态(带初速度 v0=dvel)重规划。
    mover = SphereObstacle(hc.copy(), HR, vel=HVEL, class_name="human")
    tr, info = plan_minco(np.linspace(drone, GOAL, 8), [mover], CFG,
                          opt_params=OptParams(), detour_cfg=ND, v0=dvel)
    # 中文:把当前这次完整计划密采 60 点,用来在动图里画「当前 MINCO 计划」那条淡线。
    Pplan = tr.eval(np.linspace(tr.t_start, tr.t_end, 60))
    # 中文:只执行未来 DT,这段细分成 SUB 帧记下来(净空 clr=到人距离减人半径)。
    te = min(DT, tr.t_end)
    for a in np.linspace(0.0, te, SUB, endpoint=False):
        dp = tr.eval(float(a)); hp = hc + HVEL * float(a)
        trail.append(dp.copy())
        frames.append(dict(drone=dp.copy(), human=hp.copy(), planned=Pplan.copy(),
                           trail=np.array(trail), clr=float(np.linalg.norm(dp - hp) - HR),
                           t=tg + float(a)))
    # 中文:推进一步,进入下一次重规划;到目标就提前结束。
    drone = tr.eval(te); dvel = tr.eval_deriv(te, 1); hc = hc + HVEL * te; tg += te
    if np.linalg.norm(drone[:2] - GOAL[:2]) < 0.3:
        break

# 中文:逐帧的「到目前为止最小净空」(单调不增),用来在终端打印这次跑的安全余量。
min_run = [min(f["clr"] for f in frames[:i + 1]) for i in range(len(frames))]
print(f"frames={len(frames)} ticks~{len(frames)//SUB} min_clr={min(min_run):.3f} "
      f"reached={np.linalg.norm(frames[-1]['drone'][:2]-GOAL[:2])<0.4}")

# ---------------- 3D animation ----------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")


# 中文:动画的逐帧回调,i=第几帧。每帧清空重画:人(球+安全壳)、当前 MINCO 计划、已飞尾迹、
#       无人机当前位置、目标星标,并随帧号缓慢转视角。
def draw(i):
    f = frames[i]
    ax.clear()
    ax.set_xlim(0, 8); ax.set_ylim(-2.2, 2.2); ax.set_zlim(0.4, 2.6)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    # human (HARD): body + translucent d_safe shell (MOVING)
    hx, hy, hz = sphere_xyz(f["human"], HR)
    ax.plot_surface(hx, hy, hz, color="C3", alpha=0.95, linewidth=0)
    sx, sy, sz = sphere_xyz(f["human"], HR + DSAFE)
    ax.plot_surface(sx, sy, sz, color="C3", alpha=0.10, linewidth=0)
    # current MINCO plan + executed trail + drone
    Pp = f["planned"]
    ax.plot(Pp[:, 0], Pp[:, 1], Pp[:, 2], "-", color="C0", alpha=0.5, lw=1.6)
    tr_ = f["trail"]
    ax.plot(tr_[:, 0], tr_[:, 1], tr_[:, 2], "-", color="C1", lw=2.6)
    ax.scatter(*f["drone"], color="k", s=45)
    ax.scatter(*GOAL, color="g", marker="*", s=160)
    ax.plot([], [], color="C1", lw=2.6, label="executed trail")
    ax.plot([], [], color="C0", lw=1.6, label="current MINCO plan (replanned each tick)")
    ax.plot([], [], color="C3", lw=6, alpha=0.4, label="human (HARD) + d_safe")
    ax.set_title(f"REAL-TIME 3D: replan every tick, drone arcs around the HARD obstacle   "
                 f"t={f['t']:.1f}s   clearance={f['clr']:.2f} m (d_safe {DSAFE})")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-60 + 0.35 * i)        # slow rotation


# 中文:把所有帧串成动画并用 PillowWriter 存成 GIF(12 帧/秒)。
anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_anim_perclass.gif")
anim.save(out, writer=PillowWriter(fps=12))
print("saved", out)
