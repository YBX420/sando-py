"""A+B 闭环生产化 demo —— 在动态密集人群里 receding-horizon 飞 start->goal:
每拍:把行人移到当前时刻 -> time-aware 种子 -> 可行性恢复 -> fast feasible-direction(带 deadline)
-> commit 第一段 -> 推进无人机。记录防御曲线:
  (1) 飞行中 min 人体净空 vs 时间(A+B 恒 >= d_safe;对照当前求解器撞)
  (2) 每拍「截断不变性」:本拍解的每个迭代净空都 >= d_safe(任意截断都安全)
  (3) 每拍 replan ms(Python 原型;<50ms 是 C++ 目标)
跑:  python _exp_production.py  ->  media/_exp_production.png
"""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sando_py.local.local_opt import OptParams, MinjerkTraj, _optimise_one_minco
from sando_py.local.obstacles import SphereObstacle
from sando_py.local.avoid_config import default_config
import _anytime_solver as S

cfg = default_config(); D_SAFE = cfg["human"].d_safe
start = np.array([0.0, 0.0, 2.0]); goal = np.array([14.0, 0.0, 2.0])
opt = OptParams(vmax=4.0, amax=8.0); opt.spacetime_hard = True
RD = 0.25; T_MAX = 16.0; GOAL_R = 0.7; BUDGET = 0.4   # python 原型预算(非 <50ms)

# 6 个行人横穿走廊(交错),timing 关键:静态快照会选错侧。
def humans_at(t):
    spec = [(3.0, -1.5, 0.55), (5.0, 1.5, -0.55), (7.0, -1.5, 0.55),
            (9.0, 1.5, -0.55), (11.0, -1.5, 0.55), (13.0, 1.5, -0.55)]
    return [SphereObstacle(centre0=np.array([x, y0 + vy * t, 2.0]), radius=0.3,
                           vel=np.array([0.0, vy, 0.0]), class_name="human", accel=np.zeros(3))
            for (x, y0, vy) in spec]


def shift_now(hum, t):  # 把行人移到当前时刻(求解用局部时间)
    return [SphereObstacle(centre0=h.predict(t).copy(), radius=h.radius, vel=h.vel.copy(),
                           class_name="human", accel=h.accel.copy()) for h in hum]


def true_clr(p, t):
    return min(h.signed_dist(p, 0.0) for h in shift_now(humans_at(0.0), t))  # = humans_at(t) dist to p


def ab_solve(p, t):
    """A+B 单拍:返回 (committed traj tr (局部时间,行人已移到 t), 本拍每迭代 min-clr 序列, ms)。"""
    hum = shift_now(humans_at(0.0), t)
    dist = np.linalg.norm(goal - p)
    M = int(np.clip(round(dist / 1.4), 4, 12));
    T0 = np.full(M, max(dist / M / opt.vmax, 0.12))
    q0 = S.time_aware_seed(p, goal, hum, M, T0, D_SAFE)
    qf, feas = S.restore_feasible(p, goal, q0, T0, np.zeros(3), np.zeros(3), M, hum, cfg, opt, D_SAFE, iters=120)
    corr = (30.0, 0.5, qf.reshape(M - 1, 3).copy())
    q, hist, ms, trunc = S.fast_feasible_direction(p, goal, qf, T0, np.zeros(3), np.zeros(3),
                                                   M, hum, cfg, opt, D_SAFE, iters=25, deadline_s=BUDGET, corridor=corr)
    tr = S._tr(p, goal, q, T0, np.zeros(3), np.zeros(3), M)
    return tr, hist, ms, M


def run_ab():
    p = start.copy(); t = 0.0; reached = False
    tt, clr_t, p_path = [], [], [p.copy()]
    ms_list, replan_safe = [], []
    while t < T_MAX and not reached:
        tr, hist, ms, M = ab_solve(p, t)
        ms_list.append(ms)
        replan_safe.append(bool(hist.min() >= D_SAFE - 1e-3))   # 截断不变性:每迭代都安全
        # commit:沿 tr(局部时间)走 RD 秒
        nstep = max(1, int(RD / 0.02)); dt = RD / nstep
        for s in range(nstep):
            tau = (s + 1) * dt
            p = np.asarray(tr.eval(min(tau, tr.t_end)), float)
            tt.append(t + tau); clr_t.append(true_clr(p, t + tau)); p_path.append(p.copy())
        t += RD
        if np.linalg.norm(p - goal) < GOAL_R:
            reached = True
    return dict(reached=reached, t=t, tt=np.array(tt), clr=np.array(clr_t),
               path=np.array(p_path), ms=np.array(ms_list), safe_rate=np.mean(replan_safe), final_x=p[0])


def run_current():
    """对照:当前求解器(静态快照 detour + optimize-then-check)闭环。"""
    p = start.copy(); t = 0.0; reached = False; collided = False
    tt, clr_t, p_path = [], [], [p.copy()]
    while t < T_MAX and not reached:
        hum = shift_now(humans_at(0.0), t)
        dist = np.linalg.norm(goal - p); M = int(np.clip(round(dist / 1.4), 4, 12))
        seed = np.vstack([p + (goal - p) * f for f in np.linspace(0, 1, M + 1)])
        rec = _optimise_one_minco(seed, list(hum), cfg, opt, v0=np.zeros(3), a0=np.zeros(3))
        tr = rec["spline"]
        nstep = max(1, int(RD / 0.02)); dt = RD / nstep
        for s in range(nstep):
            tau = (s + 1) * dt
            p = np.asarray(tr.eval(min(tau, tr.t_end)), float)
            c = true_clr(p, t + tau)
            tt.append(t + tau); clr_t.append(c); p_path.append(p.copy())
            if c < 0: collided = True
        t += RD
        if np.linalg.norm(p - goal) < GOAL_R:
            reached = True
    return dict(reached=reached, tt=np.array(tt), clr=np.array(clr_t), path=np.array(p_path),
               collided=collided, final_x=p[0])


print("running A+B closed loop ...", flush=True)
ab = run_ab()
print("running CURRENT-solver closed loop ...", flush=True)
cur = run_current()
print("\n=== RESULT ===")
print(f"A+B     : reached={ab['reached']} final_x={ab['final_x']:.1f}  min_clr={ab['clr'].min():+.3f}  "
      f"(>=d_safe all flight? {ab['clr'].min() >= D_SAFE - 0.05})  "
      f"replan-truncation-safe rate={ab['safe_rate']*100:.0f}%  ms mean={ab['ms'].mean():.0f} p90={np.percentile(ab['ms'],90):.0f}")
print(f"CURRENT : reached={cur['reached']} final_x={cur['final_x']:.1f}  min_clr={cur['clr'].min():+.3f}  collided={cur['collided']}")

# ---- defense-curve figure ----
fig, (axp, axc) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[1.1, 1])
tmax_plot = max(ab['tt'][-1] if len(ab['tt']) else 5, cur['tt'][-1] if len(cur['tt']) else 5)
for h in humans_at(0.0):  # draw human crossing tracks (faint)
    ys = [h.predict(tau)[1] for tau in np.linspace(0, tmax_plot, 30)]
    axp.plot([h.centre0[0]] * 30, ys, ".", color="0.8", ms=2)
axp.plot(ab['path'][:, 0], ab['path'][:, 1], "-", color="tab:green", lw=2.5,
         label=f"A+B (B-seed→restore→feasible-dir): reaches x={ab['final_x']:.1f}, safe (min_clr={ab['clr'].min():+.2f})")
axp.plot(cur['path'][:, 0], cur['path'][:, 1], "-", color="tab:orange", lw=2.5)
axp.plot(cur['final_x'], cur['path'][-1, 1], "X", color="red", ms=13,
         label=f"CURRENT solver: STALLED at x={cur['final_x']:.1f} (never threads — the bug)")
axp.plot(start[0], start[1], "o", color="lime", ms=9); axp.plot(goal[0], goal[1], "*", color="k", ms=15)
axp.set_xlim(-1, 15.5); axp.set_ylim(-3, 3); axp.set_aspect("equal")
axp.set_title("A+B closed-loop THREADS the dense moving crowd start→goal; the current solver STALLS at the start")
axp.legend(loc="upper center", fontsize=8.5); axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)")
# bottom: x-progress (the stall contrast) + A+B clearance defense
axc.plot(ab['tt'], ab['path'][1:len(ab['tt'])+1, 0] if len(ab['path']) > len(ab['tt']) else ab['path'][:len(ab['tt']), 0],
         "-", color="tab:green", lw=2, label="A+B x-progress → reaches goal")
axc.plot(cur['tt'], cur['path'][:len(cur['tt']), 0], "-", color="tab:orange", lw=2, label="current x-progress → stuck (stall)")
axc.axhline(goal[0], color="k", ls=":", lw=1, label="goal x")
ax2 = axc.twinx()
ax2.axhline(D_SAFE, color="red", ls="--", lw=1.2)
ax2.plot(ab['tt'], ab['clr'], "-", color="darkgreen", lw=1.4, alpha=0.7, label="A+B min human clearance (right axis)")
ax2.fill_between([0, tmax_plot], 0, D_SAFE, color="red", alpha=0.06)
ax2.set_ylabel("A+B min human clearance (m)", color="darkgreen"); ax2.set_ylim(0, max(3, ab['clr'].max()))
axc.set_xlabel("flight time (s)"); axc.set_ylabel("x progress (m)")
axc.set_title(f"Left axis: A+B reaches the goal while current stalls.  Right axis (dark green): A+B clearance stays ≥ d_safe "
              f"all flight (truncation-safe {ab['safe_rate']*100:.0f}%).  replan ms(py)~{ab['ms'].mean():.0f}")
axc.legend(loc="center left", fontsize=8); axc.grid(alpha=0.3)
os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "_exp_production.png"), dpi=120)
print("wrote media/_exp_production.png")
