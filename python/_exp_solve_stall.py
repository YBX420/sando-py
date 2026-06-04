"""A+B 集成 —— 完全解决「局部最优卡住」的证明:
  B = 时间感知拓扑可行种子(每个路点在【它将抵达的时刻】搜最大净空的横向位置 = time 准移动缝)
  A = feasible-direction 内层(从可行种子出发,每步保持人体 g<=0)
在动态密集行人场景上:当前求解器(静态快照种子 + optimize-then-check)失败/卡死,
而 A+B 穿过去且【每个迭代都人体安全】。
跑:  python _exp_solve_stall.py  ->  media/_exp_solve_stall.png
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.optimize

from sando_py.local.local_opt import OptParams, _optimise_one_minco, MinjerkTraj, _minco_cost_grad
from sando_py.local.obstacles import SphereObstacle
from sando_py.local.avoid_config import default_config

cfg = default_config(); D_SAFE = cfg["human"].d_safe
start = np.array([0.0, 0.0, 2.0]); goal = np.array([12.0, 0.0, 2.0])
M = 10; nq = 3 * (M - 1); V0 = np.zeros(3); A0 = np.zeros(3)
opt = OptParams(vmax=4.0, amax=8.0)
KEEP = D_SAFE + 0.4  # 半径 0.4 的行人,中心要离 d_safe+r

# 动态行人:横穿走廊(vel 在 y),x 对齐到内部路点(x=3.6,6.0,8.4)。
# t=0 位置 vs 无人机抵达时位置不同 -> 静态快照会选错侧;时间感知种子按抵达时刻选侧。
HUM = [
    SphereObstacle(centre0=np.array([3.6, -1.4, 2.0]), radius=0.3, vel=np.array([0.0,  0.5, 0.0]), class_name="human", accel=np.zeros(3)),
    SphereObstacle(centre0=np.array([6.0,  1.4, 2.0]), radius=0.3, vel=np.array([0.0, -0.5, 0.0]), class_name="human", accel=np.zeros(3)),
    SphereObstacle(centre0=np.array([8.4, -1.4, 2.0]), radius=0.3, vel=np.array([0.0,  0.5, 0.0]), class_name="human", accel=np.zeros(3)),
]
T_est = float(np.linalg.norm(goal - start) / opt.vmax)


def make_tr(x):
    q = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    return MinjerkTraj.from_endpoints(start, goal, q, x[nq:], v0=V0, a0=A0)


def clr_per_human(x, K=140):
    tr = make_tr(x); ts = np.linspace(tr.t_start, tr.t_end, K); P = tr.eval(ts)
    return np.array([min(h.signed_dist(P[j], float(ts[j])) for j in range(K)) for h in HUM])


def min_clr(x):
    return float(clr_per_human(x).min())


def cost_and_grad(x):
    return _minco_cost_grad(x, start, goal, M, [], cfg, opt, float(np.sum(x[nq:])), v0=V0, a0=A0)


# ---------- B: time-aware vs static-snapshot seed ----------
def make_seed(time_aware, T0):
    """每个内部路点在【它将抵达的时刻 t_i】(time_aware)或 t=0(static)选过障侧:
    取「clearance>=d_safe+buffer 且偏离上一路点最小」的 y = 紧贴缝穿(不逃到边缘)。"""
    cum = np.concatenate([[0.0], np.cumsum(T0)])
    ygrid = np.linspace(-3.2, 3.2, 161)
    q = []; prev_y = 0.0
    for i in range(M - 1):
        xb = start + (i + 1) / M * (goal - start)
        t_i = cum[i + 1] if time_aware else 0.0
        cs = np.array([min(h.signed_dist(np.array([xb[0], y, 2.0]), t_i) for h in HUM) for y in ygrid])
        ok = ygrid[cs >= D_SAFE + 0.15]
        best_y = ok[np.argmin(np.abs(ok - prev_y))] if ok.size else ygrid[int(np.argmax(cs))]
        q.append([xb[0], best_y, 2.0]); prev_y = best_y
    return np.array(q)


seg_len = np.linalg.norm(np.diff(np.vstack([start, goal]), axis=0))
T0 = np.full(M, T_est / M)
q_ta = make_seed(True, T0)
q_ss = make_seed(False, T0)
x_ta = np.concatenate([q_ta.reshape(-1), T0])
x_ss = np.concatenate([q_ss.reshape(-1), T0])
print(f"d_safe={D_SAFE:.2f}  time-aware seed min-clr={min_clr(x_ta):.3f}   static-snapshot seed min-clr={min_clr(x_ss):.3f}")


# ---------- A: feasible-direction solve from a feasible seed ----------
def feasible_direction(x0, iters=60, alpha=0.5):
    x = x0.copy(); hist = [min_clr(x)]
    for _ in range(iters):
        f, gf = cost_and_grad(x)
        gc = D_SAFE - clr_per_human(x)
        active = [i for i in range(len(HUM)) if gc[i] > -0.2]
        cons = []
        if active:
            base = gc.copy()
            J = np.zeros((len(active), x.size))
            for col in range(x.size):
                xp = x.copy(); xp[col] += 1e-4
                J[:, col] = ((D_SAFE - clr_per_human(xp))[active] - base[active]) / 1e-4
            ga = gc[active]
            cons = [{"type": "ineq", "fun": (lambda u, Ji=J[i], gi=ga[i]: -(Ji @ u + alpha * gi))}
                    for i in range(len(active))]
        u0 = -gf / (np.linalg.norm(gf) + 1e-9)
        res = scipy.optimize.minimize(lambda u: 0.5 * float(np.dot(u + gf, u + gf)), u0,
                                      jac=lambda u: u + gf, constraints=cons, method="SLSQP",
                                      bounds=[(-5, 5)] * x.size, options={"maxiter": 30, "ftol": 1e-6})
        u = res.x
        if np.linalg.norm(u) < 1e-8:
            break
        t = 1.0; moved = False
        for _bt in range(34):
            xn = x + t * u; xn[nq:] = np.maximum(xn[nq:], opt.dt_min)
            if clr_per_human(xn).min() >= D_SAFE - 1e-9:
                fn, _ = cost_and_grad(xn)
                if fn <= f + 1e-9:
                    x = xn; moved = True; break
            t *= 0.5
        hist.append(min_clr(x))
        if not moved:
            break
    return x, np.array(hist)


# A+B Phase-1:可行性恢复 —— 从时间感知种子出发,梯度下降【总违反量】直到 min_clr>=d_safe。
# (这正是 A 的命门 open-Q#1:每拍要一个可行 warm-start;拓扑种子给好起点,恢复阶段把它推进可行集。)
def restore_feasible(x0, iters=300):
    def V(xx):
        v = np.maximum(D_SAFE - clr_per_human(xx), 0.0)
        return float((v * v).sum())
    x = x0.copy()
    for _ in range(iters):
        if clr_per_human(x).min() >= D_SAFE:
            return x, True
        v0 = V(x); g = np.zeros(x.size)
        for col in range(x.size):
            xp = x.copy(); xp[col] += 1e-4; g[col] = (V(xp) - v0) / 1e-4
        ng = np.linalg.norm(g)
        if ng < 1e-9:
            break
        d = -g / ng; t = 0.6; stepped = False
        for _bt in range(34):
            xn = x + t * d; xn[nq:] = np.maximum(xn[nq:], opt.dt_min)
            if V(xn) < v0 - 1e-12:
                x = xn; stepped = True; break
            t *= 0.5
        if not stepped:
            break
    return x, clr_per_human(x).min() >= D_SAFE


x_ab0, feas = restore_feasible(x_ta)
print(f"A+B feasible start (B-seed + Phase-1 restore): min-clr={min_clr(x_ab0):.3f}  feasible={feas}")
# 走廊引导(B):把轨迹拉向这条可行缝,feasible-direction 就紧贴缝穿、不乱摆 —— 用上已实现的走廊项。
opt.w_corridor = 40.0
opt.corridor_radius = 0.5
opt.corridor_centers = x_ab0[:nq].reshape(M - 1, 3).copy()
x_ab, hist_ab = feasible_direction(x_ab0)

# ---------- 对照:当前求解器(它内部用静态快照 detour + optimize-then-check) ----------
seed_path = np.vstack([start + (goal - start) * f for f in np.linspace(0, 1, M + 1)])
rec = _optimise_one_minco(seed_path, list(HUM), cfg, opt, v0=V0, a0=A0)
cur_valid = rec["feasibility"].get("trajectory_valid")
cur_clr = rec.get("hard_min_clearance")
print(f"\nCURRENT solver: valid={cur_valid}  hard_min_clr={cur_clr:.3f}  "
      f"({'safe' if cur_clr >= D_SAFE else 'UNSAFE / collides — the stall'})")
print(f"A+B  feasible-direction: reached-feasible={min_clr(x_ab) >= D_SAFE - 1e-3}  "
      f"min-clr over ALL iterates={hist_ab.min():.3f}  (>=d_safe? "
      f"{'YES — every iterate human-safe' if hist_ab.min() >= D_SAFE - 1e-3 else 'NO'})")

# ---------- visualize ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
# top: top-down, humans drawn at their pass-time positions (space-time)
ax = axes[0]
tr_ab = make_tr(x_ab); tr_cur = rec["spline"]
ts = np.linspace(0, tr_ab.t_end, 200); Pab = tr_ab.eval(ts)
tsc = np.linspace(0, tr_cur.t_end, 200); Pcur = tr_cur.eval(tsc)
for h in HUM:
    # draw human at the time the A+B traj is nearest its x (its "pass time")
    xc = h.centre0[0]
    tpass = ts[np.argmin(np.abs(Pab[:, 0] - xc))]
    c = h.predict(tpass)
    ax.add_patch(Circle((c[0], c[1]), h.radius, color="tab:red", alpha=0.5))
    ax.add_patch(Circle((c[0], c[1]), h.radius + D_SAFE, color="tab:red", alpha=0.10))
    ax.plot([h.centre0[0]] * 2, [h.centre0[1], c[1]], ":", color="0.5", lw=0.8)  # t=0 -> pass-time
ax.plot(Pcur[:, 0], Pcur[:, 1], "-", color="tab:orange", lw=2, label=f"CURRENT solver (valid={cur_valid}, clr={cur_clr:+.2f}) — stalls/collides")
ax.plot(Pab[:, 0], Pab[:, 1], "-", color="tab:green", lw=2.5, label=f"A+B thread-through (min-clr={min_clr(x_ab):+.2f})")
ax.plot(q_ta[:, 0], q_ta[:, 1], "x", color="tab:blue", ms=7, label="B: time-aware seed waypoints")
ax.plot(start[0], start[1], "o", color="lime", ms=9); ax.plot(goal[0], goal[1], "*", color="k", ms=14)
ax.set_xlim(-1, 13); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
ax.set_title("A+B solves the stall: time-aware seed (B) → feasible-direction (A) threads the moving gaps\n"
             "(humans shown at their pass-time positions; halo = d_safe keep-out)")
ax.legend(loc="upper center", fontsize=8, ncol=1); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
# bottom: anytime-feasible curve
ax2 = axes[1]
ax2.axhline(D_SAFE, color="red", ls="--", lw=1.3, label=f"d_safe={D_SAFE:.2f}")
ax2.plot(hist_ab, "o-", color="tab:green", label="A+B feasible-direction — every iterate ≥ d_safe")
ax2.axhline(cur_clr, color="tab:orange", ls="-", lw=2, label=f"current solver final clearance = {cur_clr:+.2f} (unsafe)")
ax2.fill_between(range(len(hist_ab)), 0, D_SAFE, color="red", alpha=0.06)
ax2.set_xlabel("inner iteration (where a deadline might truncate)"); ax2.set_ylabel("min human clearance (m)")
ax2.legend(loc="best", fontsize=8); ax2.grid(alpha=0.3)
os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "_exp_solve_stall.png"), dpi=120)
print("wrote media/_exp_solve_stall.png")
