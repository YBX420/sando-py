"""方向 A / §7 —— AFTER 原型:anytime-feasible feasible-direction 内层求解。

核心主张:从一个【可行】轨迹出发(min 人体净空 >= d_safe),用 feasible-direction 步
(SS-QCQP 思路:搜索方向投影到保持每个人体约束 g<=0),则【每一个迭代】都保持净空 >= d_safe
=> 任意时刻 deadline 截断,返回的轨迹都人体安全(computation-invariant 行人安全)。

对照:同一起点上跑普通 cost 梯度下降(当前 optimize-then-check 内层的本质:只降 cost,不守约束)
=> 它会被 cost 拉向直线最短路 = 滑进行人 => 净空跌破 d_safe => 在那截断不安全。

输出:media/_exp_anytime_feasible.png —— 两条 min-人体净空 vs 迭代曲线。
跑:  python _exp_anytime_feasible.py
注:原型用有限差分算约束 Jacobian(非 <50ms 级),只为证明 anytime-feasible 这个性质成立。
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize

from sando_py.local.local_opt import (OptParams, _minco_seed, _optimise_one_minco,
                                       MinjerkTraj, _minco_cost_grad)
from sando_py.local.obstacles import SphereObstacle
from sando_py.local.avoid_config import default_config

cfg = default_config()
D_SAFE = cfg["human"].d_safe

# ---- 直接构造:M=4,3 个静态行人骑在中线两侧;直线最短路会撞,但小幅蛇形可行 ----
# (静态行人 + 手工可行起点 = 干净的第一张 AFTER 图;动态版是下一步)
start = np.array([0.0, 0.0, 2.0]); goal = np.array([10.0, 0.0, 2.0])
M = 4; nq = 3 * (M - 1)
V0 = np.zeros(3); A0 = np.zeros(3)
HX = [2.5, 5.0, 7.5]; HY = [0.8, -0.8, 0.8]
HUM = [SphereObstacle(centre0=np.array([hx, hy, 2.0]), radius=0.3,
                      vel=np.zeros(3), class_name="human", accel=np.zeros(3))
       for hx, hy in zip(HX, HY)]
opt = OptParams(vmax=6.0, amax=10.0)
T0 = np.full(M, 10.0 / M / opt.vmax * 1.6)   # 段时长(略松,够避让)
T_target = float(np.sum(T0))


def make_tr(x):
    q = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    return MinjerkTraj.from_endpoints(start, goal, q, x[nq:], v0=V0, a0=A0)


def clr_per_human(x, K=120):
    """每个行人在整条轨迹上的最小净空(signed_dist)。"""
    tr = make_tr(x)
    ts = np.linspace(tr.t_start, tr.t_end, K)
    P = tr.eval(ts)
    out = []
    for h in HUM:
        d = min(h.signed_dist(P[j], float(ts[j])) for j in range(K))
        out.append(d)
    return np.array(out)


def min_clr(x):
    return float(clr_per_human(x).min())


def cost_and_grad(x):
    """纯 cost(平滑+时间+速度/加速度;人体不进 cost,是硬约束)。"""
    f, g = _minco_cost_grad(x, start, goal, M, [], cfg, opt, T_target, v0=V0, a0=A0)
    return f, g


# constraint g_h(x) = d_safe - min_clr_h(x)  (<=0 feasible). active if近边界
def cons_vec(x):
    return D_SAFE - clr_per_human(x)


def cons_jac_active(x, active, eps=1e-4):
    """有限差分算【活跃】约束对 x 的雅可比(只对活跃约束,省算)。"""
    base = cons_vec(x)
    J = np.zeros((len(active), x.size))
    for col in range(x.size):
        xp = x.copy(); xp[col] += eps
        cp = cons_vec(xp)
        J[:, col] = (cp[active] - base[active]) / eps
    return J, base


# ---- 手工构造可行起点:内部点蛇形绕开每个行人,自动放大幅度直到 min_clr >= d_safe+buffer ----
def build_x(amp):
    q = np.array([[2.5, -amp, 2.0], [5.0, +amp, 2.0], [7.5, -amp, 2.0]])  # 与行人 y 相反
    return np.concatenate([q.reshape(-1), T0])

x_feas = None
for amp in np.linspace(0.4, 2.2, 28):
    xc = build_x(amp)
    if clr_per_human(xc).min() >= D_SAFE + 0.05:
        x_feas = xc; break
if x_feas is None:
    x_feas = build_x(2.2)
c0 = clr_per_human(x_feas)
print(f"d_safe={D_SAFE:.2f}  constructed feasible start: weave amp={amp:.2f}  "
      f"min-clr per human = {np.round(c0,3)}  (min={c0.min():.3f})")

bounds_lo = np.array([-np.inf] * nq + [opt.dt_min] * M)


def feasible_direction(x0, iters=40, alpha=0.5):
    """SS-QCQP-lite:方向 u=argmin 1/2||u+gf||^2 s.t. ∇g_a·u <= -alpha*g_a(活跃约束);
    回溯步长直到【真实】所有 g<=0(前向不变)。每个迭代都可行。"""
    x = x0.copy(); hist = [min_clr(x)]
    for _ in range(iters):
        f, gf = cost_and_grad(x)
        gc = cons_vec(x)
        active = [i for i in range(len(HUM)) if gc[i] > -0.15]   # 近边界的约束才约束方向
        if active:
            J, base = cons_jac_active(x, active)
            ga = gc[active]
            cons = [{"type": "ineq",
                     "fun": (lambda u, Ji=J[i], gi=ga[i]: -(Ji @ u + alpha * gi))}
                    for i in range(len(active))]
        else:
            cons = []
        # 方向子问题:min 1/2||u+gf||^2,限制 ||u|| 不要太大(信赖域)
        u0 = -gf / (np.linalg.norm(gf) + 1e-9)
        res = scipy.optimize.minimize(
            lambda u: 0.5 * float(np.dot(u + gf, u + gf)), u0, jac=lambda u: u + gf,
            constraints=cons, method="SLSQP",
            bounds=[(-5, 5)] * x.size, options={"maxiter": 30, "ftol": 1e-6})
        u = res.x
        if np.linalg.norm(u) < 1e-8:
            break
        # 回溯:保证步后【真实】所有 g<=0(净空>=d_safe)且 cost 不升
        t = 1.0; f0 = f; moved = False
        for _bt in range(30):
            xn = x + t * u
            xn[nq:] = np.maximum(xn[nq:], opt.dt_min)   # T 下界
            if clr_per_human(xn).min() >= D_SAFE - 1e-9:   # 前向不变:真实可行才接受
                fn, _ = cost_and_grad(xn)
                if fn <= f0 + 1e-9:
                    x = xn; moved = True; break
            t *= 0.5
        hist.append(min_clr(x))
        if not moved:
            break
    return x, np.array(hist)


def plain_gd(x0, iters=40):
    """当前内层本质:只降 cost、不守人体约束(回溯只看 cost 下降)。会滑进行人。"""
    x = x0.copy(); hist = [min_clr(x)]
    for _ in range(iters):
        f, gf = cost_and_grad(x)
        u = -gf / (np.linalg.norm(gf) + 1e-9)
        t = 0.3
        for _bt in range(20):
            xn = x + t * u; xn[nq:] = np.maximum(xn[nq:], opt.dt_min)
            fn, _ = cost_and_grad(xn)
            if fn <= f + 1e-9:
                x = xn; break
            t *= 0.5
        hist.append(min_clr(x))
    return x, np.array(hist)


xf, hist_fd = feasible_direction(x_feas)
xg, hist_gd = plain_gd(x_feas)
print(f"\nfeasible-direction: min-clr over ALL iterates = {hist_fd.min():.3f}  "
      f"(>= d_safe? {'YES — every iterate human-safe' if hist_fd.min() >= D_SAFE - 1e-3 else 'NO'})")
print(f"plain cost GD     : min-clr over ALL iterates = {hist_gd.min():.3f}  "
      f"(dips below d_safe? {'YES — truncating there is UNSAFE' if hist_gd.min() < D_SAFE - 1e-3 else 'no'})")

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(D_SAFE, color="red", ls="--", lw=1.3, label=f"d_safe = {D_SAFE:.2f} (human safety)")
ax.plot(hist_fd, "o-", color="tab:green", label="anytime feasible-direction (A) — every iterate ≥ d_safe")
ax.plot(hist_gd, "s-", color="tab:orange", label="current cost-descent inner — dips below (unsafe if truncated)")
ax.fill_between(range(len(hist_gd)), 0, D_SAFE, color="red", alpha=0.06)
ax.set_xlabel("inner iteration (= where a deadline might truncate)")
ax.set_ylabel("min human clearance (m)")
ax.set_title("Direction A: anytime-feasible inner solve keeps EVERY iterate human-safe → truncation-invariant safety")
ax.legend(loc="best"); ax.grid(alpha=0.3)
os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "_exp_anytime_feasible.png"), dpi=120)
print("wrote media/_exp_anytime_feasible.png")
