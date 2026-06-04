"""方向 A / §7 实验 —— BEFORE 测量:当前 optimize-then-check 的 ALM 内层在「早期外层迭代」上
是否人体不可行(=若 deadline 在那里截断,没有可证明安全的轨迹)。

做法:一条直线种子穿过 3 个移动行人(HARD,初始深度不可行)→ 跑现有 _optimise_one_minco,
用 _ALM_OUTER_TRACE 抓每个外层迭代的 (cert 违反 max_g、迭代轨迹的真实最小人体净空),打印 + 画图。

结论图:margin/clearance vs 外层迭代。早期 < 0 / < d_safe 的迭代 = 「在这里截断不安全」,
直到第一个 certified-feasible 迭代才有可 commit 的安全轨迹 —— 这正是 feasible-direction 升级要消灭的洞。
跑:  python _exp_anytime_margin.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sando_py.local.local_opt as LO
from sando_py.local.local_opt import (OptParams, _optimise_one_minco, _minco_seed,
                                       hard_clearance, MinjerkTraj)
from sando_py.local.obstacles import SphereObstacle
from sando_py.local.avoid_config import default_config

cfg = default_config()
d_safe = cfg["human"].d_safe   # human hard safety radius

# 直线种子穿过一团密集移动行人(深度不可行 -> 逼 ALM 多外层);
# alm_inner_maxiter 调小 = 每个外层只走几步 L-BFGS(紧 deadline 下的真实情形)-> 需要多个外层迭代
# 才到可行,于是早期外层迭代是 HUMAN-INFEASIBLE 的 —— 正是要展示的洞。
seed = np.array([[0, 0, 2], [2, 0, 2], [4, 0, 2], [6, 0, 2], [8, 0, 2], [10, 0, 2]], float)
_rng = np.random.default_rng(7)
humans = []
for k, xc in enumerate([2.6, 3.5, 4.5, 5.5, 6.5, 7.4]):
    yc = 0.30 * ((-1) ** k)                       # 交错骑在中线两侧,逼蛇形穿越
    vy = 0.4 * (-1) ** (k + 1)                    # 向中线挤
    humans.append(SphereObstacle(centre0=np.array([xc, yc, 2.0]), radius=0.55,
                                 vel=np.array([0.0, vy, 0.0]), class_name="human", accel=np.zeros(3)))
# 弱初始惩罚 rho0 -> 早期外层迭代被 cost 拉进行人(不可行),rho 逐渐放大才可行:
# 这正是 ALM 的真实动态,直接产出「早期 HUMAN-INFEASIBLE -> 后期 feasible」的干净轨迹。
opt = OptParams(vmax=6.0, amax=10.0, alm_rho0=0.3, alm_rho_grow=1.8, alm_inner_maxiter=20)

# capture per-outer-iter trace from the REAL solver
start, goal, q0, T0 = _minco_seed(seed, opt)
M = T0.size; nq = 3 * (M - 1)
LO._ALM_OUTER_TRACE = []
rec = _optimise_one_minco(seed, list(humans), cfg, opt, v0=np.zeros(3), a0=np.zeros(3))
trace = LO._ALM_OUTER_TRACE; LO._ALM_OUTER_TRACE = None

print(f"human d_safe = {d_safe:.2f} m ; outer iters traced = {len(trace)}")
print("outer |  cert max_g  | min_human_clr |  safe if truncated HERE?")
outs, certm, clrs = [], [], []
first_safe = None
for (outer, max_g, x) in trace:
    q = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    T = x[nq:]
    tr = MinjerkTraj.from_endpoints(start, goal, q, T, v0=np.zeros(3), a0=np.zeros(3))
    min_clr, max_breach = hard_clearance(tr, list(humans), cfg, K_eval=600)
    safe = (max_g <= 0.0) and (min_clr >= d_safe - 1e-6)
    if safe and first_safe is None:
        first_safe = outer
    print(f"  {outer:2d}  | {max_g:+11.4f} | {min_clr:12.3f}  |  {'YES' if safe else 'NO (unsafe -> would brake)'}")
    outs.append(outer); certm.append(-max_g); clrs.append(min_clr)

print(f"\nFirst CERTIFIED-FEASIBLE outer iter = {first_safe}  "
      f"-> truncating before it yields NO safe trajectory (current arch must brake / hold).")
print(f"final candidate: valid={rec.get('trajectory_valid')} reason={rec.get('failure_reason')}")

# figure: margin + clearance vs outer iter, shading the unsafe (early) region
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
a1.axhline(0, color="red", ls="--", lw=1, label="feasible boundary (margin=0)")
a1.plot(outs, certm, "o-", color="tab:blue")
a1.set_ylabel("cert margin = -max g\n(>=0 feasible)")
a1.set_title("Direction A / §7 BEFORE: current optimize-then-check — early ALM iterates are HUMAN-INFEASIBLE")
a1.legend(loc="lower right"); a1.grid(alpha=0.3)
a2.axhline(d_safe, color="red", ls="--", lw=1, label=f"d_safe={d_safe:.2f}")
a2.plot(outs, clrs, "s-", color="tab:green")
a2.set_ylabel("min human clearance (m)"); a2.set_xlabel("outer ALM iteration")
a2.legend(loc="lower right"); a2.grid(alpha=0.3)
if first_safe is not None and first_safe > 0:
    for ax in (a1, a2):
        ax.axvspan(min(outs) - 0.4, first_safe - 0.5, color="red", alpha=0.08)
    a1.text(min(outs), max(certm) * 0.5 if max(certm) > 0 else min(certm),
            "  truncate here = unsafe\n  (no certified trajectory)", color="darkred", fontsize=9, va="center")
os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "media", "_exp_anytime_margin.png"), dpi=120)
print("wrote media/_exp_anytime_margin.png")
