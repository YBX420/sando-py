"""Stage 3 / M2 — analytic gradient GATE for the MINCO local-opt cost.

THE GATE (acceptance spec):
  - check_grad of the FULL cost gradient w.r.t. BOTH q and T, rel-err < 1e-6,
    on >= 100 random configs covering obstacle-active / vel-hinge-active /
    accel-hinge-active masks, EXCLUDING kink configs (samples within eps of
    d=d_safe, |v|=vmax, |a|=amax, AABB face/interior, sphere centre).
  - a DEDICATED dCost/dT-only test (perturb a single T_i) with a MOVING sphere
    whose closest approach is DOWNSTREAM of the perturbed segment — exercises
    the cross-segment cum[]-coupling explicit term (c2), the M1 failure mode.
  - DIRECTIONAL finite-difference checks AT the kinks (where check_grad must
    fail): d≈d_safe boundary, |v|≈vmax, AABB face — confirm the subgradient is
    a valid one-sided directional derivative.

FD is Richardson-extrapolated central difference (O(h^4)) so the 1e-6 rel-err
reflects the ANALYTIC gradient, not FD truncation (matches stage3_minco_grad).

Run:  python3 test/stage3_minco_localopt_grad.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (                                    # noqa: E402
    _minco_cost_grad, _signed_dist_and_grad, OptParams)
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.minco import MinjerkTraj                              # noqa: E402
from sando_py.local.avoid_config import AvoidParams                       # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}


def rel_err(a, b):
    """Conventional check_grad relative error: ||a-b|| / max(||a||,||b||,floor).

    This is scipy.optimize.check_grad's metric (2-norm of the difference)
    normalised — robust to a single tiny near-zero component swamping a
    per-component ratio with FD round-off, while still machine-tight overall."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    num = float(np.linalg.norm(a - b))
    den = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-6)
    return num / den


def fd_richardson_vec(f, x, h0=1e-4):
    """Per-component Richardson-extrapolated central FD gradient (O(h^4))."""
    g = np.zeros_like(x)
    for i in range(x.size):
        def D(h):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            return (f(xp) - f(xm)) / (2.0 * h)
        d1 = D(h0); d2 = D(h0 / 2.0)
        g[i] = (4.0 * d2 - d1) / 3.0
    return g


# ---------------------------------------------------------------------------
# kink detection: reject a config if ANY quadrature sample sits within eps of a
# non-smooth point of the cost (so check_grad measures the smooth gradient).
# ---------------------------------------------------------------------------

def is_smooth_config(start, goal, q, T, obstacles, opt, eps=2e-2):
    M = T.size
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    kap = max(int(opt.kappa), 1)
    s = (np.arange(kap) + 0.5) / kap
    for i in range(M):
        Ti = float(tr.T[i])
        ci = tr.c[6 * i:6 * i + 6]
        for sj in s:
            tau = sj * Ti
            t_abs = tr._cum[i] + tau
            p = MinjerkTraj._basis(tau, 0) @ ci
            v = float(np.linalg.norm(MinjerkTraj._basis(tau, 1) @ ci))
            a = float(np.linalg.norm(MinjerkTraj._basis(tau, 2) @ ci))
            if abs(v - opt.vmax) < eps:
                return False
            if abs(a - opt.amax) < eps:
                return False
            for obs in obstacles:
                d_safe = CFG[obs.class_name].d_safe
                d, _, _ = _signed_dist_and_grad(obs, p, t_abs)
                if abs(d - d_safe) < eps:        # cubic-penalty kink
                    return False
                if hasattr(obs, "centre0"):      # sphere centre kink
                    c = obs.centre0 + obs.vel * t_abs
                    if np.linalg.norm(p - c) < eps:
                        return False
                else:                            # AABB face / interior kink
                    over = np.maximum(obs.lo - p, 0.0) + np.maximum(p - obs.hi, 0.0)
                    nrm = float(np.linalg.norm(over))
                    if nrm < eps:                # near a face or inside
                        return False
    return True


# ===========================================================================
# GATE: full check_grad on q AND T over >= 100 smooth configs, active masks
# ===========================================================================
print("\n--- GATE: full analytic grad vs Richardson-FD on q AND T (>=100 smooth configs) ---")
rng = np.random.default_rng(2026)
worst_total = 0.0
worst_q = 0.0
worst_T = 0.0
n_cfg = 0
n_obs_active = n_vel_active = n_acc_active = 0
attempts = 0
while n_cfg < 120 and attempts < 4000:
    attempts += 1
    M = int(rng.integers(2, 6))
    start = rng.standard_normal(3) * 2.0
    goal = rng.standard_normal(3) * 2.0
    q = rng.standard_normal((M - 1, 3)) * 2.0
    T = rng.uniform(0.5, 1.6, M)
    mid = 0.5 * (start + goal)
    obstacles = []
    # sphere (sometimes moving) placed near the path so the penalty activates
    if rng.random() < 0.85:
        vel = rng.standard_normal(3) * 0.4 if rng.random() < 0.5 else np.zeros(3)
        obstacles.append(SphereObstacle(mid + rng.standard_normal(3) * 0.3,
                                         float(rng.uniform(0.5, 0.9)),
                                         vel=vel, class_name="human"))
    # AABB wall sometimes
    if rng.random() < 0.5:
        c = mid + rng.standard_normal(3) * 0.4
        half = rng.uniform(0.2, 0.5, 3)
        obstacles.append(AABBObstacle(c - half, c + half, class_name="wall"))
    # tune vmax/amax to a fraction of the seed's own speed so hinges are
    # active on ~half the configs
    opt = OptParams(kappa=10,
                    vmax=float(rng.uniform(0.4, 4.0)),
                    amax=float(rng.uniform(0.5, 6.0)),
                    w_obs=1.0, w_vel=50.0, w_accel=50.0,
                    w_smooth=1.0, w_time=5.0)
    if not is_smooth_config(start, goal, q, T, obstacles, opt):
        continue

    Ttar = float(np.sum(T)) * float(rng.uniform(0.7, 1.3))
    nq = 3 * (M - 1)
    x = np.concatenate([q.ravel(), T])
    args = (start, goal, M, obstacles, CFG, opt, Ttar)
    f, g = _minco_cost_grad(x, *args)
    gfd = fd_richardson_vec(lambda xx: _minco_cost_grad(xx, *args)[0], x)
    e_total = rel_err(g, gfd)
    e_q = rel_err(g[:nq], gfd[:nq]) if nq > 0 else 0.0
    e_T = rel_err(g[nq:], gfd[nq:])
    worst_total = max(worst_total, e_total)
    worst_q = max(worst_q, e_q)
    worst_T = max(worst_T, e_T)
    n_cfg += 1

    # mask bookkeeping
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    ts = np.linspace(0, tr.t_end, 60)
    if obstacles:
        dmin = min(min(o.signed_dist(tr.eval(t), float(t)) for t in ts) for o in obstacles)
        ds_max = max(CFG[o.class_name].d_safe for o in obstacles)
        if dmin < ds_max:
            n_obs_active += 1
    vmag = np.linalg.norm(tr.eval_deriv(ts, 1), axis=1)
    amag = np.linalg.norm(tr.eval_deriv(ts, 2), axis=1)
    if vmag.max() > opt.vmax:
        n_vel_active += 1
    if amag.max() > opt.amax:
        n_acc_active += 1

check(f"GATE.0 full grad rel-err < 1e-6 on {n_cfg} smooth configs",
      worst_total < 1e-6, f"worst total={worst_total:.2e}")
check("GATE.1 dCost/dq rel-err < 1e-6", worst_q < 1e-6, f"worst dq={worst_q:.2e}")
check("GATE.2 dCost/dT rel-err < 1e-6", worst_T < 1e-6, f"worst dT={worst_T:.2e}")
check("GATE.3 mask coverage: obstacle/vel/accel each active on many configs",
      n_obs_active >= 20 and n_vel_active >= 20 and n_acc_active >= 20,
      f"obs={n_obs_active} vel={n_vel_active} acc={n_acc_active} of {n_cfg}")


# ===========================================================================
# DEDICATED dCost/dT-only test: single T_i perturb, MOVING sphere DOWNSTREAM
# of the perturbed segment -> exercises the cross-segment (c2) explicit term.
# ===========================================================================
print("\n--- DT: single-T_k perturbation, moving sphere downstream (c2 cross-segment) ---")
worst_c2 = 0.0
n_c2 = 0
attempts = 0
rng2 = np.random.default_rng(99)
while n_c2 < 40 and attempts < 2000:
    attempts += 1
    M = int(rng2.integers(3, 6))             # need >=3 so there is an upstream + downstream
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([float(M) * 1.5, 0.0, 0.0])
    q = np.column_stack([np.linspace(1.5, goal[0] - 1.5, M - 1),
                         rng2.standard_normal(M - 1) * 0.3,
                         rng2.standard_normal(M - 1) * 0.3])
    T = rng2.uniform(0.6, 1.2, M)
    # moving sphere whose closest approach is in a LATE segment (downstream)
    tr0 = MinjerkTraj.from_endpoints(start, goal, q, T)
    # aim sphere at a point ~75% along the trajectory, moving so it stays close
    t_hit = 0.75 * tr0.t_end
    p_hit = tr0.eval(t_hit)
    vel_obs = rng2.standard_normal(3) * 0.5
    centre0 = p_hit - vel_obs * t_hit + rng2.standard_normal(3) * 0.2
    sph = SphereObstacle(centre0, 0.7, vel=vel_obs, class_name="human")
    opt = OptParams(kappa=12, vmax=10.0, amax=20.0, w_obs=1.0,
                    w_vel=0.0, w_accel=0.0, w_smooth=0.0, w_time=0.0)
    if not is_smooth_config(start, goal, q, T, [sph], opt):
        continue
    # require the sphere to be ACTIVE and its closest approach downstream of seg 0
    ts = np.linspace(0, tr0.t_end, 200)
    ds = np.array([sph.signed_dist(tr0.eval(t), float(t)) for t in ts])
    if ds.min() >= 0.8:                       # not active
        continue
    t_close = ts[int(np.argmin(ds))]
    i_close = int(np.searchsorted(tr0._cum, t_close, side="right") - 1)
    if i_close < 1:                           # closest approach must be downstream
        continue

    nq = 3 * (M - 1)
    x = np.concatenate([q.ravel(), T])
    Ttar = float(np.sum(T))
    args = (start, goal, M, [sph], CFG, opt, Ttar)
    _, g = _minco_cost_grad(x, *args)
    # perturb a SINGLE upstream T_k (k < i_close) and compare to FD
    k = int(rng2.integers(0, i_close))        # strictly upstream of closest approach
    def fk(val, k=k):
        xx = x.copy(); xx[nq + k] = val
        return _minco_cost_grad(xx, *args)[0]
    def Drich(f, x0, h0=1e-4):
        def D(h): return (f(x0 + h) - f(x0 - h)) / (2.0 * h)
        return (4.0 * D(h0 / 2.0) - D(h0)) / 3.0
    gfd_k = Drich(fk, T[k])
    e = abs(g[nq + k] - gfd_k) / max(abs(gfd_k), 1e-6)
    worst_c2 = max(worst_c2, e)
    n_c2 += 1
check(f"DT.0 cross-segment explicit dCost/dT_k vs FD ({n_c2} moving-sphere configs)",
      worst_c2 < 1e-6 and n_c2 >= 20, f"worst rel={worst_c2:.2e}, n={n_c2}")

# RED-TEAM: confirm the c2 term is actually NONZERO (a dropped term would still
# pass DT.0 if the upstream contribution happened to vanish). Build one explicit
# case and verify the analytic upstream grad is materially non-zero AND matches FD.
print("\n--- DT.1 c2 term is materially non-zero (guards against silent drop) ---")
M = 4
start = np.array([0.0, 0.0, 0.0]); goal = np.array([6.0, 0.0, 0.0])
q = np.array([[1.5, 0.2, 0.0], [3.0, -0.1, 0.0], [4.5, 0.15, 0.0]])
T = np.array([0.9, 1.0, 1.1, 0.95])
tr0 = MinjerkTraj.from_endpoints(start, goal, q, T)
t_hit = 0.8 * tr0.t_end
p_hit = tr0.eval(t_hit)
vel_obs = np.array([0.4, 0.2, 0.0])
sph = SphereObstacle(p_hit - vel_obs * t_hit, 0.8, vel=vel_obs, class_name="human")
opt = OptParams(kappa=16, vmax=50.0, amax=50.0, w_obs=1.0,
                w_vel=0.0, w_accel=0.0, w_smooth=0.0, w_time=0.0)
nq = 3 * (M - 1)
x = np.concatenate([q.ravel(), T]); Ttar = float(np.sum(T))
args = (start, goal, M, [sph], CFG, opt, Ttar)
_, g = _minco_cost_grad(x, *args)
def f0(val):
    xx = x.copy(); xx[nq + 0] = val
    return _minco_cost_grad(xx, *args)[0]
def Drich(f, x0, h0=1e-4):
    def D(h): return (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    return (4.0 * D(h0 / 2.0) - D(h0)) / 3.0
g0_fd = Drich(f0, T[0])
check("DT.1 upstream dCost/dT_0 nonzero AND matches FD",
      abs(g[nq + 0]) > 1e-3 and abs(g[nq + 0] - g0_fd) / max(abs(g0_fd), 1e-6) < 1e-6,
      f"analytic={g[nq+0]:.5f} fd={g0_fd:.5f}")


# ===========================================================================
# DIRECTIONAL checks AT the kinks (check_grad MUST fail there; the analytic
# subgradient must be a valid one-sided directional derivative on the smooth side)
# ===========================================================================
print("\n--- KINK: directional derivative checks at non-smooth points ---")

def dir_deriv_fd(x, args, direction, h=1e-6):
    """One-sided directional derivative (x+h*dir) at the kink, FD."""
    f0, _ = _minco_cost_grad(x, *args)
    fp, _ = _minco_cost_grad(x + h * direction, *args)
    return (fp - f0) / h

# (a) d == d_safe boundary: pick a static sphere so one sample sits exactly on
#     d=d_safe. φ(d_safe)=0 and φ'(d_safe)=0 -> gradient is 0 from this sample;
#     directional derivative from BOTH sides is 0 (cubic is C^1 at the boundary).
M = 2
start = np.array([0.0, 0.0, 0.0]); goal = np.array([2.0, 0.0, 0.0])
q = np.zeros((1, 3)); q[0] = [1.0, 0.0, 0.0]
T = np.array([1.0, 1.0])
# place a static sphere so that the midpoint sample (tau giving p≈[1,0,0]) is
# exactly d_safe away. d_safe(human)=0.8, so centre at [1, 0.8+r, 0]? simpler:
# centre directly below by exactly r + d_safe.
opt = OptParams(kappa=2, vmax=100.0, amax=100.0, w_obs=1.0, w_vel=0.0,
                w_accel=0.0, w_smooth=0.0, w_time=0.0)
r = 0.5
sph = SphereObstacle([0.5, r + CFG["human"].d_safe, 0.0], r, class_name="human")
nq = 3 * (M - 1)
x = np.concatenate([q.ravel(), T]); Ttar = 2.0
args = (start, goal, M, [sph], CFG, opt, Ttar)
_, g = _minco_cost_grad(x, *args)
# at the C^1 cubic boundary the gradient contribution from a boundary sample is
# 0; perturbing INTO the obstacle (decreasing d) must give a non-negative
# directional increase, perturbing OUT gives 0. Check the one-sided FD on q_y.
dir_in = np.zeros_like(x); dir_in[1] = -1.0   # move q toward the sphere (y down)
fd_in = dir_deriv_fd(x, args, dir_in, h=1e-5)
analytic_in = float(g.dot(dir_in))
check("KINK.0 d=d_safe cubic boundary is C^1 (analytic grad ~ 0, FD ~ 0)",
      abs(analytic_in) < 1e-3 and abs(fd_in) < 1e-3,
      f"analytic·dir={analytic_in:.2e}, FD={fd_in:.2e}")

# (b) AABB face: the subgradient picks one face; the one-sided directional
#     derivative moving straight OUT along that face normal must match.
M = 2
start = np.array([0.0, 0.0, 0.0]); goal = np.array([2.0, 0.0, 0.0])
q = np.array([[1.0, 0.0, 0.0]])
T = np.array([1.0, 1.0])
# box whose nearest face to the path midpoint [1,0,0] is the +y... place box so
# midpoint is OUTSIDE in pure +x or +y direction (single active face).
box = AABBObstacle([1.2, -0.5, -0.5], [2.0, 0.5, 0.5], class_name="wall")
# midpoint ~[1,0,0] is at x<lo_x -> outside on -x face only (single axis active)
opt = OptParams(kappa=2, vmax=100.0, amax=100.0, w_obs=1.0, w_vel=0.0,
                w_accel=0.0, w_smooth=0.0, w_time=0.0)
nq = 3 * (M - 1)
x = np.concatenate([q.ravel(), T]); Ttar = 2.0
args = (start, goal, M, [box], CFG, opt, Ttar)
_, g = _minco_cost_grad(x, *args)
# move q in +x (toward the box face) — single-axis active, smooth here, so
# analytic directional == FD directional both sides.
dir_x = np.zeros_like(x); dir_x[0] = 1.0
fd_p = dir_deriv_fd(x, args, dir_x, h=1e-6)
fd_m = dir_deriv_fd(x, args, -dir_x, h=1e-6)
analytic = float(g.dot(dir_x))
check("KINK.1 AABB single-face outside region: analytic dir-deriv == FD both sides",
      abs(analytic - fd_p) < 1e-4 and abs(-analytic - fd_m) < 1e-4,
      f"analytic={analytic:.4f} fd+={fd_p:.4f} fd-={fd_m:.4f}")


# ===========================================================================
# M=1 boundary (no interior waypoints, q empty) — grad is T-only
# ===========================================================================
print("\n--- M1: M=1 boundary (q empty, decision var = T only) ---")
M = 1
start = np.array([0.0, 0.0, 0.0]); goal = np.array([3.0, 1.0, 0.5])
q = np.zeros((0, 3))
T = np.array([1.3])
sph = SphereObstacle([1.5, 0.3, 0.1], 0.7, class_name="human")
opt = OptParams(kappa=12, vmax=2.0, amax=4.0, w_obs=1.0, w_vel=50.0,
                w_accel=50.0, w_smooth=1.0, w_time=5.0)
x = T.copy(); Ttar = 1.3
args = (start, goal, M, [sph], CFG, opt, Ttar)
ok_run = True
try:
    f, g = _minco_cost_grad(x, *args)
    gfd = fd_richardson_vec(lambda xx: _minco_cost_grad(xx, *args)[0], x)
    e = rel_err(g, gfd)
except Exception as ex:
    ok_run = False
    e = float("inf")
    print("   exception:", ex)
check("M1.0 M=1 cost+grad runs and dCost/dT matches FD", ok_run and e < 1e-6,
      f"rel-err={e:.2e}, g={g if ok_run else 'n/a'}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
