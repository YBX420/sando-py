"""Stage 3 — analytic-gradient GATE for the per-class HARD (ALM) cost.

THE GATE (acceptance spec):
  - check_grad of the INNER augmented-Lagrangian gradient w.r.t. BOTH q and T,
    at a FIXED (lambda, rho, frozen-normals), rel-err < 1e-5 on >= 100 random
    configs with an ACTIVE hard constraint, EXCLUDING kink configs (||P-c||~0,
    z_m = lambda/rho + g ~ 0 penalty seam, AABB interior).
  - a DEDICATED drop-the-explicit-T test: zeroing the explicit dP/dT_i piece of
    the chain must BREAK the gate (proves the C2B/T^j reparam term is load-bearing,
    the same trap as M2's cross-segment c2 drop).
  - DIRECTIONAL FD checks AT the penalty seam (z_m ~ 0): the max(0,.) kink — the
    analytic subgradient must be a valid one-sided directional derivative.

FD is Richardson-extrapolated central difference (O(h^4)).

Run:  python3 test/stage3_minco_perclass_grad.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (                                    # noqa: E402
    _minco_cost_grad, _seg_normals, _partition_obstacles, _alm_constraints,
    _trust_mask, _certificate_margin_spacetime, hard_clearance, OptParams)
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
    a = np.asarray(a, float); b = np.asarray(b, float)
    num = float(np.linalg.norm(a - b))
    den = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-6)
    return num / den


def fd_richardson_vec(f, x, h0=1e-4):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = 1.0
        def D(h):
            return (f(x + h * e) - f(x - h * e)) / (2.0 * h)
        d1 = D(h0); d2 = D(h0 / 2.0)
        g[i] = (4.0 * d2 - d1) / 3.0
    return g


def alm_smooth_config(tr, hard_obs, normals, lam, rho, eps=2e-2):
    """Reject a config near a kink of the ALM penalty: a control point on the
    sphere centre (||P-c||~0) OR a constraint at the max(0,.) seam (z_m~0)."""
    cons, P = _alm_constraints(tr, hard_obs, CFG, normals=normals)
    g = cons["g"]
    z = lam + rho * g
    if np.any(np.abs(z) < eps * max(rho, 1.0)):
        return False
    # sphere-centre proximity (||P-c||~0 makes the halfspace dist degenerate)
    for i in range(tr.M):
        t_rep = float(tr._cum[i + 1])
        for k in range(6):
            for obs in hard_obs:
                if hasattr(obs, "centre0"):
                    c = obs.predict(t_rep)
                    if np.linalg.norm(P[i, k] - c) < eps:
                        return False
                else:
                    over = (np.maximum(obs.lo - P[i, k], 0.0)
                            + np.maximum(P[i, k] - obs.hi, 0.0))
                    if float(np.linalg.norm(over)) < eps:
                        return False
    return True


# ===========================================================================
# GATE: inner ALM grad on q AND T over >= 100 active+smooth configs
# ===========================================================================
print("\n--- GATE: ALM inner grad vs Richardson-FD on q AND T (>=100 active configs) ---")
rng = np.random.default_rng(20260529)
worst_total = worst_q = worst_T = 0.0
n_cfg = 0
n_active = 0
attempts = 0
opt = OptParams(kappa=8, vmax=80.0, amax=120.0,
                w_obs=0.0, w_vel=0.0, w_accel=0.0, w_smooth=0.0, w_time=0.0)
while n_cfg < 120 and attempts < 6000:
    attempts += 1
    M = int(rng.integers(2, 6))
    start = rng.standard_normal(3) * 2.0
    goal = rng.standard_normal(3) * 2.0
    q = rng.standard_normal((M - 1, 3)) * 2.0
    T = rng.uniform(0.5, 1.6, M)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    mid = tr.eval(0.5 * tr.t_end)
    # place a sphere near the path so the halfspace constraint activates
    obs_list = [SphereObstacle(mid + rng.standard_normal(3) * 0.3,
                               float(rng.uniform(0.4, 0.9)), class_name="human")]
    # occasionally add a hard AABB (all-hard ablation arm) to exercise that branch
    if rng.random() < 0.3:
        c = mid + rng.standard_normal(3) * 0.5
        half = rng.uniform(0.2, 0.5, 3)
        obs_list.append(AABBObstacle(c - half, c + half, class_name="human"))
    soft, hard = _partition_obstacles(obs_list, CFG, "hard")  # all-hard ablation
    normals = _seg_normals(tr, hard)
    Nc = M * 6 * len(hard)
    lam = rng.uniform(0.0, 4.0, Nc)
    rho = float(rng.uniform(5.0, 60.0))
    if not alm_smooth_config(tr, hard, normals, lam, rho):
        continue
    cons, _ = _alm_constraints(tr, hard, CFG, normals=normals)
    if not np.any(lam + rho * cons["g"] > 0.0):   # require an ACTIVE constraint
        continue

    nq = 3 * (M - 1)
    x = np.concatenate([q.ravel(), T])
    Ttar = float(np.sum(T))
    kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals)
    def f(xx, kw=kw):
        return _minco_cost_grad(xx, start, goal, M, soft, CFG, opt, Ttar, **kw)[0]
    _, g = _minco_cost_grad(x, start, goal, M, soft, CFG, opt, Ttar, **kw)
    gfd = fd_richardson_vec(f, x)
    worst_total = max(worst_total, rel_err(g, gfd))
    if nq > 0:
        worst_q = max(worst_q, rel_err(g[:nq], gfd[:nq]))
    worst_T = max(worst_T, rel_err(g[nq:], gfd[nq:]))
    n_cfg += 1
    n_active += 1

check(f"GATE.0 ALM inner grad rel-err < 1e-5 on {n_cfg} active configs",
      worst_total < 1e-5 and n_cfg >= 100, f"worst total={worst_total:.2e}, n={n_cfg}")
check("GATE.1 dCost/dq rel-err < 1e-5", worst_q < 1e-5, f"worst dq={worst_q:.2e}")
check("GATE.2 dCost/dT rel-err < 1e-5", worst_T < 1e-5, f"worst dT={worst_T:.2e}")
check("GATE.3 all gated configs had an ACTIVE constraint", n_active == n_cfg,
      f"active={n_active}/{n_cfg}")


# ===========================================================================
# DROP-EXPLICIT-T ablation: zeroing dP/dT_i must BREAK the T gradient gate.
# ===========================================================================
print("\n--- DROP-T: explicit dP/dT_i is load-bearing (gate FAILS without it) ---")
M = 3
start = np.array([0.0, 0.0, 0.0]); goal = np.array([4.5, 0.4, 0.2])
q = np.array([[1.5, 0.2, 0.1], [3.0, -0.1, 0.15]])
T = np.array([0.8, 1.0, 0.9])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
mid = tr.eval(0.5 * tr.t_end)
human = SphereObstacle(mid + np.array([0.0, 0.3, 0.0]), 0.6, class_name="human")
soft, hard = _partition_obstacles([human], CFG, None)
normals = _seg_normals(tr, hard)
lam = np.full(M * 6, 2.0); rho = 30.0
optd = OptParams(kappa=8, vmax=80, amax=120, w_obs=0, w_vel=0, w_accel=0,
                 w_smooth=0, w_time=0)
nq = 3 * (M - 1); x = np.concatenate([q.ravel(), T]); Ttar = float(np.sum(T))
kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals)
def fall(xx):
    return _minco_cost_grad(xx, start, goal, M, soft, CFG, optd, Ttar, **kw)[0]
_, g_with = _minco_cost_grad(x, start, goal, M, soft, CFG, optd, Ttar, **kw)
gfd = fd_richardson_vec(fall, x)
err_with_T = rel_err(g_with[nq:], gfd[nq:])

orig = MinjerkTraj.control_points_dT_explicit
MinjerkTraj.control_points_dT_explicit = lambda self: np.zeros((self.M, 6, 3))
try:
    _, g_drop = _minco_cost_grad(x, start, goal, M, soft, CFG, optd, Ttar, **kw)
finally:
    MinjerkTraj.control_points_dT_explicit = orig
err_drop_T = rel_err(g_drop[nq:], gfd[nq:])

check("DROP-T.0 WITH explicit-T: dCost/dT matches FD (< 1e-5)", err_with_T < 1e-5,
      f"rel-err={err_with_T:.2e}")
check("DROP-T.1 WITHOUT explicit-T: gate FAILS (rel-err >> 1e-5) => load-bearing",
      err_drop_T > 1e-2, f"rel-err={err_drop_T:.2e} (with-T was {err_with_T:.2e})")


# ===========================================================================
# KINK: the PHR penalty p(z)=(1/2rho)max(0,z)^2 has p'(z)=max(0,z)/rho which is
# CONTINUOUS at z=0 -> the cost is C^1 at the max(0,.) seam (the kink is only in
# the 2nd derivative). So a config with a constraint EXACTLY at z_m=0 is still a
# valid check_grad point: central FD of the FULL gradient matches analytic.
# This is the precise statement of why the seam does not break the GATE.
# ===========================================================================
print("\n--- KINK: PHR penalty is C^1 at the max(0,.) seam (z_m=0 config) ---")
M = 2
start = np.array([0.0, 0.0, 0.0]); goal = np.array([3.0, 0.0, 0.0])
q = np.array([[1.5, 0.0, 0.0]])
T = np.array([1.0, 1.0])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
human = SphereObstacle([1.5, 1.0, 0.0], 0.3, class_name="human")  # off to +y
soft, hard = _partition_obstacles([human], CFG, None)
normals = _seg_normals(tr, hard)
cons, _ = _alm_constraints(tr, hard, CFG, normals=normals)
g = cons["g"]; m_star = int(np.argmax(g))     # most-violated constraint
rho = 20.0
lam = np.zeros(g.size)
lam[m_star] = -rho * g[m_star]                # force z_{m_star} = 0 exactly (seam)
optd = OptParams(kappa=8, vmax=80, amax=120, w_obs=0, w_vel=0, w_accel=0,
                 w_smooth=0, w_time=0)
nq = 3 * (M - 1); x = np.concatenate([q.ravel(), T]); Ttar = 2.0
kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals)
_, gk = _minco_cost_grad(x, start, goal, M, soft, CFG, optd, Ttar, **kw)
def fseam(xx):
    return _minco_cost_grad(xx, start, goal, M, soft, CFG, optd, Ttar, **kw)[0]
# confirm we really sit ON the seam
z_seam = lam[m_star] + rho * g[m_star]
gfd = fd_richardson_vec(fseam, x)
err_seam = rel_err(gk, gfd)
check("KINK.0 sits exactly on the max(0,.) seam (z_{m*} = 0)",
      abs(z_seam) < 1e-9, f"z_seam={z_seam:.2e}")
check("KINK.1 C^1 seam: central-FD of FULL gradient matches analytic (< 1e-4)",
      err_seam < 1e-4, f"rel-err at seam={err_seam:.2e}")


# ===========================================================================
# TIGHT: the halfspace certificate is TIGHT — the per-control-point BALL
# relaxation is NOT (it can overstate clearance, leaking between control pts).
# This is WHY Stage 3 ships the supporting-halfspace form (the design risk #1).
# A deliberately V-shaped control polygon: the degree-5 Bezier curve cuts the
# corner and dips CLOSER to c than any single control point.
# ===========================================================================
print("\n--- TIGHT: halfspace certificate vs the (leaky) ball relaxation ---")
from math import comb as _C                                              # noqa: E402
c = np.array([0.0, 0.0, 0.0])
P_v = np.array([[-1.5, 0.2, 0.0], [-0.9, 1.4, 0.0], [-0.3, 2.0, 0.0],
                [0.3, 2.0, 0.0], [0.9, 1.4, 0.0], [1.5, 0.2, 0.0]])
def _bern(t):
    return np.array([_C(5, k) * t**k * (1 - t)**(5 - k) for k in range(6)])
ts = np.linspace(0, 1, 2000)
curve_min = min(np.linalg.norm(_bern(t) @ P_v - c) for t in ts)
ball_min = min(float(np.linalg.norm(p - c)) for p in P_v)
a = (P_v.mean(axis=0) - c); a /= np.linalg.norm(a)
hs_min = min(float(a.dot(p - c)) for p in P_v)
check("TIGHT.0 halfspace projection lower-bounds the true min curve distance",
      hs_min <= curve_min + 1e-9, f"hs_min={hs_min:.4f} <= curve_dmin={curve_min:.4f}")
check("TIGHT.1 the BALL relaxation OVERSTATES clearance (leaks between ctrl pts)",
      ball_min > curve_min + 1e-3,
      f"ball_min={ball_min:.4f} > curve_dmin={curve_min:.4f} "
      f"(leak={ball_min - curve_min:.4f})")


# ===========================================================================
# S4a/S4b GATE: SPACE-TIME ALM with a MOVING human. The new dCost/dT Source 2
# (dc_h/dt * dt_{i,k}/dT chained: same-segment k/5 + cross-segment reverse-cumsum)
# is exercised ONLY here -- a vel=0 gate silently never tests it (M2-trap blind
# spot). M>=3 with the constraint active across MULTIPLE segments is mandatory:
# the WRONG i-1-only-neighbor chain accidentally matches complex-step on lucky
# draws but fails the multi-segment reverse-cumsum truth. Normals AND the trust
# mask are FROZEN inside the eval (else the cost is nonsmooth in T).
# ===========================================================================
print("\n--- S4: SPACE-TIME ALM moving-human grad (Source 2 dc/dt-chain) ---")
rng = np.random.default_rng(20260601)
worst_mv = worst_mv_T = 0.0
n_mv = 0
saw_split = 0
attempts = 0
optmv = OptParams(kappa=8, vmax=80.0, amax=120.0,
                  w_obs=0.0, w_vel=0.0, w_accel=0.0, w_smooth=0.0, w_time=0.0,
                  spacetime_hard=True)
while n_mv < 100 and attempts < 6000:
    attempts += 1
    M = int(rng.integers(3, 6))                    # M>=3: cross-segment chain live
    start = rng.standard_normal(3) * 2.0
    goal = rng.standard_normal(3) * 2.0
    q = rng.standard_normal((M - 1, 3)) * 2.0
    T = rng.uniform(0.5, 1.2, M)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    mid = tr.eval(0.4 * tr.t_end)
    vel = np.array([-0.7, 0.3, 0.0]) + rng.standard_normal(3) * 0.3
    human = SphereObstacle(mid + rng.standard_normal(3) * 0.3,
                           float(rng.uniform(0.4, 0.9)), vel=vel, class_name="human")
    # half the draws use a finite tau_trust to exercise the S4b hard/soft split
    optmv.tau_trust = 1.0e9 if (n_mv % 2 == 0) else float(rng.uniform(0.7, 1.6))
    soft, hard = _partition_obstacles([human], CFG, "hard")
    normals = _seg_normals(tr, hard, opt=optmv)
    tmask = _trust_mask(tr, optmv)
    Nc = M * 6 * len(hard)
    lam = rng.uniform(0.0, 4.0, Nc)
    rho = float(rng.uniform(5.0, 60.0))
    cons, _ = _alm_constraints(tr, hard, CFG, normals=normals, opt=optmv,
                               trust_mask=tmask)
    z = lam + rho * cons["g"]
    if not np.any(z > 0.0):                          # require an ACTIVE constraint
        continue
    # reject seam-proximity on the TRUSTED rows only (untrusted have g=-1e9)
    zt = z[cons["trust"]]
    if zt.size and np.any(np.abs(zt) < 2e-2 * max(rho, 1.0)):
        continue
    if not tmask.all():
        saw_split += 1
    nq = 3 * (M - 1)
    x = np.concatenate([q.ravel(), T])
    Ttar = float(np.sum(T))
    kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals,
              alm_trust_mask=tmask)
    def fmv(xx, kw=kw):
        return _minco_cost_grad(xx, start, goal, M, soft, CFG, optmv, Ttar, **kw)[0]
    _, gmv = _minco_cost_grad(x, start, goal, M, soft, CFG, optmv, Ttar, **kw)
    gfd = fd_richardson_vec(fmv, x)
    worst_mv = max(worst_mv, rel_err(gmv, gfd))
    worst_mv_T = max(worst_mv_T, rel_err(gmv[nq:], gfd[nq:]))
    n_mv += 1

check(f"S4.0 moving-human space-time grad rel-err < 1e-5 on {n_mv} active configs",
      worst_mv < 1e-5 and n_mv >= 100, f"worst total={worst_mv:.2e}, n={n_mv}")
check("S4.1 moving-human dCost/dT (Source 2 chain) rel-err < 1e-5",
      worst_mv_T < 1e-5, f"worst dT={worst_mv_T:.2e}")
check("S4.2 exercised the S4b hard/soft trust split", saw_split > 0,
      f"splits={saw_split}")

# DROP Source 2: zero dgdt -> the moving-human T gradient must BREAK (load-bearing)
print("\n--- S4 DROP-dgdt: the dc/dt absolute-time chain is load-bearing ---")
M = 4
start = np.zeros(3); goal = np.array([6.0, 0.5, 0.2])
q = np.array([[1.5, 0.3, 0.0], [3.0, -0.2, 0.1], [4.5, 0.1, 0.05]])
T = np.array([0.8, 0.9, 0.7, 0.85])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
mid = tr.eval(0.4 * tr.t_end)
human = SphereObstacle(mid + np.array([0.0, 0.3, 0.0]), 0.6,
                       vel=[-0.7, 0.3, 0.0], class_name="human")
optmv2 = OptParams(kappa=8, vmax=80, amax=120, w_obs=0, w_vel=0, w_accel=0,
                   w_smooth=0, w_time=0, tau_trust=1.0e9, spacetime_hard=True)
soft, hard = _partition_obstacles([human], CFG, "hard")
normals = _seg_normals(tr, hard, opt=optmv2)
tmask = _trust_mask(tr, optmv2)
nq = 3 * (M - 1); x = np.concatenate([q.ravel(), T]); Ttar = float(np.sum(T))
lam = np.full(M * 6, 2.0); rho = 30.0
kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals,
          alm_trust_mask=tmask)
def fmv2(xx):
    return _minco_cost_grad(xx, start, goal, M, soft, CFG, optmv2, Ttar, **kw)[0]
_, g_with = _minco_cost_grad(x, start, goal, M, soft, CFG, optmv2, Ttar, **kw)
gfd = fd_richardson_vec(fmv2, x)
err_with = rel_err(g_with[nq:], gfd[nq:])
import sando_py.local.local_opt as _LO
_orig_ac = _LO._alm_constraints
def _zero_dgdt(*a, **k):
    cons, P = _orig_ac(*a, **k); cons["dgdt"] = np.zeros_like(cons["dgdt"]); return cons, P
_LO._alm_constraints = _zero_dgdt
try:
    _, g_drop = _minco_cost_grad(x, start, goal, M, soft, CFG, optmv2, Ttar, **kw)
finally:
    _LO._alm_constraints = _orig_ac
err_drop = rel_err(g_drop[nq:], gfd[nq:])
check("S4.3 WITH dgdt: moving-human dCost/dT matches FD (< 1e-5)", err_with < 1e-5,
      f"rel-err={err_with:.2e}")
check("S4.4 WITHOUT dgdt: gate FAILS (>> 1e-5) => dc/dt chain load-bearing",
      err_drop > 1e-2, f"rel-err={err_drop:.2e} (with was {err_with:.2e})")


# ===========================================================================
# S4 CERTIFICATE SOUNDNESS: the inflated-radius single-halfspace continuous-time
# certificate (R_i = r + d_safe + ||vel||*T_i) must imply ZERO dense breach at
# the TRUE moving-obstacle curve times; a naive un-inflated per-node check leaks.
# ===========================================================================
print("\n--- S4 CERT: inflated continuous-time certificate is SOUND (per-node leaks) ---")
rng = np.random.default_rng(424242)
n_sound = 0; worst_falsesafe = float("-inf"); uninfl_leak = float("-inf")
optc = OptParams(tau_trust=1.0e9, spacetime_hard=True)
for _ in range(3000):
    M = int(rng.integers(2, 5))
    start = rng.standard_normal(3) * 2.0; goal = rng.standard_normal(3) * 2.0
    q = rng.standard_normal((M - 1, 3)) * 2.0; T = rng.uniform(0.4, 1.0, M)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    mid = tr.eval(0.5 * tr.t_end)
    vel = rng.standard_normal(3) * rng.uniform(0.5, 3.0)
    human = SphereObstacle(mid + rng.standard_normal(3) * 0.6,
                           float(rng.uniform(0.3, 0.7)), vel=vel, class_name="human")
    soft, hard = _partition_obstacles([human], CFG, "hard")
    tmask = _trust_mask(tr, optc)
    cm = _certificate_margin_spacetime(tr, hard, CFG, optc, trust_mask=tmask)
    _, mb = hard_clearance(tr, hard, CFG, K_eval=2000)
    if cm >= 0:
        n_sound += 1
        worst_falsesafe = max(worst_falsesafe, mb)
    P = tr.control_points(); cent = P.mean(axis=1)
    t0 = tr._cum[:M]; c0 = human.centre0[None, :] + human.vel[None, :] * t0[:, None]
    a = cent - c0; a /= np.linalg.norm(a, axis=1, keepdims=True)
    proj = np.einsum("md,mkd->mk", a, P - c0[:, None, :])
    Rn = human.radius + 0.8                          # NO inflation -> leaky
    if float(np.min(proj - Rn)) >= 0:
        uninfl_leak = max(uninfl_leak, mb)
check("S4.5 inflated cert>=0 implies dense max_breach<=0 (continuous-time sound)",
      n_sound > 0 and worst_falsesafe <= 1e-6,
      f"n_sound={n_sound}, worst false-safe breach={worst_falsesafe:.3f}")
check("S4.6 un-inflated per-node check LEAKS (proves inflation necessary)",
      uninfl_leak > 1e-3, f"worst leak breach={uninfl_leak:.3f}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
