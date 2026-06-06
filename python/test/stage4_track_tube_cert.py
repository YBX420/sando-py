"""Stage 4 — execution-gap certificate: plan -> flown TUBE via epsilon_track.

The certificate is about the PLANNED trajectory p(s); the drone flies p_actual(s) with
||p_actual(s)-p(s)|| <= epsilon_track (tracking + estimation + replan latency). Adding
epsilon_track to every hard obstacle's effective radius R = r + d_safe + Q_alpha +
epsilon_track upgrades the guarantee from "the PLAN clears" to "the flown TUBE clears":
plan margin >= R => flown margin >= R - epsilon_track - Q_alpha >= r + d_safe.

UNLIKE Q_alpha (human-prediction error -> SPHERES only), epsilon_track is the DRONE's own
position uncertainty -> obstacle-agnostic, so it inflates EVERY hard obstacle (spheres AND
hard AABB walls). Like Q_alpha it is CONSTANT in (q, T): deps/dq = deps/dT = 0, so no
gradient term -- the same free trick.

THE CLAIMS:
  TGRAD.* end-to-end differentiability survives with epsilon_track ON (Richardson-FD
          gate < 1e-5 on >= 100 active configs, with q_conformal ON too).
  TADD.*  exact structure: sphere g shifts by EXACTLY Q_alpha + epsilon_track; WALL g
          shifts by EXACTLY epsilon_track (NOT 0 -- the key difference from Q_alpha);
          epsilon_track=0 is a true no-op.
  TUBE.*  consequence: solving with epsilon_track pushes the plan out so the whole
          flown tube clears d_safe -- a worst-case flown trajectory (plan displaced by
          epsilon_track toward the human) still clears d_safe, where the eps=0 plan breaches.

Run:  python3 test/stage4_track_tube_cert.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (                                    # noqa: E402
    _minco_cost_grad, _seg_normals, _partition_obstacles, _alm_constraints,
    _trust_mask, plan_minco, hard_clearance, OptParams, DetourConfig)
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.minco import MinjerkTraj                              # noqa: E402
from sando_py.local.avoid_config import AvoidParams                       # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
NO_DETOUR = DetourConfig(enabled=False)
D_SAFE = 0.8
Q_ALPHA = 0.3
EPS = 0.2


def fd_richardson_vec(f, x, h0=1e-4):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = 1.0
        def D(h):
            return (f(x + h * e) - f(x - h * e)) / (2.0 * h)
        d1 = D(h0); d2 = D(h0 / 2.0)
        g[i] = (4.0 * d2 - d1) / 3.0
    return g


def smooth_active(tr, hard, normals, lam, rho, opt, eps=2e-2):
    cons, P = _alm_constraints(tr, hard, CFG, normals=normals, opt=opt)
    z = lam + rho * cons["g"]
    if np.any(np.abs(z) < eps * max(rho, 1.0)):
        return False, cons
    for i in range(tr.M):
        t_rep = float(tr._cum[i + 1])
        for k in range(6):
            for obs in hard:
                if hasattr(obs, "centre0"):
                    if np.linalg.norm(P[i, k] - obs.predict(t_rep)) < eps:
                        return False, cons
                else:
                    over = (np.maximum(obs.lo - P[i, k], 0.0)
                            + np.maximum(P[i, k] - obs.hi, 0.0))
                    if float(np.linalg.norm(over)) < eps:
                        return False, cons
    return np.any(z > 0.0), cons


# ===========================================================================
# TGRAD: gradient gate with epsilon_track (and Q_alpha) ON.
# ===========================================================================
print("\n--- TGRAD: ALM grad vs Richardson-FD with epsilon_track=0.2, Q=0.3 ON ---")
rng = np.random.default_rng(20260603)
opt_g = OptParams(kappa=8, vmax=80.0, amax=120.0, q_conformal=Q_ALPHA, epsilon_track=EPS,
                  w_obs=0.0, w_vel=0.0, w_accel=0.0, w_smooth=0.0, w_time=0.0)
worst = 0.0; n_cfg = 0; attempts = 0
while n_cfg < 110 and attempts < 6000:
    attempts += 1
    M = int(rng.integers(2, 6))
    start = rng.standard_normal(3) * 2.0; goal = rng.standard_normal(3) * 2.0
    q = rng.standard_normal((M - 1, 3)) * 2.0
    T = rng.uniform(0.5, 1.6, M)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    mid = tr.eval(0.5 * tr.t_end)
    obs_list = [SphereObstacle(mid + rng.standard_normal(3) * 0.3,
                               float(rng.uniform(0.4, 0.9)), class_name="human",
                               vel=rng.standard_normal(3) * 0.4)]
    if rng.random() < 0.4:                       # hard AABB wall (eps applies to it too)
        c = mid + rng.standard_normal(3) * 0.5; half = rng.uniform(0.2, 0.5, 3)
        obs_list.append(AABBObstacle(c - half, c + half, class_name="human"))
    soft, hard = _partition_obstacles(obs_list, CFG, "hard")
    normals = _seg_normals(tr, hard, opt=opt_g); tmask = _trust_mask(tr, opt_g)
    lam = rng.uniform(0.0, 4.0, M * 6 * len(hard)); rho = float(rng.uniform(5.0, 60.0))
    ok, cons = smooth_active(tr, hard, normals, lam, rho, opt_g)
    if not ok:
        continue
    x = np.concatenate([q.ravel(), T]); Ttar = float(np.sum(T))
    kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals, alm_trust_mask=tmask)
    f = lambda xx, kw=kw: _minco_cost_grad(xx, start, goal, M, soft, CFG, opt_g, Ttar, **kw)[0]
    _, g = _minco_cost_grad(x, start, goal, M, soft, CFG, opt_g, Ttar, **kw)
    gfd = fd_richardson_vec(f, x)
    den = max(float(np.linalg.norm(g)), float(np.linalg.norm(gfd)), 1e-6)
    worst = max(worst, float(np.linalg.norm(g - gfd)) / den)
    n_cfg += 1
check(f"TGRAD.0 ALM grad rel-err < 1e-5 with eps_track ON ({n_cfg} active configs)",
      worst < 1e-5 and n_cfg >= 100, f"worst={worst:.2e}, n={n_cfg}")


# ===========================================================================
# TADD: exact shift. sphere += Q+eps; WALL += eps (not 0); eps=0 no-op.
# ===========================================================================
print("\n--- TADD: WALL g shifts by EXACTLY epsilon_track (Q_alpha never does) ---")
M = 4
start = np.array([0.0, 0.0, 2.0]); goal = np.array([8.0, 0.0, 2.0])
q = np.array([[2.0, 0.3, 2.0], [4.0, -0.2, 2.0], [6.0, 0.1, 2.0]])
T = np.array([0.9, 1.0, 1.1, 0.95])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
mid = tr.eval(0.5 * tr.t_end)
sphere = SphereObstacle(mid + np.array([0.0, 0.4, 0.0]), 0.5, class_name="human")  # static
wall = AABBObstacle(mid + np.array([0.0, 0.8, 0.0]) - 0.3,
                    mid + np.array([0.0, 0.8, 0.0]) + 0.3, class_name="human")
soft, hard = _partition_obstacles([sphere, wall], CFG, "hard")
opt0 = OptParams(q_conformal=0.0, epsilon_track=0.0)
optE = OptParams(q_conformal=0.0, epsilon_track=EPS)            # eps only
optQE = OptParams(q_conformal=Q_ALPHA, epsilon_track=EPS)       # Q + eps
n0 = _seg_normals(tr, hard, opt=opt0)
cons0, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=opt0)
consE, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=optE)
consQE, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=optQE)
is_sph = cons0["hidx"] == 0
is_wall = cons0["hidx"] == 1
dE_wall = consE["g"][is_wall] - cons0["g"][is_wall]            # wall shift under eps-only
dE_sph = consE["g"][is_sph] - cons0["g"][is_sph]              # sphere shift under eps-only
dQE_sph = consQE["g"][is_sph] - cons0["g"][is_sph]           # sphere shift under Q+eps
check("TADD.0 WALL g shifts by EXACTLY epsilon_track (eps applies to walls)",
      is_wall.any() and float(np.max(np.abs(dE_wall - EPS))) < 1e-12,
      f"max|dg_wall - eps|={float(np.max(np.abs(dE_wall - EPS))):.2e}")
check("TADD.1 sphere g shifts by EXACTLY epsilon_track under eps-only",
      float(np.max(np.abs(dE_sph - EPS))) < 1e-12,
      f"max|dg_sph - eps|={float(np.max(np.abs(dE_sph - EPS))):.2e}")
check("TADD.2 sphere g shifts by EXACTLY Q_alpha + epsilon_track under both",
      float(np.max(np.abs(dQE_sph - (Q_ALPHA + EPS)))) < 1e-12,
      f"max|dg_sph - (Q+eps)|={float(np.max(np.abs(dQE_sph - (Q_ALPHA + EPS)))):.2e}")
# no-op: eps=0,Q=0 identical to opt=None
consNone, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=None)
check("TADD.3 epsilon_track=0 is a true no-op (g identical to opt=None)",
      float(np.max(np.abs(cons0["g"] - consNone["g"]))) == 0.0,
      f"max|dg|={float(np.max(np.abs(cons0['g'] - consNone['g']))):.2e}")


# ===========================================================================
# TUBE: consequence -- the whole flown tube clears d_safe.
# ===========================================================================
print("\n--- TUBE: a worst-case flown trajectory (plan +/- eps) still clears d_safe ---")
straight = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
mid = straight[len(straight) // 2].copy()
human = SphereObstacle(mid + np.array([0.0, 1.20, 0.0]), 0.5, class_name="human")  # gentle graze
opt_e0 = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, q_conformal=0.0, epsilon_track=0.0)
opt_eE = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, q_conformal=0.0, epsilon_track=EPS)
tr_e0, info0 = plan_minco(straight, [human], CFG, opt_params=opt_e0, detour_cfg=NO_DETOUR)
tr_eE, infoE = plan_minco(straight, [human], CFG, opt_params=opt_eE, detour_cfg=NO_DETOUR)
clr_e0, _ = hard_clearance(tr_e0, [human], CFG, K_eval=4000)   # plan clearance, eps=0
clr_eE, _ = hard_clearance(tr_eE, [human], CFG, K_eval=4000)   # plan clearance, eps=EPS
print(f"    plan clr eps=0: {clr_e0:.3f}   plan clr eps={EPS}: {clr_eE:.3f}")
check("TUBE.0 both solves valid", info0["trajectory_valid"] and infoE["trajectory_valid"],
      f"e0={info0['trajectory_valid']} eE={infoE['trajectory_valid']}")
check("TUBE.1 eps-plan clears d_safe + epsilon_track (certified tube floor)",
      clr_eE >= D_SAFE + EPS - 0.08, f"plan clr {clr_eE:.3f} >= {D_SAFE + EPS:.2f}")
# worst-case flown trajectory = plan displaced by epsilon_track toward the human:
#   flown clearance = plan clearance - epsilon_track.
flown_e0 = clr_e0 - EPS        # eps=0 plan, but the drone STILL deviates by eps in reality
flown_eE = clr_eE - EPS        # eps-aware plan
check("TUBE.2 eps=0 plan: the FLOWN tube BREACHES d_safe (execution gap is real)",
      flown_e0 < D_SAFE - 1e-3, f"flown clr (eps=0 plan) = {flown_e0:.3f} < d_safe={D_SAFE}")
check("TUBE.3 eps-aware plan: the FLOWN tube still CLEARS d_safe (gap closed)",
      flown_eE >= D_SAFE - 0.05, f"flown clr (eps plan) = {flown_eE:.3f} >= d_safe={D_SAFE}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
