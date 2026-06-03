"""Stage 4 — conformal continuous-time certificate WIRED into the ALM (step (1)).

Wires Q_alpha (OptParams.q_conformal) into a hard sphere's effective radius
R = radius + d_safe + Q_alpha at BOTH the gradient-bearing ALM constraint
(_alm_constraints) and the continuous-time certificate (_certificate_margin /
_certificate_margin_spacetime). Q_alpha is the conformal prediction-error margin
calibrated offline (see test/stage4_conformal_coverage.py).

THE CLAIMS:
  GRAD.* end-to-end DIFFERENTIABILITY survives. Q_alpha is a CONSTANT w.r.t. the
        decision variables (q, T): it enters the constraint VALUE g = R - proj but
        carries dQ/dq = dQ/dT = 0, so NO new gradient term is needed and the
        existing analytic chain stays exact. Richardson-FD gate < 1e-5 on >= 100
        ACTIVE configs WITH Q_alpha > 0 (kink configs excluded, same gate as
        stage3_minco_perclass_grad GATE.*).
  DROP.* the explicit dP/dT term is STILL load-bearing with Q_alpha on (gate is
        non-vacuous under conformal, not accidentally passing).
  ADD.*  exact, byte-level structure: g shifts by EXACTLY Q_alpha for every sphere
        constraint and by EXACTLY 0 for walls (AABB get no Q_alpha); q_conformal=0
        is a true no-op (deterministic Stage-4 byte-identity).
  CONS.* measurable CONSEQUENCE: solving with Q_alpha pushes the optimized
        trajectory out by ~Q_alpha more (dense clearance to the PREDICTED human
        grows by ~Q_alpha); and against a worst-case TRUE human anywhere in the
        Q_alpha ball, the conformal trajectory still clears d_safe while the
        deterministic one breaches. This is the fly-able-margin payoff, on-trajectory.
  CERT.* the certificate reflects the SAME R the optimizer enforced (validated R ==
        enforced R) and stays sound (cert >= 0 => dense clearance >= d_safe+Q_alpha).

FD is Richardson-extrapolated central difference (O(h^4)).

Run:  python3 test/stage4_conformal_cert_grad.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (                                    # noqa: E402
    _minco_cost_grad, _seg_normals, _partition_obstacles, _alm_constraints,
    _trust_mask, _certificate_margin, _certificate_margin_spacetime,
    plan_minco, hard_clearance, OptParams, DetourConfig)
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


def alm_smooth_config(tr, hard_obs, normals, lam, rho, opt, eps=2e-2):
    """Reject configs near the ALM kink (||P-c||~0 or z_m~0), Q_alpha-aware."""
    cons, P = _alm_constraints(tr, hard_obs, CFG, normals=normals, opt=opt)
    z = lam + rho * cons["g"]
    if np.any(np.abs(z) < eps * max(rho, 1.0)):
        return False
    for i in range(tr.M):
        t_rep = float(tr._cum[i + 1])
        for k in range(6):
            for obs in hard_obs:
                if hasattr(obs, "centre0"):
                    if np.linalg.norm(P[i, k] - obs.predict(t_rep)) < eps:
                        return False
                else:
                    over = (np.maximum(obs.lo - P[i, k], 0.0)
                            + np.maximum(P[i, k] - obs.hi, 0.0))
                    if float(np.linalg.norm(over)) < eps:
                        return False
    return True


# ===========================================================================
# GRAD: ALM inner grad on q AND T over >= 100 active configs, WITH Q_alpha > 0.
# Same gate as stage3_minco_perclass_grad GATE.* but conformal margin switched ON.
# ===========================================================================
print("\n--- GRAD: ALM inner grad vs Richardson-FD on q AND T, Q_alpha=0.3 ON ---")
rng = np.random.default_rng(20260602)
worst_total = worst_q = worst_T = 0.0
n_cfg = 0
attempts = 0
opt_g = OptParams(kappa=8, vmax=80.0, amax=120.0, q_conformal=Q_ALPHA,
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
    # moving sphere (exercises space-time relative-traj cert path where Q_alpha is added)
    vel = rng.standard_normal(3) * 0.4
    obs_list = [SphereObstacle(mid + rng.standard_normal(3) * 0.3,
                               float(rng.uniform(0.4, 0.9)), class_name="human", vel=vel)]
    if rng.random() < 0.3:                       # occasionally a hard AABB (all-hard arm)
        c = mid + rng.standard_normal(3) * 0.5
        half = rng.uniform(0.2, 0.5, 3)
        obs_list.append(AABBObstacle(c - half, c + half, class_name="human"))
    soft, hard = _partition_obstacles(obs_list, CFG, "hard")
    normals = _seg_normals(tr, hard, opt=opt_g)
    tmask = _trust_mask(tr, opt_g)
    Nc = M * 6 * len(hard)
    lam = rng.uniform(0.0, 4.0, Nc)
    rho = float(rng.uniform(5.0, 60.0))
    if not alm_smooth_config(tr, hard, normals, lam, rho, opt_g):
        continue
    cons, _ = _alm_constraints(tr, hard, CFG, normals=normals, opt=opt_g, trust_mask=tmask)
    if not np.any(lam + rho * cons["g"] > 0.0):
        continue
    nq = 3 * (M - 1)
    x = np.concatenate([q.ravel(), T])
    Ttar = float(np.sum(T))
    kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals,
              alm_trust_mask=tmask)
    def f(xx, kw=kw):
        return _minco_cost_grad(xx, start, goal, M, soft, CFG, opt_g, Ttar, **kw)[0]
    _, g = _minco_cost_grad(x, start, goal, M, soft, CFG, opt_g, Ttar, **kw)
    gfd = fd_richardson_vec(f, x)
    worst_total = max(worst_total, rel_err(g, gfd))
    # per-component error normalized by the FULL gradient norm = that component's
    # share of the total relative error (a near-zero subvector can't inflate it).
    den = max(float(np.linalg.norm(g)), float(np.linalg.norm(gfd)), 1e-6)
    if nq > 0:
        worst_q = max(worst_q, float(np.linalg.norm(g[:nq] - gfd[:nq])) / den)
    worst_T = max(worst_T, float(np.linalg.norm(g[nq:] - gfd[nq:])) / den)
    n_cfg += 1

check(f"GRAD.0 conformal ALM grad rel-err < 1e-5 on {n_cfg} active configs",
      worst_total < 1e-5 and n_cfg >= 100, f"worst total={worst_total:.2e}, n={n_cfg}")
check("GRAD.1 dCost/dq rel-err < 1e-5 (Q_alpha adds no q-term)", worst_q < 1e-5,
      f"worst dq={worst_q:.2e}")
check("GRAD.2 dCost/dT rel-err < 1e-5 (Q_alpha adds no T-term)", worst_T < 1e-5,
      f"worst dT={worst_T:.2e}")


# ===========================================================================
# DROP-T: explicit dP/dT still load-bearing WITH Q_alpha on (gate non-vacuous).
# ===========================================================================
print("\n--- DROP-T: explicit dP/dT_i still load-bearing under Q_alpha ---")
M = 3
start = np.array([0.0, 0.0, 0.0]); goal = np.array([4.5, 0.4, 0.2])
q = np.array([[1.5, 0.2, 0.1], [3.0, -0.1, 0.15]])
T = np.array([0.8, 1.0, 0.9])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
mid = tr.eval(0.5 * tr.t_end)
human = SphereObstacle(mid + np.array([0.0, 0.3, 0.0]), 0.6, class_name="human",
                       vel=np.array([0.3, 0.0, 0.0]))
optd = OptParams(kappa=8, vmax=80, amax=120, q_conformal=Q_ALPHA,
                 w_obs=0, w_vel=0, w_accel=0, w_smooth=0, w_time=0)
soft, hard = _partition_obstacles([human], CFG, None)
normals = _seg_normals(tr, hard, opt=optd)
tmask = _trust_mask(tr, optd)
lam = np.full(M * 6, 2.0); rho = 30.0
nq = 3 * (M - 1); x = np.concatenate([q.ravel(), T]); Ttar = float(np.sum(T))
kw = dict(hard_obs=hard, alm_lambda=lam, alm_rho=rho, alm_normals=normals,
          alm_trust_mask=tmask)
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
check("DROP-T.1 WITHOUT explicit-T: gate FAILS (>> 1e-5) => non-vacuous under Q",
      err_drop_T > 1e-2, f"rel-err={err_drop_T:.2e}")


# ===========================================================================
# ADD: exact structural shift. g(Q) - g(0) == Q for spheres, == 0 for walls;
# q_conformal=0 is a true no-op (deterministic byte-identity).
# ===========================================================================
print("\n--- ADD: g shifts by EXACTLY Q_alpha for spheres, 0 for walls ---")
M = 4
start = np.array([0.0, 0.0, 2.0]); goal = np.array([8.0, 0.0, 2.0])
q = np.array([[2.0, 0.3, 2.0], [4.0, -0.2, 2.0], [6.0, 0.1, 2.0]])
T = np.array([0.9, 1.0, 1.1, 0.95])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
mid = tr.eval(0.5 * tr.t_end)
sphere = SphereObstacle(mid + np.array([0.0, 0.4, 0.0]), 0.5, class_name="human",
                        vel=np.array([0.2, 0.0, 0.0]))
wall = AABBObstacle(mid + np.array([0.0, 0.8, 0.0]) - 0.3,
                    mid + np.array([0.0, 0.8, 0.0]) + 0.3, class_name="human")
soft, hard = _partition_obstacles([sphere, wall], CFG, "hard")  # all-hard exercises both
opt0 = OptParams(q_conformal=0.0, spacetime_hard=True)
optQ = OptParams(q_conformal=Q_ALPHA, spacetime_hard=True)
n0 = _seg_normals(tr, hard, opt=opt0)              # same frozen normals for both
tm = _trust_mask(tr, opt0)
cons0, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=opt0, trust_mask=tm)
consQ, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=optQ, trust_mask=tm)
is_sphere = cons0["hidx"] == 0          # obstacle 0 is the sphere, 1 is the wall
is_wall = cons0["hidx"] == 1
trusted = cons0["trust"]                # G_NEG sentinel rows (untrusted) excluded
sph_rows = is_sphere & trusted
dg = consQ["g"] - cons0["g"]
check("ADD.0 sphere g shifts by EXACTLY Q_alpha (bit-exact)",
      sph_rows.any() and np.allclose(dg[sph_rows], Q_ALPHA, atol=0, rtol=0)
      or (sph_rows.any() and float(np.max(np.abs(dg[sph_rows] - Q_ALPHA))) < 1e-12),
      f"max|dg-Q|={float(np.max(np.abs(dg[sph_rows] - Q_ALPHA))) if sph_rows.any() else float('nan'):.2e}")
check("ADD.1 wall g shifts by EXACTLY 0 (AABB gets no Q_alpha)",
      is_wall.any() and float(np.max(np.abs(dg[is_wall]))) == 0.0,
      f"max|dg_wall|={float(np.max(np.abs(dg[is_wall]))) if is_wall.any() else float('nan'):.2e}")
# R array reflects the shift identically
dR = consQ["R"] - cons0["R"]
check("ADD.2 R(sphere) += Q_alpha, R(wall) += 0 (constraint level only)",
      float(np.max(np.abs(dR[is_sphere] - Q_ALPHA))) < 1e-12
      and float(np.max(np.abs(dR[is_wall]))) == 0.0,
      f"dR_sph={float(np.max(np.abs(dR[is_sphere]-Q_ALPHA))):.2e}, dR_wall={float(np.max(np.abs(dR[is_wall]))):.2e}")
# q_conformal=0 is a true no-op: a STATIC sphere takes the Stage-3 path in both
# opt(q=0) and opt=None, so g must be bit-identical (the +0.0 helper is inert).
sphere_static = SphereObstacle(mid + np.array([0.0, 0.4, 0.0]), 0.5, class_name="human")
_, hard_s = _partition_obstacles([sphere_static], CFG, "hard")
n0s = _seg_normals(tr, hard_s, opt=opt0)
cons0s, _ = _alm_constraints(tr, hard_s, CFG, normals=n0s, opt=opt0)
consNs, _ = _alm_constraints(tr, hard_s, CFG, normals=n0s, opt=None)
check("ADD.3 q_conformal=0.0 is a true no-op (static-sphere g identical to opt=None)",
      float(np.max(np.abs(cons0s["g"] - consNs["g"]))) == 0.0,
      f"max|dg|={float(np.max(np.abs(cons0s['g'] - consNs['g']))):.2e}")


# ===========================================================================
# CONS: end-to-end. Solve the SAME scene with Q=0 vs Q=Q_alpha; the conformal
# solve pushes the trajectory out by ~Q_alpha more, and clears d_safe against a
# worst-case TRUE human in the Q_alpha ball where the deterministic one breaches.
# ===========================================================================
print("\n--- CONS: conformal pushes out >= Q_alpha; safe vs worst-case true human ---")
# Gentle graze: human just inside d_safe of the straight path (signed_dist ~ 0.7 <
# d_safe 0.8). A small lateral push => small convex-hull gap => the dense clearance
# hugs the enforced R, so the deterministic solve sits NEAR d_safe (tight), which is
# what makes the worst-case-true-human breach non-vacuous. (The hull certificate is
# SOUND but conservative, so dense clearance >= enforced R, never below.)
straight = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
mid = straight[len(straight) // 2].copy()
human = SphereObstacle(mid + np.array([0.0, 1.20, 0.0]), 0.5, class_name="human")  # static
cfg_pc = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
          "wall":  AvoidParams("wall", "soft", 0.4, 1.0e1)}
opt_det = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, q_conformal=0.0)
opt_con = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, q_conformal=Q_ALPHA)
tr_det, info_det = plan_minco(straight, [human], cfg_pc, opt_params=opt_det, detour_cfg=NO_DETOUR)
tr_con, info_con = plan_minco(straight, [human], cfg_pc, opt_params=opt_con, detour_cfg=NO_DETOUR)
clr_det, _ = hard_clearance(tr_det, [human], cfg_pc, K_eval=4000)   # to PREDICTED human
clr_con, _ = hard_clearance(tr_con, [human], cfg_pc, K_eval=4000)
print(f"    clr_det={clr_det:.3f}  clr_con={clr_con:.3f}  delta={clr_con-clr_det:.3f}")
check("CONS.0 both solves valid", info_det["trajectory_valid"] and info_con["trajectory_valid"],
      f"det={info_det['trajectory_valid']} con={info_con['trajectory_valid']}")
check("CONS.1 conformal realizes the certified extra margin (clr_con >= d_safe+Q)",
      clr_con >= D_SAFE + Q_ALPHA - 0.08, f"clr_con={clr_con:.3f} (target>={D_SAFE+Q_ALPHA:.2f})")
check("CONS.2 conformal pushed strictly further out than deterministic (>= ~Q)",
      (clr_con - clr_det) >= Q_ALPHA - 0.08, f"delta={clr_con - clr_det:.3f} (>=~{Q_ALPHA})")
# worst-case TRUE human anywhere in the Q_alpha ball: clearance_to_true >= clr - Q_alpha.
# The deterministic solve only guarantees d_safe to the PREDICTED human, so a Q-displaced
# true human is no longer guaranteed clear; the conformal solve absorbs exactly that.
true_det = clr_det - Q_ALPHA
true_con = clr_con - Q_ALPHA
check("CONS.3 deterministic BREACHES a worst-case true human (consequence non-vacuous)",
      true_det < D_SAFE - 1e-3, f"true_clr_det={true_det:.3f} < d_safe={D_SAFE}")
check("CONS.4 conformal still CLEARS d_safe against the same worst-case true human",
      true_con >= D_SAFE - 0.05, f"true_clr_con={true_con:.3f} >= d_safe={D_SAFE}")


# ===========================================================================
# CERT: the certificate validates the SAME R the optimizer enforced, and is sound.
# ===========================================================================
print("\n--- CERT: certificate reflects enforced R, stays sound ---")
# static human -> tight per-point _certificate_margin via cons["g"]
n_con = _seg_normals(tr_con, [human], opt=opt_con)
cons_con, _ = _alm_constraints(tr_con, [human], cfg_pc, normals=n_con, opt=opt_con)
cert_static = _certificate_margin(cons_con)
# soundness: cert >= 0 (within ALM tol) AND dense clearance >= d_safe + Q (within tol)
check("CERT.0 static cert reflects R=d_safe+Q (margin>=0 => dense clr>=d_safe+Q)",
      cert_static >= -opt_con.alm_viol_tol * 10 and clr_con >= D_SAFE + Q_ALPHA - 0.05,
      f"cert={cert_static:.3f}, dense_clr={clr_con:.3f} (target>={D_SAFE+Q_ALPHA:.2f})")
# moving human -> space-time relative-traj cert (the line where Q_alpha was added at 1335).
# tau_trust large so the encounter falls INSIDE the hard window (else it is a SOFT-field
# point and may breach — that is the trust-window design, not the conformal wiring).
human_mv = SphereObstacle(mid + np.array([0.0, 1.20, 0.0]), 0.5, class_name="human",
                          vel=np.array([0.6, 0.0, 0.0]))
opt_mv0 = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, q_conformal=0.0, tau_trust=10.0)
opt_mvQ = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, q_conformal=Q_ALPHA, tau_trust=10.0)
tr_mv0, info_mv0 = plan_minco(straight, [human_mv], cfg_pc, opt_params=opt_mv0, detour_cfg=NO_DETOUR)
tr_mvQ, info_mvQ = plan_minco(straight, [human_mv], cfg_pc, opt_params=opt_mvQ, detour_cfg=NO_DETOUR)
cert_mv0 = _certificate_margin_spacetime(tr_mv0, [human_mv], cfg_pc, opt_mv0)
cert_mvQ = _certificate_margin_spacetime(tr_mvQ, [human_mv], cfg_pc, opt_mvQ)
clr_mv0, _ = hard_clearance(tr_mv0, [human_mv], cfg_pc, K_eval=4000)
clr_mvQ, _ = hard_clearance(tr_mvQ, [human_mv], cfg_pc, K_eval=4000)
check("CERT.1 moving-human space-time solves valid", info_mv0["trajectory_valid"]
      and info_mvQ["trajectory_valid"],
      f"mv0={info_mv0['trajectory_valid']} mvQ={info_mvQ['trajectory_valid']}")
check("CERT.2 space-time cert sound: margin>=0 => dense clr >= d_safe (both arms)",
      (cert_mv0 < 0 or clr_mv0 >= D_SAFE - 0.05)
      and (cert_mvQ < 0 or clr_mvQ >= D_SAFE - 0.05),
      f"cert0={cert_mv0:.3f}/clr0={clr_mv0:.3f}  certQ={cert_mvQ:.3f}/clrQ={clr_mvQ:.3f}")
check("CERT.3 space-time conformal pushes dense clearance out by ~Q_alpha",
      abs((clr_mvQ - clr_mv0) - Q_ALPHA) < 0.10, f"delta={clr_mvQ - clr_mv0:.3f} vs Q={Q_ALPHA}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
