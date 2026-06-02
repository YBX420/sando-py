"""Stage 4 — DETERMINISTIC worst-case safety mode (toward "never collide").

conformal Q_alpha gives PROBABILISTIC coverage (1-alpha): an alpha failure rate,
data-hungry, distribution-dependent -- it cannot reach an "every flight hour" target.
The deterministic mode replaces Q_alpha with a WORST-CASE reachable-set radius from
the human's PHYSICAL limits: within the trust window the human (trusted current CA
estimate + a bounded UNOBSERVABLE acceleration manoeuvre + bounded estimate error)
cannot leave Ball(c_nominal(t), reach), reach = est_pos_err + est_vel_err*tau +
0.5*a_max_human*tau^2. Enforcing relative margin >= radius + d_safe + reach makes a
collision IMPOSSIBLE under those assumptions -- not merely 1-alpha unlikely.

THE CLAIMS:
  DET.reach  the enforced R in deterministic mode is exactly radius + d_safe + reach
             (+ epsilon_track); reach matches the physical formula.
  DET.grad   reach is CONSTANT in (q, T) (evaluated at tau_trust) -> no gradient term,
             Richardson-FD gate < 1e-5 (same free trick as Q_alpha / epsilon_track).
  DET.cf     THE IRON PROOF -- counterfactual worst manoeuvres: against THOUSANDS of
             physically-admissible worst-case human accelerations (|a| <= a_max_human in
             ANY direction) over the trust window, the deterministic plan clears d_safe
             EVERY TIME (0 collisions), while the conformal plan (small Q_alpha) is
             breached by the physical worst case -> probabilistic margin != worst-case safe.
  DET.hover  a human whose reachable set blocks the corridor -> NO valid plan
             (trajectory_valid=False) = the honest STOP/HOVER, never a false 'valid'.
  DET.noop   safety_mode='conformal' (default) is byte-identical to before.

SCOPE: this certifies the PLANNER layer only -- "given the human is detected and the
estimate is in-bound, the plan cannot collide within the trust window". The per-flight-
hour system target also needs perception (miss-detection) + hardware fault budgets that
live OUTSIDE this planner.

Run:  python3 test/stage4_deterministic_safety.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (                                    # noqa: E402
    _minco_cost_grad, _seg_normals, _partition_obstacles, _alm_constraints,
    _trust_mask, _pred_margin, plan_minco, hard_clearance, OptParams, DetourConfig)
from sando_py.local.obstacles import SphereObstacle                       # noqa: E402
from sando_py.local.minco import MinjerkTraj                              # noqa: E402
from sando_py.local.avoid_config import AvoidParams                       # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4)}
ND = DetourConfig(enabled=False)
D_SAFE = 0.8
TAU = 0.7
A_MAX_H = 2.0                      # human physical max |accel| (m/s^2)
REACH = 0.5 * A_MAX_H * TAU * TAU  # = 0.49 m  (est errors 0 in the DET.cf scene)


def rel_err(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-6)


def fd_richardson_vec(f, x, h0=1e-4):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = 1.0
        D = lambda h: (f(x + h * e) - f(x - h * e)) / (2.0 * h)
        d1 = D(h0); d2 = D(h0 / 2.0)
        g[i] = (4.0 * d2 - d1) / 3.0
    return g


# ===========================================================================
# DET.reach + DET.noop : the enforced R and the no-op default
# ===========================================================================
print("\n--- DET.reach: enforced R = radius + d_safe + reach (+ eps); reach = phys formula ---")
opt_d = OptParams(safety_mode="deterministic", a_max_human=A_MAX_H, tau_trust=TAU,
                  est_pos_err=0.05, est_vel_err=0.1)
reach_expect = 0.05 + 0.1 * TAU + 0.5 * A_MAX_H * TAU * TAU
check("DET.reach.0 _pred_margin matches the physical reach formula",
      abs(_pred_margin(opt_d) - reach_expect) < 1e-12,
      f"_pred_margin={_pred_margin(opt_d):.4f} vs {reach_expect:.4f}")
# build a config and read cons["R"]
M = 4
start = np.array([0.0, 0.0, 2.0]); goal = np.array([8.0, 0.0, 2.0])
q = np.array([[2.0, 0.3, 2.0], [4.0, -0.2, 2.0], [6.0, 0.1, 2.0]])
T = np.array([0.9, 1.0, 1.1, 0.95])
tr = MinjerkTraj.from_endpoints(start, goal, q, T)
mid = tr.eval(0.5 * tr.t_end)
sphere = SphereObstacle(mid + np.array([0.0, 0.4, 0.0]), 0.5, class_name="human",
                        vel=np.array([0.2, 0.0, 0.0]))
_, hard = _partition_obstacles([sphere], CFG, "hard")
n0 = _seg_normals(tr, hard, opt=opt_d); tm = _trust_mask(tr, opt_d)
consd, _ = _alm_constraints(tr, hard, CFG, normals=n0, opt=opt_d, trust_mask=tm)
trusted = consd["trust"]
R_expect = 0.5 + D_SAFE + reach_expect          # radius 0.5 + d_safe + reach (+0 eps)
check("DET.reach.1 enforced R (trusted sphere) == radius + d_safe + reach",
      np.allclose(consd["R"][trusted], R_expect, atol=1e-12),
      f"R={consd['R'][trusted][0]:.4f} vs {R_expect:.4f}")
# no-op: default conformal mode == old behaviour. Use a STATIC sphere so opt=None (spacetime
# off) and opt(conformal,q=0) take the SAME Stage-3 path (a moving sphere would differ only by
# the spacetime trust-window sentinel, unrelated to safety_mode).
sphere_static = SphereObstacle(mid + np.array([0.0, 0.4, 0.0]), 0.5, class_name="human")
_, hard_s = _partition_obstacles([sphere_static], CFG, "hard")
n0s = _seg_normals(tr, hard_s, opt=OptParams())
consc, _ = _alm_constraints(tr, hard_s, CFG, normals=n0s, opt=OptParams())   # conformal, q=0
consNone, _ = _alm_constraints(tr, hard_s, CFG, normals=n0s, opt=None)
check("DET.noop default safety_mode='conformal' with q=0 is byte-identical to opt=None",
      float(np.max(np.abs(consc["g"] - consNone["g"]))) == 0.0,
      f"max|dg|={float(np.max(np.abs(consc['g'] - consNone['g']))):.2e}")


# ===========================================================================
# DET.grad : reach is constant -> gradient gate unaffected
# ===========================================================================
print("\n--- DET.grad: deterministic reach is constant in (q,T) -> gradient gate holds ---")
rng = np.random.default_rng(20260604)
opt_g = OptParams(kappa=8, vmax=80.0, amax=120.0, safety_mode="deterministic",
                  a_max_human=A_MAX_H, tau_trust=5.0,        # large so points stay trusted
                  w_obs=0.0, w_vel=0.0, w_accel=0.0, w_smooth=0.0, w_time=0.0)
worst = 0.0; n_cfg = 0; attempts = 0
while n_cfg < 100 and attempts < 6000:
    attempts += 1
    Mm = int(rng.integers(2, 6))
    s = rng.standard_normal(3) * 2.0; g_ = rng.standard_normal(3) * 2.0
    qq = rng.standard_normal((Mm - 1, 3)) * 2.0; TT = rng.uniform(0.5, 1.6, Mm)
    trr = MinjerkTraj.from_endpoints(s, g_, qq, TT)
    md = trr.eval(0.5 * trr.t_end)
    obs = [SphereObstacle(md + rng.standard_normal(3) * 0.3, float(rng.uniform(0.4, 0.9)),
                          class_name="human", vel=rng.standard_normal(3) * 0.4)]
    soft, hd = _partition_obstacles(obs, CFG, "hard")
    nz = _seg_normals(trr, hd, opt=opt_g); tmk = _trust_mask(trr, opt_g)
    lam = rng.uniform(0.0, 4.0, Mm * 6 * len(hd)); rho = float(rng.uniform(5.0, 60.0))
    cons, P = _alm_constraints(trr, hd, CFG, normals=nz, opt=opt_g, trust_mask=tmk)
    z = lam + rho * cons["g"]
    if np.any(np.abs(z) < 2e-2 * max(rho, 1.0)) or not np.any(z > 0):
        continue
    x = np.concatenate([qq.ravel(), TT]); Ttar = float(np.sum(TT))
    kw = dict(hard_obs=hd, alm_lambda=lam, alm_rho=rho, alm_normals=nz, alm_trust_mask=tmk)
    f = lambda xx, kw=kw: _minco_cost_grad(xx, s, g_, Mm, soft, CFG, opt_g, Ttar, **kw)[0]
    _, gan = _minco_cost_grad(x, s, g_, Mm, soft, CFG, opt_g, Ttar, **kw)
    gfd = fd_richardson_vec(f, x)
    worst = max(worst, rel_err(gan, gfd)); n_cfg += 1
check(f"DET.grad.0 deterministic-mode ALM grad rel-err < 1e-5 ({n_cfg} configs)",
      worst < 1e-5 and n_cfg >= 90, f"worst={worst:.2e}, n={n_cfg}")


# ===========================================================================
# DET.cf : THE IRON PROOF -- counterfactual worst manoeuvres
# Tests the BUDGET MATH directly (decoupled from optimizer tightness): build a trajectory
# that clears the NOMINAL static human by exactly d_safe+margin everywhere, then hit the human
# with every physically-admissible worst manoeuvre (|a|<=a_max_human in any direction over the
# window). margin=reach -> 0 collisions (the human cannot leave Ball(c0, reach) in-window, and
# the trajectory cleared c0 by d_safe+reach); margin=Q<reach -> the physical worst case BREACHES.
# ===========================================================================
print("\n--- DET.cf: counterfactual physical worst manoeuvres (|a|<=a_max_human) ---")
C0 = np.array([1.2, 0.0, 1.5]); V0 = np.array([0.0, 0.0, 0.0]); RH = 0.4   # nominal static human


def straight_traj(y_off, speed=2.2):
    """A (near-)straight min-jerk trajectory at constant y=y_off that passes over the human's x
    within the window; its closest approach to the static nominal human is exactly y_off - RH."""
    start = np.array([0.0, y_off, 1.5]); goal = np.array([3.0, y_off, 1.5])
    return MinjerkTraj.from_endpoints(start, goal, np.zeros((0, 3)), np.array([3.0 / speed]),
                                      v0=np.array([speed, 0.0, 0.0]))


def worst_breaches(tr, c0, v0, n=5000):
    """#breaches and worst min-clearance over N physical worst manoeuvres (|a|<=a_max_human in
    ANY direction) of the human, evaluated densely over the trust window. PER-TIME: at each
    sampled t the human is at c0 + v0*t + 0.5*a*t^2, ||deviation|| <= 0.5*a_max*t^2 <= reach."""
    ts = np.linspace(0.0, min(TAU, tr.t_end), 80)
    pts = tr.eval(ts)                                            # planned path over the window
    dirs = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    dirs += list(rng.standard_normal((n, 3)))
    nb = 0; worst_clr = np.inf
    for d in dirs:
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            continue
        a = (np.asarray(d, float) / nrm) * A_MAX_H               # full-magnitude worst accel
        c_true = c0[None, :] + v0[None, :] * ts[:, None] + 0.5 * a[None, :] * (ts[:, None] ** 2)
        clr = float(np.min(np.linalg.norm(pts - c_true, axis=1) - RH))
        worst_clr = min(worst_clr, clr)
        if clr < D_SAFE - 1e-3:
            nb += 1
    return nb, worst_clr


tr_reach = straight_traj(RH + D_SAFE + REACH)      # clears nominal by exactly d_safe + reach
tr_q = straight_traj(RH + D_SAFE + 0.1)            # clears nominal by only d_safe + 0.1 (< reach)
clr_reach, _ = hard_clearance(tr_reach, [human_nom := SphereObstacle(C0, RH, vel=V0, class_name="human")],
                              CFG, K_eval=4000)
nb_reach, wc_reach = worst_breaches(tr_reach, C0, V0)
nb_q, wc_q = worst_breaches(tr_q, C0, V0)
print(f"   reach-budget traj (clears nominal {clr_reach:.2f}): {nb_reach} breaches, worst clr={wc_reach:.3f}")
print(f"   Q-budget traj (clears nominal {RH+D_SAFE+0.1-RH:.2f}=d_safe+0.1): {nb_q} breaches, worst clr={wc_q:.3f}")
check("DET.cf.0 reach-budget trajectory clears the nominal by ~d_safe+reach (setup valid)",
      clr_reach >= D_SAFE + REACH - 0.02, f"nominal clr={clr_reach:.3f} ~= {D_SAFE+REACH:.2f}")
check("DET.cf.1 IRON PROOF: 0 collisions over ALL physical worst manoeuvres (reach budget)",
      nb_reach == 0 and wc_reach >= D_SAFE - 1e-3,
      f"{nb_reach} breaches, worst clr={wc_reach:.3f} >= d_safe={D_SAFE}")
check("DET.cf.2 a sub-reach (probabilistic Q) budget IS breached by the physical worst case",
      nb_q > 0 and wc_q < D_SAFE,
      f"{nb_q} breaches, worst clr={wc_q:.3f} < d_safe -> Q (1-alpha) is NOT worst-case safe")


# end-to-end: deterministic plan_minco on a feasible scene -> the FIXED validity gate confirms
# the reach margin was achieved (valid <=> nominal clearance >= d_safe + reach).
print("\n--- DET.e2e: deterministic plan_minco achieves the reach margin (gate now honest) ---")
ASTAR = np.linspace([0.0, 0.0, 1.5], [8.0, 0.0, 1.5], 8)
human_e2e = SphereObstacle([1.8, 1.5, 1.5], RH, class_name="human")
opt_det = OptParams(vmax=3.0, amax=3.0, maxiter=400, safety_mode="deterministic",
                    a_max_human=A_MAX_H, tau_trust=TAU)
tr_e2e, info_e2e = plan_minco(ASTAR, [human_e2e], CFG, opt_params=opt_det, detour_cfg=ND)
clr_e2e, _ = hard_clearance(tr_e2e, [human_e2e], CFG, K_eval=4000)
check("DET.e2e.0 deterministic plan valid IFF it achieved the enforced d_safe+reach margin",
      info_e2e["trajectory_valid"] and clr_e2e >= D_SAFE + REACH - 0.05,
      f"valid={info_e2e['trajectory_valid']}, nominal clr={clr_e2e:.3f} >= {D_SAFE+REACH:.2f}")


# ===========================================================================
# DET.speed : the SPEED-bound reach model (don't trust direction) -- the design correction
# the real-data safety budget forced. The human is within v_max*tau of NOW under ANY motion
# (incl. a reversal the accel/CA model misses), so the cert clears the STATIC c0 by v_max*tau.
# ===========================================================================
print("\n--- DET.speed: speed-bound reach (physical, covers any speed-bounded motion incl reversal) ---")
V_MAX_H = 2.5
TAU_S = 0.7
SPEED_REACH = V_MAX_H * TAU_S                      # = 1.75 m
opt_speed = OptParams(safety_mode="deterministic", reach_model="speed",
                      v_max_human=V_MAX_H, tau_trust=TAU_S, est_pos_err=0.05)
check("DET.speed.0 _pred_margin (speed model) = est_pos_err + v_max_human*tau",
      abs(_pred_margin(opt_speed) - (0.05 + SPEED_REACH)) < 1e-12,
      f"_pred_margin={_pred_margin(opt_speed):.4f} vs {0.05 + SPEED_REACH:.4f}")


def worst_breaches_speed(tr, c0, v_max, n=5000):
    """#breaches under ANY speed-bounded human motion: c_true(t) = c0 + v_max*dir*t with |dir|=1
    (constant full-speed in any direction -- the worst speed-bounded adversary, incl. a reversal
    of whatever it was doing). Per-time: ||c_true(t)-c0|| = v_max*t <= v_max*tau."""
    ts = np.linspace(0.0, min(TAU_S, tr.t_end), 80)
    pts = tr.eval(ts)
    dirs = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
    dirs += list(rng.standard_normal((n, 3)))
    nb = 0; worst_clr = np.inf
    for d in dirs:
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            continue
        v = (np.asarray(d, float) / nrm) * v_max
        c_true = c0[None, :] + v[None, :] * ts[:, None]
        clr = float(np.min(np.linalg.norm(pts - c_true, axis=1) - RH))
        worst_clr = min(worst_clr, clr)
        if clr < D_SAFE - 1e-3:
            nb += 1
    return nb, worst_clr


# speed-bound trajectory clears the STATIC c0 by d_safe + v_max*tau; accel-budget traj clears
# only by d_safe + 0.5*a_max*tau^2 (< v_max*tau here) -> the speed adversary breaches it.
tr_speed = straight_traj(RH + D_SAFE + SPEED_REACH)
tr_accelbudget = straight_traj(RH + D_SAFE + 0.5 * A_MAX_H * TAU_S * TAU_S)   # 0.49 m budget
nb_sp, wc_sp = worst_breaches_speed(tr_speed, C0, V_MAX_H)
nb_ab, wc_ab = worst_breaches_speed(tr_accelbudget, C0, V_MAX_H)
print(f"   speed-budget traj (reach {SPEED_REACH:.2f}): {nb_sp} breaches, worst clr={wc_sp:.3f}")
print(f"   accel-budget traj (reach {0.5*A_MAX_H*TAU_S*TAU_S:.2f}): {nb_ab} breaches, worst clr={wc_ab:.3f}")
check("DET.speed.1 IRON PROOF: speed-bound reach -> 0 collisions under ANY speed-bounded motion",
      nb_sp == 0 and wc_sp >= D_SAFE - 1e-3,
      f"{nb_sp} breaches, worst clr={wc_sp:.3f} >= d_safe (covers reversal the accel model misses)")
check("DET.speed.2 the accel/CA budget IS breached by a full-speed (e.g. reversing) human",
      nb_ab > 0 and wc_ab < D_SAFE,
      f"{nb_ab} breaches, worst clr={wc_ab:.3f} < d_safe -> trusting v0 under-covers a fast/reversing human")

# end-to-end: plan_minco speed mode treats the MOVING human as static at c0 and achieves v_max*tau.
# Uses a SHORT tau (the safety-budget lever: short replan window keeps the physical reach feasible)
# and a human far enough off-path that the start clears the no-go ball.
print("\n--- DET.speed.e2e: plan_minco speed mode treats moving human static at c0, clears v_max*tau ---")
TAU_E2E = 0.4
REACH_E2E = V_MAX_H * TAU_E2E                      # = 1.0 m (short tau -> feasible reach)
human_mv = SphereObstacle([1.8, 2.6, 1.5], RH, vel=np.array([1.5, 0.0, 0.0]), class_name="human")
opt_sp_e2e = OptParams(vmax=3.0, amax=3.0, maxiter=400, safety_mode="deterministic",
                       reach_model="speed", v_max_human=V_MAX_H, tau_trust=TAU_E2E)
tr_spe, info_spe = plan_minco(ASTAR, [human_mv], CFG, opt_params=opt_sp_e2e, detour_cfg=ND)
# clearance to the STATIC c0 (what speed mode certifies), not the moving prediction
static_c0 = SphereObstacle(human_mv.centre0, RH, class_name="human")
clr_static, _ = hard_clearance(tr_spe, [static_c0], CFG, K_eval=4000)
check("DET.speed.e2e.0 speed-mode plan valid AND clears the STATIC c0 by d_safe + v_max*tau (short tau)",
      info_spe["trajectory_valid"] and clr_static >= D_SAFE + REACH_E2E - 0.05,
      f"valid={info_spe['trajectory_valid']}, static-c0 clr={clr_static:.3f} >= {D_SAFE+REACH_E2E:.2f} (tau={TAU_E2E})")


# ===========================================================================
# DET.hover : reachable set blocks the corridor -> honest STOP, never false 'valid'
# ===========================================================================
print("\n--- DET.hover: a huge reachable set blocks the only gap -> STOP (valid=False) ---")
# human dead-centre on a straight 2-point seed + a huge a_max_human -> reach engulfs the path
block_astar = np.linspace([0.0, 0.0, 1.5], [6.0, 0.0, 1.5], 2)
blocker = SphereObstacle([3.0, 0.0, 1.5], 0.5, class_name="human")
opt_block = OptParams(vmax=3.0, amax=3.0, safety_mode="deterministic",
                      a_max_human=8.0, tau_trust=1.2)            # reach ~ 5.8 m -> engulfs
_, info_block = plan_minco(block_astar, [blocker], CFG, opt_params=opt_block, detour_cfg=ND)
check("DET.hover.0 blocked corridor -> trajectory_valid=False (honest STOP, not false safe)",
      not info_block["trajectory_valid"],
      f"valid={info_block['trajectory_valid']} reason={info_block.get('failure_reason')}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
