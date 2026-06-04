"""Stage 4 — LARGE-BATCH soundness audit of the moving-human continuous-time
certificate (the question the static leak taught us to always ask).

The static convex-hull cert was once UNSOUND: control points all cleared the ball
(cert>0) yet the curve dipped 0.143 m into the human BETWEEN control points. It was
caught by densely sampling the WHOLE curve and finding the leak. This file asks the
SAME question of the MOVING-human relative-trajectory certificate, at scale:

  Over thousands of RANDOM dynamic scenes (moving humans up to 5 m/s + accel, both
  optimizer-produced AND optimizer-DECOUPLED hand-built MINCO trajectories), densely
  sample the curve at its OWN continuous time s and evaluate the TRUE clearance
  signed_dist(p(s), c_human(s)). Measure the VIOLATION RATE of the two implications:

    SOUND.lb   cert is a true LOWER BOUND on the continuous-time margin:
               cert(Q=0) <= dense_min_clearance - d_safe   (never over-claims).
    SOUND.safe cert(Q) >= 0  =>  dense_min_clearance >= d_safe + Q  (no continuous-
               time leak below the enforced floor) for Q in {0.0, 0.3}.

  A single violation in either => the certificate is unsound (a leak), exactly the
  failure mode the static version had. Target: 0 violations, and the batch must be
  NON-VACUOUS (many near-boundary cert~0 cases where a leak would actually show).

Decoupled hand-built trajectories matter: the optimizer only emits well-behaved
curves, so testing only its output could hide a leak. Random MINCO trajectories with
high curvature + fast humans are the adversarial regime that broke the naive cert.

Run:  python3 test/stage4_spacetime_cert_soundness.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (                                    # noqa: E402
    plan_minco, hard_clearance, _certificate_margin_spacetime, _trust_mask,
    _hard_d_safe, OptParams, DetourConfig)
from sando_py.local.obstacles import SphereObstacle                       # noqa: E402
from sando_py.local.minco import MinjerkTraj                              # noqa: E402
from sando_py.local.avoid_config import AvoidParams                       # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4)}
D_SAFE = 0.8
TAU = 1.0e6          # all control points trusted-hard: we test the CERT MATH, not the trust window
K_DENSE = 20000      # dense continuous-time samples per trajectory (fine enough to catch a thin leak)
TOL = 1e-6


def dense_true_min_clearance(tr, human, K=K_DENSE):
    """Independent continuous-time min signed_dist: at each sampled time s evaluate
    the human at its OWN predicted position c(s) (NOT a frozen single time).
    Vectorized: predict(ts) broadcasts over the whole time grid."""
    ts = np.linspace(tr.t_start, tr.t_end, K)
    pts = tr.eval(ts)                                       # (K,3)
    c = human.predict(ts)                                  # (K,3) human at each sample's OWN time
    return float(np.min(np.linalg.norm(pts - c, axis=1) - human.radius))


def rel_margin(tr, human, include_accel, Q=0.0):
    """Mirror of _certificate_margin_spacetime's relative-trajectory margin, with a
    switch to DROP the 0.5*accel*s^2 Bernstein term. include_accel=True reproduces
    production (cross-checked below); include_accel=False is the velocity-only
    ablation that IGNORES the human's acceleration in the relative trajectory."""
    M = tr.M
    P = tr.control_points()                                  # (M,6,3)
    t0 = tr._cum[:M]
    T = tr.T
    c0 = human.predict(t0)                                   # (M,3)
    v_eff = human.vel[None, :] + human.accel[None, :] * t0[:, None]
    kf = np.arange(6, dtype=np.float64)
    lin = v_eff[:, None, :] * ((kf / 5.0)[None, :, None] * T[:, None, None])
    quad = human.accel[None, None, :] * ((T * T)[:, None, None]
                                         * (kf * (kf - 1.0) / 40.0)[None, :, None])
    c_bern = c0[:, None, :] + lin + (quad if include_accel else 0.0)
    D = P - c_bern
    cD = D.mean(axis=1)
    nrm = np.linalg.norm(cD, axis=1); ok = nrm > 1e-12
    a = np.where(ok[:, None], cD / np.where(ok[:, None], nrm[:, None], 1.0),
                 np.array([1.0, 0.0, 0.0]))
    proj = np.einsum("md,mkd->mk", a, D)
    return float(np.min(proj - (human.radius + D_SAFE + Q)))


def audit(tr, human, opt_q0, opt_qP, Q):
    """Return (cert0, certQ, dense_clr) and update the global violation tallies."""
    tm0 = _trust_mask(tr, opt_q0)
    tmQ = _trust_mask(tr, opt_qP)
    cert0 = _certificate_margin_spacetime(tr, [human], CFG, opt_q0, trust_mask=tm0)
    certQ = _certificate_margin_spacetime(tr, [human], CFG, opt_qP, trust_mask=tmQ)
    dense = dense_true_min_clearance(tr, human)
    return cert0, certQ, dense


# ===========================================================================
# Batch generators
# ===========================================================================
rng = np.random.default_rng(20260602)
opt_q0 = OptParams(spacetime_hard=True, tau_trust=TAU, q_conformal=0.0)
opt_qP = OptParams(spacetime_hard=True, tau_trust=TAU, q_conformal=0.3)
Q = 0.3

n_total = 0
n_certpos0 = 0          # cert(Q=0) >= -tol  (claims safe at the deterministic floor)
n_certposQ = 0          # cert(Q=0.3) >= -tol
n_nearbnd = 0           # near-boundary cert(Q=0) in [0, 0.1]  (non-vacuity witness)
v_lb = 0                # SOUND.lb violations: cert0 > true_margin (over-claim)
v_safe0 = 0             # SOUND.safe violations at Q=0
v_safeQ = 0             # SOUND.safe violations at Q=0.3
worst_lb = (-1e9, None)         # largest (cert0 - true_margin) seen
tight0 = (1e9, None)            # smallest cert0-positive slack (dense - d_safe)
tightQ = (1e9, None)            # smallest certQ-positive slack (dense - (d_safe+Q))


def consume(tr, human, tag):
    global n_total, n_certpos0, n_certposQ, n_nearbnd, v_lb, v_safe0, v_safeQ
    global worst_lb, tight0, tightQ
    cert0, certQ, dense = audit(tr, human, opt_q0, opt_qP, Q)
    n_total += 1
    true_margin = dense - D_SAFE
    # SOUND.lb: cert(Q=0) must be a LOWER bound on the true continuous-time margin
    over = cert0 - true_margin
    if over > worst_lb[0]:
        worst_lb = (over, f"{tag}: cert0={cert0:.4f} true_margin={true_margin:.4f}")
    if over > TOL:
        v_lb += 1
    # SOUND.safe at Q=0
    if cert0 >= -TOL:
        n_certpos0 += 1
        slack = dense - D_SAFE
        if slack < tight0[0]:
            tight0 = (slack, f"{tag}: cert0={cert0:.4f} dense={dense:.4f}")
        if dense < D_SAFE - TOL:
            v_safe0 += 1
        if 0.0 <= cert0 <= 0.1:
            n_nearbnd += 1
    # SOUND.safe at Q=0.3
    if certQ >= -TOL:
        n_certposQ += 1
        slack = dense - (D_SAFE + Q)
        if slack < tightQ[0]:
            tightQ = (slack, f"{tag}: certQ={certQ:.4f} dense={dense:.4f} floor={D_SAFE+Q:.2f}")
        if dense < (D_SAFE + Q) - TOL:
            v_safeQ += 1


# --- (1) DECOUPLED hand-built random MINCO trajectories + random moving humans ---
# The adversarial regime: curves the optimizer would never emit (high curvature),
# fast humans whose c(s) sweeps far within a segment.
print("\n--- generating decoupled hand-built dynamic scenes ---")
for _ in range(4500):
    M = int(rng.integers(2, 6))
    start = rng.standard_normal(3) * 2.5
    goal = start + rng.standard_normal(3) * 4.0
    q = (np.linspace(start, goal, M + 1)[1:-1]
         + rng.standard_normal((M - 1, 3)) * rng.uniform(0.3, 2.0))   # high-curvature jitter
    T = rng.uniform(0.4, 1.8, M)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    # human placed near a random point on the curve, offset by ~ R..R+0.7 toward the
    # CONCAVE side, so the curve bulges toward it between control points (the leak regime)
    s = rng.uniform(0.15, 0.85) * tr.t_end
    base = tr.eval(s)
    off = rng.standard_normal(3); off /= max(np.linalg.norm(off), 1e-9)
    r_h = float(rng.uniform(0.3, 0.6))
    dist = (r_h + D_SAFE) + rng.uniform(-0.15, 0.8)        # straddle the boundary
    vel = rng.standard_normal(3) * rng.uniform(0.0, 5.0)   # up to ~5 m/s crossers
    accel = rng.standard_normal(3) * rng.uniform(0.0, 3.0)
    human = SphereObstacle(base + off * dist, r_h, class_name="human", vel=vel, accel=accel)
    consume(tr, human, "decoupled")

# --- (2) OPTIMIZER-produced trajectories (the integrated path actually flown) ---
print("--- generating optimizer-produced dynamic scenes (plan_minco) ---")
NO_DET = DetourConfig(enabled=False)
START = np.array([0.0, 0.0, 1.5]); GOAL = np.array([8.0, 0.0, 1.5])
seed = np.linspace(START, GOAL, 8)
for _ in range(120):
    hy = float(rng.uniform(0.6, 2.0))
    hx = float(rng.uniform(3.0, 5.0))
    vx = float(rng.uniform(-1.5, 1.5)); vy = float(rng.uniform(-3.0, 3.0))
    human = SphereObstacle(np.array([hx, hy, 1.5]), float(rng.uniform(0.3, 0.5)),
                           class_name="human", vel=np.array([vx, vy, 0.0]))
    opt = OptParams(vmax=3.0, amax=3.0, spacetime_hard=True, tau_trust=TAU,
                    q_conformal=float(rng.choice([0.0, 0.3])))
    tr, info = plan_minco(seed, [human], CFG, opt_params=opt, detour_cfg=NO_DET)
    consume(tr, human, "optimized")


# ===========================================================================
# ACCEL: the certificate handles acceleration EXACTLY (not by buried padding).
# The relative trajectory degree-elevates the human's FULL CA motion
# c(s)=c0+v_eff*s+0.5*accel*s^2 to exact Bernstein control points, so 0.5*a*s^2 is
# absorbed verbatim and R needs NO ||v||*T or 0.5||a||*T^2 inflation. We prove this
# the hard way: (0) build cases where 0.5|a|T^2 is LARGE (not buried), (1) the full
# cert stays sound, (2) DROPPING the accel term FALSELY certifies safe while the true
# CA human breaches -> the accel term is load-bearing, exactly handled.
# ===========================================================================
print("\n--- ACCEL: acceleration handled exactly (velocity-only ablation must leak) ---")
xchk = 0.0                       # cross-check: full rel_margin vs production cert
max_accel_disp = 0.0             # largest 0.5|a|T^2 in the adversarial batch (non-buried witness)
n_accel = 0
n_full_leak = 0                  # full cert leaks (must be 0)
n_full_overclaim = 0            # full cert over-claims vs true margin (must be 0)
n_velonly_falsesafe = 0          # velocity-only certifies safe but true CA breaches (load-bearing witness)
for _ in range(2500):
    M = int(rng.integers(2, 5))
    start = rng.standard_normal(3) * 2.0
    goal = start + rng.standard_normal(3) * 3.0
    q = (np.linspace(start, goal, M + 1)[1:-1]
         + rng.standard_normal((M - 1, 3)) * rng.uniform(0.3, 1.2))
    T = rng.uniform(1.0, 1.8, M)                       # large T so 0.5|a|T^2 is significant
    tr = MinjerkTraj.from_endpoints(start, goal, q, T)
    s = rng.uniform(0.2, 0.8) * tr.t_end
    base = tr.eval(s)
    off = rng.standard_normal(3); off /= max(np.linalg.norm(off), 1e-9)
    r_h = float(rng.uniform(0.3, 0.5))
    dist = (r_h + D_SAFE) + rng.uniform(0.1, 1.2)
    a_mag = rng.uniform(2.0, 3.0)
    human = SphereObstacle(base + off * dist, r_h, class_name="human",
                           vel=off * (-rng.uniform(0.0, 0.5)),   # small inward/zero initial vel
                           accel=off * (-a_mag))                  # accelerate TOWARD the path
    n_accel += 1
    Tmax = float(tr.T.max())
    max_accel_disp = max(max_accel_disp, 0.5 * a_mag * Tmax * Tmax)
    full = rel_margin(tr, human, include_accel=True)
    velonly = rel_margin(tr, human, include_accel=False)
    dense = dense_true_min_clearance(tr, human)
    true_margin = dense - D_SAFE
    # cross-check the full helper reproduces production (all trusted via TAU)
    prod = _certificate_margin_spacetime(tr, [human], CFG, opt_q0,
                                         trust_mask=_trust_mask(tr, opt_q0))
    xchk = max(xchk, abs(full - prod))
    if full > true_margin + 1e-6:
        n_full_overclaim += 1
    if full >= -TOL and dense < D_SAFE - TOL:
        n_full_leak += 1
    # the load-bearing witness: velocity-only says safe, but the true CA human breaches
    if velonly >= -TOL and dense < D_SAFE - 1e-3:
        n_velonly_falsesafe += 1

print(f"accel scenes={n_accel}  max 0.5|a|T^2 = {max_accel_disp:.2f} m  (Bernstein-absorbed exactly)")
print(f"full-cert leaks={n_full_leak}  full-cert over-claims={n_full_overclaim}  "
      f"velocity-only false-safe={n_velonly_falsesafe}  xcheck|full-prod|={xchk:.2e}")
check("ACCEL.x full rel-margin == production _certificate_margin_spacetime (mirror correct)",
      xchk < 1e-9, f"max|full-prod|={xchk:.2e}")
check("ACCEL.0 the 0.5|a|T^2 term is LARGE in the batch (not buried by padding)",
      max_accel_disp > 1.5, f"max 0.5|a|T^2 = {max_accel_disp:.2f} m")
check("ACCEL.1 full cert sound under high accel (0 leaks, 0 over-claims)",
      n_full_leak == 0 and n_full_overclaim == 0,
      f"leaks={n_full_leak}, over-claims={n_full_overclaim}")
check("ACCEL.2 velocity-only ablation LEAKS (drops 0.5a*s^2 => accel term load-bearing)",
      n_velonly_falsesafe >= 10,
      f"{n_velonly_falsesafe} false-safe cases where dropping accel certifies an unsafe traj")


# ===========================================================================
# Verdict
# ===========================================================================
print(f"\nscenes audited: {n_total}  | cert-safe(Q=0): {n_certpos0}  cert-safe(Q=0.3): {n_certposQ}"
      f"  | near-boundary(cert0 in [0,0.1]): {n_nearbnd}")
print(f"worst (cert0 - true_margin) = {worst_lb[0]:.2e}   [{worst_lb[1]}]")
print(f"tightest cert-safe slack  Q=0:  {tight0[0]:.4f}   [{tight0[1]}]")
print(f"tightest cert-safe slack  Q=0.3:{tightQ[0]:.4f}   [{tightQ[1]}]")
print(f"VIOLATIONS  lb={v_lb}  safe(Q=0)={v_safe0}  safe(Q=0.3)={v_safeQ}")

check("SOUND.0 batch is NON-VACUOUS (>=300 cert-safe AND >=50 near-boundary cases)",
      n_certpos0 >= 300 and n_nearbnd >= 50,
      f"cert-safe={n_certpos0}, near-boundary={n_nearbnd}")
check("SOUND.lb cert(Q=0) is a true lower bound on the continuous-time margin (0 over-claims)",
      v_lb == 0, f"{v_lb} over-claims; worst (cert0 - true_margin)={worst_lb[0]:.2e}")
check("SOUND.safe(Q=0) cert>=0 => dense continuous-time clearance >= d_safe (0 leaks)",
      v_safe0 == 0, f"{v_safe0}/{n_certpos0} leaks; tightest slack={tight0[0]:.4f}")
check("SOUND.safe(Q=0.3) cert>=0 => dense clearance >= d_safe+Q_alpha (0 leaks)",
      v_safeQ == 0, f"{v_safeQ}/{n_certposQ} leaks; tightest slack={tightQ[0]:.4f}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
