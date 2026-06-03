"""Stage 3/4 — moving-human continuous-time certificate on FAST humans.

Regression for the relative-trajectory certificate fix. The OLD per-segment
isotropic ball inflation  R = r + d_safe + ||v||*T_i + 0.5*||a||*T_i^2  (all 6
control points share one R) gave catastrophic FALSE NEGATIVES for fast crossers:
the human moves along ONE direction, but the ball inflates in ALL directions, so
cert dropped to ~-5 on trajectories that were actually +4 m clear. The fix is a
RELATIVE-TRAJECTORY supporting halfspace: degree-elevate the human's local CA
motion c(s)=c0+v_eff*s+0.5*a*s^2 to its quintic Bernstein control points c_k,
subtract from the trajectory control points (D_ik = P_ik - c_k), and put one
supporting halfspace on D. Sound by convex hull (P(s)-c(s)=sum_k B_k(s) D_ik),
and tight (no isotropic blow-up).

Checks:
  CERT.fast.0  the OLD ball inflation IS false-negative on fast humans
               (cert<0 while the trajectory is truly safe) — proves the bug.
  CERT.fast.1  the relative cert is NOT false-negative (>=0) on those same cases.
  CERT.fast.2  the relative cert stays SOUND: it never exceeds the true margin
               (cert <= true_signed_dist - d_safe), and cert>=0 => dense no-leak.

Run:  python3 test/stage3_minco_cert_fast.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (plan_minco, OptParams, DetourConfig, hard_clearance,  # noqa: E402
                                      _certificate_margin_spacetime, _trust_mask, _hard_d_safe)
from sando_py.local.obstacles import SphereObstacle, AABBObstacle  # noqa: E402
from sando_py.local.avoid_config import AvoidParams  # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

START = np.array([0.0, 0.0, 1.5]); GOAL = np.array([8.0, 0.0, 1.5])
WALL = AABBObstacle([3.0, 1.2, 0.0], [5.0, 3.0, 3.0], class_name="wall")
CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall", "soft", 0.4, 1.0e1)}
HR = 0.4
DET = DetourConfig(enabled=True)


def old_ball_margin(tr, obs, d_safe, trust_mask):
    """The OLD per-segment isotropic ball inflation, kept here only to prove it
    was false-negative (CERT.fast.0). NOT used by the planner anymore."""
    M = tr.M
    P = tr.control_points()
    centroid = P.mean(axis=1)
    t0 = tr._cum[:M]
    c0 = obs.predict(t0)
    a = centroid - c0
    nrm = np.linalg.norm(a, axis=1); ok = nrm > 1e-12
    a = np.where(ok[:, None], a / np.where(ok[:, None], nrm[:, None], 1.0),
                 np.array([1.0, 0.0, 0.0]))
    proj = np.einsum("md,mkd->mk", a, P - c0[:, None, :])
    vnorm = float(np.linalg.norm(obs.vel)); anorm = float(np.linalg.norm(obs.accel))
    R = float(obs.radius) + d_safe + vnorm * tr.T + 0.5 * anorm * tr.T * tr.T
    m = proj - R[:, None]
    return float(np.min(m[trust_mask])) if np.any(trust_mask) else float("inf")


# fast crossing humans (||v|| 3-5 m/s) timed near the drone's path
SCEN = [
    ("fast-v3", [4.0, -2.0, 1.5], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]),
    ("fast-v4", [4.0, -2.5, 1.5], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]),
    ("fast-v5", [4.5, -3.0, 1.5], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]),
    ("ca-fast", [4.0, -2.0, 1.5], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]),
]

print("\n--- moving-human certificate on fast crossers: relative fix vs old ball inflation ---")
n_old_fn = 0
for name, h0, hv, ha in SCEN:
    human = SphereObstacle(np.array(h0, float), HR, vel=np.array(hv, float),
                           class_name="human", accel=np.array(ha, float))
    opt = OptParams(vmax=3.0, amax=3.0, spacetime_hard=True, tau_trust=10.0)
    tr, info = plan_minco(np.linspace(START, GOAL, 8), [human, WALL], CFG,
                          opt_params=opt, detour_cfg=DET)
    tm = _trust_mask(tr, opt)
    ds = _hard_d_safe(human, CFG)
    cert = _certificate_margin_spacetime(tr, [human], CFG, opt, trust_mask=tm)   # relative (current)
    old = old_ball_margin(tr, human, ds, tm)                                     # old ball inflation
    clr, _ = hard_clearance(tr, [human], CFG, K_eval=5000)
    true_margin = clr - ds
    truly_safe = clr >= ds
    if old < 0.0 and truly_safe:
        n_old_fn += 1
    # CERT.fast.1: relative cert not false-negative on a truly-safe fast crosser
    check(f"CERT.fast.1[{name}] relative cert >=0 on truly-safe fast human",
          (cert >= -1e-6) and truly_safe,
          f"cert={cert:.3f} (old ball={old:.3f}) true_clr={clr:.3f}")
    # CERT.fast.2: SOUND — cert is a lower bound on the true margin, and cert>=0 => no dense leak
    sound = (cert <= true_margin + 1e-6) and ((cert < 0.0) or truly_safe)
    check(f"CERT.fast.2[{name}] relative cert SOUND (<= true margin, >=0 => no leak)",
          sound, f"cert={cert:.3f} <= true_margin={true_margin:.3f}, leak_ok={truly_safe}")

# CERT.fast.0: the OLD inflation really was false-negative on these fast cases (non-empty: proves the bug)
check("CERT.fast.0 OLD ball inflation IS false-negative on fast humans (bug existed)",
      n_old_fn >= 3, f"{n_old_fn}/{len(SCEN)} fast cases were false-negative under old ball inflation")

total = len(results); passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
