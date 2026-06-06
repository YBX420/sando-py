"""Stage 3 — per-class HARD/ALM invariants (geometric + ALM structural).

  I.0 endpoints pinned exactly (ALM never moves start / goal)
  I.1 translation invariance: shift scene -> hard clearance certificate shifts too
  I.2 continuous-time guarantee holds on a BATCH of random hard-human scenes
  I.3 certificate margin >= 0 implies dense clearance >= d_safe (NO gap leak), batch
  I.4 conservatism knob: inflating d_safe never DECREASES achieved clearance
  I.5 moving human: hard constraint at the conservative segment-end time still clears
  I.6 ALM idempotence: re-solving from the converged trajectory keeps it valid
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402
from sando_py.local.obstacles import SphereObstacle                        # noqa: E402
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


NO_DETOUR = DetourConfig(enabled=False)
D_SAFE = 0.8
CFG = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
OPT = OptParams(maxiter=200, kappa=14, vmax=3.0, amax=3.0)
rng = np.random.default_rng(2026)


def dense_min_clr(tr, obs, K=2000):
    ts = np.linspace(tr.t_start, tr.t_end, K)
    pts = tr.eval(ts)
    return float(min(obs.signed_dist(p, float(t)) for p, t in zip(pts, ts)))


def scene(off=0.3, r=0.4):
    p = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
    mid = p[len(p) // 2].copy()
    h = SphereObstacle(mid + np.array([0.0, off, 0.0]), r, class_name="human")
    return p, h


# ---------------------------------------------------------------- I.0 endpoints
print("\n--- I.0 endpoints pinned exactly through ALM ---")
bad, worst = 0, 0.0
for _ in range(8):
    p, h = scene(off=float(rng.uniform(0.2, 0.5)), r=float(rng.uniform(0.3, 0.5)))
    tr, _ = plan_minco(p, [h], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
    es = float(np.max(np.abs(tr.eval(tr.t_start) - p[0])))
    ee = float(np.max(np.abs(tr.eval(tr.t_end) - p[-1])))
    if es > 1e-9 or ee > 1e-9:
        bad += 1
    worst = max(worst, es, ee)
check("I.0 ALM pins start/goal exactly (structural)", bad == 0, f"worst={worst:.2e}")


# ---------------------------------------------------------------- I.1 translation
print("\n--- I.1 translation invariance of the hard clearance ---")
bad, worst = 0, 0.0
for _ in range(5):
    p, h = scene(off=0.3, r=0.4)
    delta = rng.standard_normal(3) * 2.0
    tr1, info1 = plan_minco(p, [h], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
    h2 = SphereObstacle(h.centre0 + delta, h.radius, class_name="human")
    tr2, info2 = plan_minco(p + delta, [h2], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
    # the achieved continuous min clearance is translation-invariant
    e = abs(info1["continuous_min_clearance"] - info2["continuous_min_clearance"])
    if e > 0.05:
        bad += 1
    worst = max(worst, e)
check("I.1 hard clearance is translation invariant", bad == 0, f"worst|dclr|={worst:.3e}")


# ---------------------------------------------------------------- I.2 batch guarantee
print("\n--- I.2 continuous-time guarantee on a BATCH of random hard-human scenes ---")
n, n_valid, n_clear, worst_breach = 0, 0, 0, 0.0
for _ in range(12):
    off = float(rng.uniform(0.15, 0.6))
    r = float(rng.uniform(0.3, 0.55))
    p, h = scene(off=off, r=r)
    tr, info = plan_minco(p, [h], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
    n += 1
    clr = dense_min_clr(tr, h, K=2000)
    if info["trajectory_valid"]:
        n_valid += 1
        if clr >= D_SAFE - 1e-6:
            n_clear += 1
        else:
            worst_breach = max(worst_breach, D_SAFE - clr)
check("I.2 every VALID hard solve clears the human >= d_safe (2000-pt dense)",
      n_valid > 0 and n_clear == n_valid,
      f"valid={n_valid}/{n}, cleared={n_clear}/{n_valid}, worst leaked breach={worst_breach:.4f}")


# ---------------------------------------------------------------- I.3 no gap leak
print("\n--- I.3 certificate_margin >= 0 => dense clearance >= d_safe (no leak), batch ---")
n, bad = 0, 0
for _ in range(12):
    p, h = scene(off=float(rng.uniform(0.2, 0.5)), r=float(rng.uniform(0.3, 0.5)))
    tr, info = plan_minco(p, [h], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
    if info["certificate_margin"] >= -1e-6:   # control polygon clears
        n += 1
        clr = dense_min_clr(tr, h, K=2000)
        if clr < D_SAFE - 1e-3:               # but the curve leaked -> certificate is a LIE
            bad += 1
check("I.3 whenever the certificate says 'clear', the dense curve IS clear (no leak)",
      n > 0 and bad == 0, f"checked={n}, leaks={bad}")


# ---------------------------------------------------------------- I.4 conservatism knob
print("\n--- I.4 inflating d_safe never DECREASES achieved clearance ---")
p, h = scene(off=0.3, r=0.4)
clrs = []
for ds in [0.6, 0.8, 1.0]:
    cfg = {"human": AvoidParams("human", "hard", ds, 1.0e4),
           "wall":  AvoidParams("wall", "soft", 0.4, 1.0e1)}
    tr, _ = plan_minco(p, [h], cfg, opt_params=OPT, detour_cfg=NO_DETOUR)
    clrs.append(dense_min_clr(tr, h, K=2000))
check("I.4 larger d_safe -> larger (or equal) achieved clearance (monotone knob)",
      all(b >= a - 0.02 for a, b in zip(clrs, clrs[1:])),
      f"clearances at d_safe[0.6,0.8,1.0]={['%.3f' % c for c in clrs]}")


# ---------------------------------------------------------------- I.5 moving human
print("\n--- I.5 moving human: conservative segment-end time still clears densely ---")
p = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
mid = p[len(p) // 2].copy()
# human moving slowly across the path; t_rep = segment END (conservative)
mov = SphereObstacle(mid + np.array([0.0, 0.6, 0.0]), 0.4,
                     vel=np.array([0.0, -0.15, 0.0]), class_name="human")
tr_m, info_m = plan_minco(p, [mov], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
clr_m = dense_min_clr(tr_m, mov, K=2000)   # signed_dist evaluates the MOVING centre
check("I.5 moving-human dense clearance reported; valid arms clear >= d_safe",
      (not info_m["trajectory_valid"]) or clr_m >= D_SAFE - 1e-6,
      f"valid={info_m['trajectory_valid']} moving dense clr={clr_m:.4f}")


# ---------------------------------------------------------------- I.6 idempotence
print("\n--- I.6 re-solving from the converged trajectory stays valid ---")
p, h = scene(off=0.3, r=0.4)
tr1, info1 = plan_minco(p, [h], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
# feed the converged trajectory's waypoints back in as the A* seed
seed2 = tr1.eval(np.linspace(tr1.t_start, tr1.t_end, 12))
tr2, info2 = plan_minco(seed2, [h], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
check("I.6 re-solve from converged stays valid + cleared",
      info2["trajectory_valid"] and dense_min_clr(tr2, h, K=2000) >= D_SAFE - 1e-6,
      f"valid={info2['trajectory_valid']} clr={dense_min_clr(tr2, h, K=2000):.4f}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
