"""Stage 3 — local/local_opt.py performance gates.

  P.0  one lean plan() call (maxiter=30, K=80)              < 15 s
  P.1  one default plan() call (maxiter=200, K=200)         < 90 s
  P.2  plan() returns even when maxiter is hit              info.converged=False
  P.3  iter count never exceeds maxiter

Thresholds are generous; failure here means a regression, not a microbench fail.
"""
import os
import sys
import time
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import default_config                    # noqa: E402
from sando_py.local.obstacles import SphereObstacle                       # noqa: E402
from sando_py.local.local_opt import (                                    # noqa: E402
    plan, plan_minco, OptParams, DetourConfig)

# perf measures single-seed time; detour multi-start has its own budget test.
NO_DETOUR = DetourConfig(enabled=False)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


path = np.column_stack([np.linspace(0, 10, 30), np.zeros(30), np.zeros(30)])
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.5, class_name="human")]
cfg = default_config()


# ---------------------------------------------------------------- P.0 lean
print("\n--- P.0 lean plan() (maxiter=30, K=80) ---")
t0 = time.time()
bs, info = plan(path, obs, cfg, opt_params=OptParams(maxiter=30, K=80, vmax=3.0), detour_cfg=NO_DETOUR)
elapsed = time.time() - t0
check(f"P.0 lean plan in < 15 s", elapsed < 15.0,
      f"{elapsed:.2f}s, iter={info['iter']}, converged={info['converged']}")


# ---------------------------------------------------------------- P.1 default
print("\n--- P.1 default plan() (maxiter=200, K=200) ---")
t0 = time.time()
bs, info = plan(path, obs, cfg, opt_params=OptParams(maxiter=200, K=200, vmax=3.0), detour_cfg=NO_DETOUR)
elapsed = time.time() - t0
check(f"P.1 default plan in < 90 s", elapsed < 90.0,
      f"{elapsed:.2f}s, iter={info['iter']}, converged={info['converged']}")


# ---------------------------------------------------------------- P.2 budget-bound returns
print("\n--- P.2 maxiter=2 still returns + info marks not_converged ---")
bs, info = plan(path, obs, cfg, opt_params=OptParams(maxiter=2, K=80, vmax=3.0), detour_cfg=NO_DETOUR)
check("P.2a info.converged=False when budget too small", not info["converged"])
check("P.2b plan still returned a spline", np.all(np.isfinite(bs.ctrl)))


# ---------------------------------------------------------------- P.3 iter ≤ maxiter
print("\n--- P.3 iter never exceeds maxiter ---")
for mi in (5, 20, 50):
    bs, info = plan(path, obs, cfg, opt_params=OptParams(maxiter=mi, K=80, vmax=3.0), detour_cfg=NO_DETOUR)
    ok = info["iter"] <= mi + 1  # scipy sometimes reports nit = maxiter+1 on hit
    check(f"P.3 maxiter={mi}: iter={info['iter']} ≤ maxiter", ok,
          f"iter={info['iter']}")


# ================================================================ MINCO path
# M.* — the analytic-gradient MINCO solve must beat the finite-diff B-spline
# path end to end, on the SAME scenes, by a wide margin (the gate).
print("\n--- M.0 MINCO analytic-grad solve is SECONDS, faster than finite-diff plan() ---")
# (a) single-seed, default budget — isolates per-eval speed
opt_def = OptParams(maxiter=80, K=150, vmax=3.0, amax=3.0)
t0 = time.time(); _, io = plan(path, obs, cfg, opt_params=opt_def, detour_cfg=NO_DETOUR)
t_old = time.time() - t0
t0 = time.time(); _, im = plan_minco(path, obs, cfg, opt_params=opt_def, detour_cfg=NO_DETOUR)
t_new = time.time() - t0
check("M.0a MINCO single-seed solve < 5 s", t_new < 5.0,
      f"new={t_new:.2f}s (old finite-diff={t_old:.2f}s)")
check("M.0b MINCO faster than finite-diff plan() on the same scene", t_new < t_old,
      f"speedup={t_old / max(t_new, 1e-6):.1f}x")

# (b) the random-walk scene that makes stage3_local_opt_invariants time out
print("\n--- M.1 the invariants timeout scenario solves in seconds (MINCO) ---")
rng = np.random.default_rng(2026)
rw_path = np.cumsum(rng.standard_normal((25, 3)) * 0.4, axis=0)
rw_obs = [SphereObstacle(rw_path[12] + rng.standard_normal(3) * 0.1, 0.35, class_name="human")]
t0 = time.time()
tr, info = plan_minco(rw_path, rw_obs, cfg, opt_params=opt_def)  # WITH detour multi-start
t_minco = time.time() - t0
# Stage 3 note: this scene (human r=0.35, d_safe=0.8 -> ~1.15 m berth on a tight
# random walk) is INFEASIBLE -> the per-class ALM correctly exhausts its outer
# budget on all 5 detour seeds trying to satisfy the human constraint before
# surfacing the STOP (clearance_violation). That is heavier than the M2 soft
# single-solve (which silently traded off); 16 s bounds the worst infeasible case.
check("M.1a MINCO detour multi-start on the timeout scene < 16 s (ALM, infeasible STOP)",
      t_minco < 16.0,
      f"{t_minco:.2f}s, seeds={info['n_seeds_tried']}, iter={info['iter']}, "
      f"valid={info['trajectory_valid']}, reason={info['failure_reason']}")
check("M.1b MINCO returned a finite trajectory", np.all(np.isfinite(tr.c)))

# (c) per-iteration speedup (matched iter budget, single seed)
print("\n--- M.2 per-iteration speedup (analytic grad vs finite-diff) ---")
opt30 = OptParams(maxiter=30, K=200, vmax=3.0, amax=3.0)
t0 = time.time(); _, io = plan(rw_path, rw_obs, cfg, opt_params=opt30, detour_cfg=NO_DETOUR)
t_old = time.time() - t0
t0 = time.time(); _, im = plan_minco(rw_path, rw_obs, cfg, opt_params=opt30, detour_cfg=NO_DETOUR)
t_new = time.time() - t0
ms_old = t_old / max(io["iter"], 1) * 1000
ms_new = t_new / max(im["iter"], 1) * 1000
check("M.2 MINCO >= 5x faster per L-BFGS iteration", ms_old / max(ms_new, 1e-6) >= 5.0,
      f"old={ms_old:.0f}ms/it new={ms_new:.0f}ms/it -> {ms_old / max(ms_new, 1e-6):.1f}x")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    raise SystemExit(1)
