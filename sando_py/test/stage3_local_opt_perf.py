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
from sando_py.local.local_opt import plan, OptParams, DetourConfig        # noqa: E402

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


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    raise SystemExit(1)
