"""Stage 3 — local/cost.py performance gates (catch hangs / blow-ups).

L-BFGS calls cost (and its finite-diff gradient) hundreds of times per
replan — so we want every single call to be fast, not just one-shot.
Thresholds are generous; failures here mean an algorithmic regression.

  P.0 K = 500, 1 obstacle, 1 cost call
  P.1 K = 2000, 50 obstacles, 1 cost call
  P.2 200 cost calls (LBFGS-budget mimic) — cumulative
  P.3 finite-diff gradient over ctrl (scipy.optimize.approx_fprime)
"""
import importlib.util
import os
import sys
import time
import numpy as np
import scipy.optimize

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local")
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(os.path.join(LOCAL_DIR, fname)))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

_bs = _load("bspline_standalone", "bspline.py")
UniformBSpline = _bs.UniformBSpline

from sando_py.local.avoid_config import AvoidParams, default_config       # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.cost import obstacle_cost                             # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

CAP_SECS = 30.0
rng = np.random.default_rng(2026)


def make_spline(n_ctrl=30):
    path = np.cumsum(rng.standard_normal((100, 3)) * 0.3, axis=0)
    return UniformBSpline.fit_path(path, num_ctrl=n_ctrl, degree=5, dt=1.0, clamp_endpoints=True)


def make_obstacles(n):
    obs = []
    for _ in range(n):
        c = rng.uniform(-5, 5, 3)
        r = float(rng.uniform(0.1, 0.5))
        obs.append(SphereObstacle(c, r, class_name="human"))
    return obs


# ---------------------------------------------------------------- P.0
print("\n--- P.0 single cost call, K=500, 1 obstacle ---")
bs = make_spline()
obs = make_obstacles(1)
cfg = default_config()
t0 = time.time()
c = obstacle_cost(bs, obs, cfg, K=500)
elapsed = time.time() - t0
check(f"P.0 K=500, 1 obs in < {CAP_SECS}s", elapsed < CAP_SECS and np.isfinite(c), f"{elapsed:.3f}s")


# ---------------------------------------------------------------- P.1
print("\n--- P.1 K=2000, 50 obstacles ---")
obs50 = make_obstacles(50)
t0 = time.time()
c = obstacle_cost(bs, obs50, cfg, K=2000)
elapsed = time.time() - t0
check(f"P.1 K=2000, 50 obs in < {CAP_SECS}s", elapsed < CAP_SECS and np.isfinite(c), f"{elapsed:.3f}s")


# ---------------------------------------------------------------- P.2
print("\n--- P.2 200 cost calls (LBFGS-budget mimic) ---")
obs5 = make_obstacles(5)
t0 = time.time()
for _ in range(200):
    obstacle_cost(bs, obs5, cfg, K=300)
elapsed = time.time() - t0
check(f"P.2 200 cost calls in < {CAP_SECS}s", elapsed < CAP_SECS, f"{elapsed:.3f}s")


# ---------------------------------------------------------------- P.3
print("\n--- P.3 finite-diff gradient over ctrl (scipy.optimize.approx_fprime) ---")
obs = [SphereObstacle([0.0, 0.0, 0.0], 0.5, class_name="human")]
cfg_one = {"human": AvoidParams("human", "hard", 1.0, 100.0)}
ctrl0 = bs.ctrl.copy().ravel()
shp = bs.ctrl.shape

def f(x):
    return obstacle_cost(UniformBSpline(x.reshape(shp), bs.p, bs.dt), obs, cfg_one, K=200)

t0 = time.time()
g = scipy.optimize.approx_fprime(ctrl0, f, 1e-5)
elapsed = time.time() - t0
check(f"P.3 finite-diff gradient ({ctrl0.size} dim) in < {CAP_SECS}s",
      elapsed < CAP_SECS and np.all(np.isfinite(g)),
      f"{elapsed:.3f}s, dim={ctrl0.size}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    raise SystemExit(1)
