"""Stage 3 — local/cost.py extreme boundaries + degenerate inputs.

  B.0 K = 1 (single sample) doesn't crash
  B.1 K very large (10000) stays finite and matches a reference
  B.2 d_safe = 0 → penalty fires only when d < 0 (penetration)
  B.3 weight = 0 → cost = 0 regardless of geometry
  B.4 negative d_safe → cost = 0 (no obstacle is "closer than -X" outside)
  B.5 radius = 0 sphere (point obstacle) works
  B.6 AABB lo == hi (degenerate point) works
  B.7 AABB hi < lo on any axis → ValueError at construction
  B.8 obstacle entirely contains the spline → cost large but finite, no inf/nan
  B.9 unknown class_name in config → KeyError
  B.10 mixed classes coexist correctly
  B.11 minimum spline (num_ctrl = degree+1) supported by cost
  B.12 K specified explicitly overrides the default
"""
import importlib.util
import os
import sys
import numpy as np

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local")
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(os.path.join(LOCAL_DIR, fname)))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

_bs = _load("bspline_standalone", "bspline.py")
UniformBSpline = _bs.UniformBSpline

from sando_py.local.avoid_config import AvoidParams                       # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.cost import obstacle_cost, penalty                    # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


def straight_spline(n_ctrl=14):
    path = np.column_stack([np.linspace(0, 10, 60), np.zeros(60), np.zeros(60)])
    return UniformBSpline.fit_path(path, num_ctrl=n_ctrl, degree=5, dt=1.0, clamp_endpoints=True)


bs = straight_spline()
cfg = {"human": AvoidParams("human", "hard", 0.8, 1e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 10.0)}


# ---------------------------------------------------------------- B.0 K=1
print("\n--- B.0 K = 1 (single sample) ---")
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.2, class_name="human")]
c = obstacle_cost(bs, obs, cfg, K=1)
check("B.0 K=1 produces a finite cost", np.isfinite(c) and c >= 0, f"cost={c:.3e}")


# ---------------------------------------------------------------- B.1 K very large
print("\n--- B.1 K = 10000 ---")
c = obstacle_cost(bs, obs, cfg, K=10000)
ref = obstacle_cost(bs, obs, cfg, K=2000)
check("B.1 K=10000 finite + within 5% of K=2000", np.isfinite(c) and abs(c - ref) / ref < 0.05,
      f"c={c:.4e}, ref={ref:.4e}")


# ---------------------------------------------------------------- B.2 d_safe = 0
print("\n--- B.2 d_safe = 0 (penalty only on penetration) ---")
cfg_zero = {"human": AvoidParams("human", "hard", 0.0, 1.0)}
obs_touch = [SphereObstacle([5.0, 0.21, 0.0], 0.2, class_name="human")]  # surface ≈ 0.01 from path
obs_deep = [SphereObstacle([5.0, 0.0, 0.0], 0.5, class_name="human")]    # path well inside
c_touch = obstacle_cost(bs, obs_touch, cfg_zero, K=500)
c_deep = obstacle_cost(bs, obs_deep, cfg_zero, K=500)
check("B.2a d_safe=0, just outside surface → cost = 0", c_touch == 0.0, f"cost={c_touch:.3e}")
check("B.2b d_safe=0, deep penetration → cost > 0", c_deep > 0)


# ---------------------------------------------------------------- B.3 weight = 0
print("\n--- B.3 weight = 0 ---")
cfg_zero_w = {"human": AvoidParams("human", "hard", 1.0, 0.0)}
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.5, class_name="human")]
c = obstacle_cost(bs, obs, cfg_zero_w, K=300)
check("B.3 weight=0 → cost = 0 regardless of geometry", c == 0.0)


# ---------------------------------------------------------------- B.4 negative d_safe
print("\n--- B.4 negative d_safe ---")
cfg_neg = {"human": AvoidParams("human", "hard", -0.5, 100.0)}
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.3, class_name="human")]  # path inside
# d at sample on path ≈ -0.3 ; diff = d_safe - d = -0.5 - (-0.3) = -0.2 < 0 → no penalty
c = obstacle_cost(bs, obs, cfg_neg, K=300)
check("B.4 negative d_safe yields 0 cost even on penetration shallower than |d_safe|",
      c == 0.0, f"cost={c:.3e}")


# ---------------------------------------------------------------- B.5 point obstacle (radius=0)
print("\n--- B.5 sphere with radius = 0 (point) ---")
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.0, class_name="human")]
c = obstacle_cost(bs, obs, cfg, K=300)
check("B.5 radius=0 sphere works (finite, > 0 on path)", np.isfinite(c) and c > 0, f"cost={c:.3e}")


# ---------------------------------------------------------------- B.6 AABB lo == hi
print("\n--- B.6 AABB lo == hi (point box) ---")
obs = [AABBObstacle([5.0, 0.0, 0.0], [5.0, 0.0, 0.0], class_name="wall")]
c = obstacle_cost(bs, obs, cfg, K=300)
check("B.6 degenerate AABB works (finite, > 0)", np.isfinite(c) and c > 0, f"cost={c:.3e}")


# ---------------------------------------------------------------- B.7 AABB hi < lo
print("\n--- B.7 AABB hi < lo on any axis raises ---")
check("B.7 AABB hi < lo raises", raises(lambda: AABBObstacle([1, 0, 0], [0, 1, 1])))


# ---------------------------------------------------------------- B.8 obstacle contains spline
print("\n--- B.8 huge obstacle contains the spline entirely ---")
obs = [SphereObstacle([5.0, 0.0, 0.0], 100.0, class_name="human")]  # huge sphere
c = obstacle_cost(bs, obs, cfg, K=300)
check("B.8 huge obstacle: cost finite (no inf/nan)", np.isfinite(c) and c > 0, f"cost={c:.3e}")


# ---------------------------------------------------------------- B.9 unknown class_name
print("\n--- B.9 unknown class_name in config ---")
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.2, class_name="alien")]
check("B.9 unknown class raises KeyError",
      raises(lambda: obstacle_cost(bs, obs, cfg, K=200), KeyError))


# ---------------------------------------------------------------- B.10 mixed classes
print("\n--- B.10 mixed classes coexist ---")
mixed = [SphereObstacle([3.0, 0.0, 0.0], 0.15, class_name="human"),
         AABBObstacle([6.9, -0.1, -0.1], [7.1, 0.1, 0.1], class_name="wall")]
c_mix = obstacle_cost(bs, mixed, cfg, K=400)
c_h = obstacle_cost(bs, [mixed[0]], cfg, K=400)
c_w = obstacle_cost(bs, [mixed[1]], cfg, K=400)
check("B.10 mixed-class cost = sum of per-class costs",
      np.isclose(c_mix, c_h + c_w), f"mix={c_mix:.4e}, sum={c_h+c_w:.4e}")


# ---------------------------------------------------------------- B.11 minimum spline (p+1 ctrl)
print("\n--- B.11 minimum spline (num_ctrl = degree+1) ---")
ctrl_min = np.random.default_rng(0).standard_normal((6, 3))  # 5-degree needs 6 ctrl pts
bs_min = UniformBSpline(ctrl_min, degree=5, dt=1.0)
obs = [SphereObstacle(bs_min.eval(0.5 * (bs_min.t_start + bs_min.t_end)), 0.3, class_name="human")]
c = obstacle_cost(bs_min, obs, cfg, K=200)
check("B.11 minimum-size spline works with cost", np.isfinite(c), f"cost={c:.3e}")


# ---------------------------------------------------------------- B.12 explicit K overrides default
print("\n--- B.12 explicit K override ---")
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.3, class_name="human")]
c_default = obstacle_cost(bs, obs, cfg)  # K defaults to ~140 (10*num_ctrl=140)
c_explicit = obstacle_cost(bs, obs, cfg, K=2000)
# values differ (different K → different mean), but both finite + > 0
check("B.12 K override changes value but both finite",
      np.isfinite(c_default) and np.isfinite(c_explicit) and c_default != c_explicit)


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
