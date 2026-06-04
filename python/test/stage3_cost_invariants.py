"""Stage 3 — local/cost.py invariants + gradient oracle + random batches.

  I.0 translation invariance: shift spline + every obstacle by c → cost unchanged
  I.1 weight linearity:       cost ∝ weight (fix everything else)
  I.2 d_safe monotonicity:    larger d_safe → cost not smaller (same scene)
  I.3 class additivity:       total cost = Σ per-class subtotal
  I.4 non-negative:           cost ≥ 0 on 200 random scenes
  I.5 zero outside safe band: obstacle pushed far → cost = 0 (batch)
  I.6 1st-order Taylor consistency: cost is differentiable in ctrl (gradient oracle)
  I.7 K convergence:          larger K refines toward a limit (batch)

Run:  python3 test/stage3_cost_invariants.py
"""
import importlib.util
import os
import sys
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

from sando_py.local.avoid_config import AvoidParams                       # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.cost import obstacle_cost                             # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

rng = np.random.default_rng(2026)


def random_spline(n_ctrl=14):
    """Random 3D quintic spline whose extent fits in ~ [-3, 3]^3."""
    path = np.cumsum(rng.standard_normal((50, 3)) * 0.3, axis=0)
    return UniformBSpline.fit_path(path, num_ctrl=n_ctrl, degree=5, dt=1.0, clamp_endpoints=True)


def random_obstacles(n_sphere=2, n_aabb=1):
    obs = []
    for _ in range(n_sphere):
        c = rng.uniform(-5, 5, 3)
        r = float(rng.uniform(0.1, 0.6))
        obs.append(SphereObstacle(c, r, class_name="human"))
    for _ in range(n_aabb):
        lo = rng.uniform(-5, 4, 3)
        hi = lo + rng.uniform(0.2, 1.0, 3)
        obs.append(AABBObstacle(lo, hi, class_name="wall"))
    return obs


def default_cfg():
    return {"wall":  AvoidParams("wall",  "soft", 0.4, 10.0),
            "human": AvoidParams("human", "hard", 0.8, 1.0e4)}


# ---------------------------------------------------------------- I.0 translation invariance
print("\n--- I.0 translation invariance (50 random scenes) ---")
bad, worst = 0, 0.0
for _ in range(50):
    bs = random_spline()
    obs = random_obstacles()
    cfg = default_cfg()
    c0 = obstacle_cost(bs, obs, cfg, K=400)
    delta = rng.standard_normal(3)
    bs_s = UniformBSpline(bs.ctrl + delta, bs.p, bs.dt)
    obs_s = []
    for o in obs:
        if isinstance(o, SphereObstacle):
            obs_s.append(SphereObstacle(o.centre0 + delta, o.radius, o.vel, o.class_name))
        else:
            obs_s.append(AABBObstacle(o.lo + delta, o.hi + delta, o.class_name))
    c1 = obstacle_cost(bs_s, obs_s, cfg, K=400)
    rel = abs(c1 - c0) / max(abs(c0), 1e-9)
    if rel > 1e-9:
        bad += 1
    worst = max(worst, rel)
check("I.0 cost invariant under joint translation", bad == 0, f"worst rel|err|={worst:.2e}, bad={bad}/50")


# ---------------------------------------------------------------- I.1 weight linearity
print("\n--- I.1 weight linearity (40 random scenes) ---")
bad, worst = 0, 0.0
for _ in range(40):
    bs = random_spline()
    obs = random_obstacles(n_sphere=1, n_aabb=0)
    cfg1 = {"human": AvoidParams("human", "hard", 0.8, 1.0)}
    cfg2 = {"human": AvoidParams("human", "hard", 0.8, 17.5)}
    c1 = obstacle_cost(bs, obs, cfg1, K=300)
    c2 = obstacle_cost(bs, obs, cfg2, K=300)
    if c1 == 0.0:
        continue  # obstacle didn't hit the band — skip degenerate case
    ratio = c2 / c1
    rel = abs(ratio - 17.5) / 17.5
    if rel > 1e-9:
        bad += 1
    worst = max(worst, rel)
check("I.1 cost is linear in weight", bad == 0, f"worst rel|err|={worst:.2e}")


# ---------------------------------------------------------------- I.2 d_safe monotonicity
print("\n--- I.2 d_safe monotonicity (40 random scenes) ---")
bad = 0
for _ in range(40):
    bs = random_spline()
    obs = random_obstacles(n_sphere=2, n_aabb=1)
    cfg_lo = {"human": AvoidParams("human", "hard", 0.3, 1e3),
              "wall":  AvoidParams("wall",  "soft", 0.2, 10.0)}
    cfg_hi = {"human": AvoidParams("human", "hard", 1.5, 1e3),
              "wall":  AvoidParams("wall",  "soft", 1.5, 10.0)}
    c_lo = obstacle_cost(bs, obs, cfg_lo, K=300)
    c_hi = obstacle_cost(bs, obs, cfg_hi, K=300)
    if c_hi + 1e-12 < c_lo:
        bad += 1
check("I.2 cost monotone non-decreasing in d_safe", bad == 0, f"violations={bad}/40")


# ---------------------------------------------------------------- I.3 class additivity
print("\n--- I.3 class additivity (30 random scenes) ---")
bad, worst = 0, 0.0
for _ in range(30):
    bs = random_spline()
    sph = [SphereObstacle(rng.uniform(-5, 5, 3), float(rng.uniform(0.1, 0.4)),
                          class_name="human") for _ in range(2)]
    boxes = [AABBObstacle(*(lambda lo: (lo, lo + rng.uniform(0.2, 0.8, 3)))(rng.uniform(-5, 4, 3)),
                          class_name="wall") for _ in range(2)]
    cfg = default_cfg()
    c_h = obstacle_cost(bs, sph, cfg, K=300)
    c_w = obstacle_cost(bs, boxes, cfg, K=300)
    c_both = obstacle_cost(bs, sph + boxes, cfg, K=300)
    rel = abs(c_both - (c_h + c_w)) / max(c_both, 1e-9)
    if rel > 1e-9:
        bad += 1
    worst = max(worst, rel)
check("I.3 cost = Σ per-class subtotal", bad == 0, f"worst rel={worst:.2e}")


# ---------------------------------------------------------------- I.4 non-negative
print("\n--- I.4 non-negative cost (200 random scenes) ---")
neg = 0
worst_neg = 0.0
for _ in range(200):
    bs = random_spline()
    obs = random_obstacles(n_sphere=int(rng.integers(0, 4)), n_aabb=int(rng.integers(0, 3)))
    c = obstacle_cost(bs, obs, default_cfg(), K=200)
    if c < 0:
        neg += 1
        worst_neg = min(worst_neg, c)
check("I.4 cost ≥ 0 on all random scenes", neg == 0, f"violations={neg}/200, min={worst_neg:.2e}")


# ---------------------------------------------------------------- I.5 zero outside safe band
print("\n--- I.5 obstacle pushed far away → cost = 0 (50 batch) ---")
bad = 0
for _ in range(50):
    bs = random_spline()
    # spline lives near ||p|| ≲ 6; place obstacles at radius 100
    centre = rng.standard_normal(3); centre = centre / np.linalg.norm(centre) * 100.0
    obs = [SphereObstacle(centre, 0.5, class_name="human")]
    c = obstacle_cost(bs, obs, default_cfg(), K=200)
    if c != 0.0:
        bad += 1
check("I.5 cost = 0 when every obstacle is far outside the safe band", bad == 0)


# ---------------------------------------------------------------- I.6 1st-order Taylor consistency
print("\n--- I.6 cost is differentiable in ctrl (Taylor 1st-order check) ---")
# at a point where the obstacle is actually inside the safe band so gradient != 0
bs = UniformBSpline.fit_path(
    np.column_stack([np.linspace(0, 10, 60), np.zeros(60), np.zeros(60)]),
    num_ctrl=14, degree=5, dt=1.0, clamp_endpoints=True,
)
obs = [SphereObstacle([5.0, 0.0, 0.0], 0.2, class_name="human")]
cfg = {"human": AvoidParams("human", "hard", 1.0, 100.0)}
ctrl_shape = bs.ctrl.shape
ctrl0 = bs.ctrl.copy().ravel()

def cost_of(x):
    return obstacle_cost(
        UniformBSpline(x.reshape(ctrl_shape), bs.p, bs.dt),
        obs, cfg, K=200,
    )

g = scipy.optimize.approx_fprime(ctrl0, cost_of, 1e-5)
finite = np.all(np.isfinite(g))
check("I.6a finite-difference gradient is finite everywhere", finite,
      f"max|g|={float(np.max(np.abs(g))):.2e}")
# Taylor: f(x + eps*v) ≈ f(x) + eps * g.v   (remainder O(eps^2))
v = rng.standard_normal(ctrl0.size); v = v / np.linalg.norm(v)
eps = 1e-4
f0 = cost_of(ctrl0)
predicted = f0 + eps * float(g @ v)
actual = cost_of(ctrl0 + eps * v)
rel_err = abs(predicted - actual) / max(abs(actual), 1e-9)
check("I.6b 1st-order Taylor consistency holds", rel_err < 1e-2,
      f"rel|pred-actual|={rel_err:.2e}")


# ---------------------------------------------------------------- I.7 K convergence
print("\n--- I.7 K convergence (30 random scenes, K=1000 vs K=4000) ---")
bad, worst = 0, 0.0
for _ in range(30):
    bs = random_spline()
    obs = random_obstacles(n_sphere=1, n_aabb=1)
    cfg = default_cfg()
    c_mid = obstacle_cost(bs, obs, cfg, K=1000)
    c_hi = obstacle_cost(bs, obs, cfg, K=4000)
    if c_hi == 0:
        continue
    rel = abs(c_mid - c_hi) / c_hi
    if rel > 0.05:  # 5% noise budget — sample-mean of a step-like field
        bad += 1
    worst = max(worst, rel)
check("I.7 K=1000 within 5% of K=4000 on random scenes",
      bad == 0, f"worst rel={worst:.2%}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
