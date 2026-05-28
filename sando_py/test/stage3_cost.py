"""Stage 3 — local/cost.py functional baseline.

Coverage:
  A. empty obstacle list → cost = 0
  B. obstacle far away → cost = 0 (no penalty inside the safe band)
  C. obstacle on the path → cost > 0 and matches the EGO formula exactly
  D. AABB wall: far → 0, on the path → > 0
  E. multiple obstacles → cost is additive (= sum of single-obstacle costs)
  F. hard vs soft: same shape, only weight differs → ratio = weight ratio
  G. dynamic sphere: when it sweeps through the path, t-windowed cost rises
  H. EGO penalty is C² at d = d_safe (smooth, no discontinuity)
  I. class isolation: tweaking unused class params doesn't change the cost
  J. K (sample count) controls precision: more samples → cost converges

Standalone — loads modules directly. Run:  python3 test/stage3_cost.py
"""
import importlib.util
import os
import numpy as np

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local")

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.abspath(os.path.join(LOCAL_DIR, fname)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_bspline = _load("bspline_standalone", "bspline.py")
UniformBSpline = _bspline.UniformBSpline

# obstacles + cost rely on a relative `from .avoid_config import` — easiest
# is to install the local package onto sys.path so the relative imports work.
import sys
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams, default_config        # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.cost import obstacle_cost, penalty                    # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---- shared fixture: a straight-line spline along x from 0 to 10 -----------
def make_straight_spline(d=3, n_ctrl=14):
    """Spline whose eval is approximately a line x = t-mapped, y=z=0."""
    path = np.column_stack([np.linspace(0.0, 10.0, 60), np.zeros(60), np.zeros(60)])
    return UniformBSpline.fit_path(path, num_ctrl=n_ctrl, degree=5, dt=1.0, clamp_endpoints=True)


# ---------------------------------------------------------------- A. empty obstacles
print("\n--- A. empty obstacle list ---")
bs = make_straight_spline()
cfg = default_config()
check("A.0 obstacle_cost([], ...) == 0", obstacle_cost(bs, [], cfg) == 0.0)


# ---------------------------------------------------------------- B. obstacle far away
print("\n--- B. obstacle far from spline ---")
far_sphere = SphereObstacle(centre0=[5.0, 100.0, 0.0], radius=0.3, class_name="human")
c = obstacle_cost(bs, [far_sphere], cfg, K=200)
check("B.0 far sphere yields cost = 0", c == 0.0, f"cost={c:.2e}")


# ---------------------------------------------------------------- C. sphere on the path
print("\n--- C. sphere on the path: cost matches EGO formula ---")
# put a sphere right on the spline path at x=5, radius small enough so the
# spline (which lies on x-axis) intersects it near x=5.
sphere = SphereObstacle(centre0=[5.0, 0.0, 0.0], radius=0.2, class_name="human")
# build a small config so we can predict the cost easily
cfg_one = {"human": AvoidParams("human", mode="hard", d_safe=1.0, weight=100.0)}
K = 2001
c_actual = obstacle_cost(bs, [sphere], cfg_one, K=K)
# recompute the expected sum from the same sampling
ts = np.linspace(bs.t_start, bs.t_end, K)
pts = bs.eval(ts)
expected = 0.0
for p in pts:
    d = float(np.linalg.norm(p - sphere.centre0) - sphere.radius)
    diff = cfg_one["human"].d_safe - d
    if diff > 0:
        expected += cfg_one["human"].weight * diff ** 3
expected /= K  # cost is averaged over K samples
check("C.0 cost matches direct re-derivation of the EGO formula",
      np.isclose(c_actual, expected), f"actual={c_actual:.4e}, expected={expected:.4e}")
check("C.1 cost > 0 when the sphere overlaps the spline", c_actual > 0)


# ---------------------------------------------------------------- D. AABB wall
print("\n--- D. AABB wall behaviour ---")
wall_far = AABBObstacle(lo=[3.0, 50.0, -1.0], hi=[4.0, 51.0, 1.0], class_name="wall")
wall_on = AABBObstacle(lo=[4.9, -0.1, -0.1], hi=[5.1, 0.1, 0.1], class_name="wall")
c_far = obstacle_cost(bs, [wall_far], cfg, K=400)
c_on = obstacle_cost(bs, [wall_on], cfg, K=400)
check("D.0 wall far away → cost = 0", c_far == 0.0)
check("D.1 wall on the path → cost > 0", c_on > 0.0)


# ---------------------------------------------------------------- E. additivity
print("\n--- E. multiple obstacles are additive ---")
s1 = SphereObstacle([2.0, 0.0, 0.0], 0.15, class_name="human")
s2 = SphereObstacle([7.0, 0.0, 0.0], 0.20, class_name="human")
c1 = obstacle_cost(bs, [s1], cfg, K=600)
c2 = obstacle_cost(bs, [s2], cfg, K=600)
c12 = obstacle_cost(bs, [s1, s2], cfg, K=600)
check("E.0 cost(s1) + cost(s2) == cost([s1, s2])",
      np.isclose(c12, c1 + c2), f"single sum={c1+c2:.4e}, combined={c12:.4e}")


# ---------------------------------------------------------------- F. hard vs soft = pure weight ratio
print("\n--- F. hard vs soft is purely a weight ratio ---")
cfg_soft = {"x": AvoidParams("x", "soft", d_safe=0.5, weight=5.0)}
cfg_hard = {"x": AvoidParams("x", "hard", d_safe=0.5, weight=5000.0)}
obs = SphereObstacle([5.0, 0.0, 0.0], 0.1, class_name="x")
c_soft = obstacle_cost(bs, [obs], cfg_soft, K=300)
c_hard = obstacle_cost(bs, [obs], cfg_hard, K=300)
ratio = c_hard / c_soft if c_soft > 0 else float("inf")
check("F.0 hard/soft cost ratio == weight ratio (1000x)",
      np.isclose(ratio, 1000.0), f"ratio={ratio:.2f}")


# ---------------------------------------------------------------- G. dynamic sphere over time
print("\n--- G. dynamic sphere: cost depends on (t, predict(t)) ---")
# stationary version
s_static = SphereObstacle([5.0, 0.0, 0.0], 0.3, vel=None, class_name="human")
c_static = obstacle_cost(bs, [s_static], cfg, K=600)
# moving along +y so it sweeps OUT of the spline as t grows
v = (bs.t_end - bs.t_start)
s_moving = SphereObstacle([5.0, -10.0, 0.0], 0.3,
                          vel=[0.0, 20.0 / v, 0.0], class_name="human")
# centre(t_start) = (5, -10, 0), centre(t_end) = (5, +10, 0)
# at t_start it's far below the spline → little contribution there
# at t_end it's far above the spline → little contribution there
# in between it crosses y=0 at some t, briefly contributing
c_moving = obstacle_cost(bs, [s_moving], cfg, K=600)
check("G.0 moving sphere produces a finite, smaller cost than a stationary one on the path",
      0 < c_moving < c_static, f"static={c_static:.3e}, moving={c_moving:.3e}")
# also: stationary sphere predict(t) at any t returns the same point
p0 = s_static.predict(0.0)
p1 = s_static.predict(123.0)
check("G.1 static sphere predict(t) is time-invariant", np.allclose(p0, p1))


# ---------------------------------------------------------------- H. C² smoothness at d = d_safe
print("\n--- H. EGO penalty is C² at d = d_safe ---")
# at d = d_safe, penalty = 0; just inside it grows like (d_safe - d)^3
eps = 1e-3
p0 = penalty(0.5 + eps, 0.5)  # just outside → 0
p_at = penalty(0.5, 0.5)       # at the boundary → 0
p_in = penalty(0.5 - eps, 0.5)  # just inside → eps^3
check("H.0 penalty just outside d_safe == 0", p0 == 0.0)
check("H.1 penalty at d_safe == 0", p_at == 0.0)
check("H.2 penalty just inside ≈ eps^3", np.isclose(p_in, eps ** 3),
      f"got {p_in:.3e}, want {eps**3:.3e}")
# numerical 1st derivative of (d_safe-d)^3 is -3(d_safe-d)^2 → 0 at the boundary
h = 1e-5
fwd = (penalty(0.5 - h, 0.5) - penalty(0.5, 0.5)) / (-h)
check("H.3 d penalty/d d at the boundary ≈ 0 (C¹)", abs(fwd) < 1e-7,
      f"forward diff = {fwd:.3e}")


# ---------------------------------------------------------------- I. class isolation
print("\n--- I. tweaking an unused class doesn't change the cost ---")
cfg_a = {"wall": AvoidParams("wall", "soft", 0.4, 10.0),
         "human": AvoidParams("human", "hard", 0.8, 1e4)}
cfg_b = dict(cfg_a)
cfg_b["wall"] = AvoidParams("wall", "soft", d_safe=99.0, weight=1e9)  # massively perturbed
obs = SphereObstacle([5.0, 0.0, 0.0], 0.2, class_name="human")
ca = obstacle_cost(bs, [obs], cfg_a, K=300)
cb = obstacle_cost(bs, [obs], cfg_b, K=300)
check("I.0 cost unchanged when only the unused 'wall' params move", ca == cb,
      f"a={ca:.4e}, b={cb:.4e}")


# ---------------------------------------------------------------- J. K convergence
print("\n--- J. larger K refines the cost estimate (converges) ---")
sphere = SphereObstacle([5.0, 0.0, 0.0], 0.2, class_name="human")
c_low = obstacle_cost(bs, [sphere], cfg, K=50)
c_mid = obstacle_cost(bs, [sphere], cfg, K=500)
c_hi = obstacle_cost(bs, [sphere], cfg, K=5000)
# they should be of the same order; high-K is the closest to ground truth.
gap_low = abs(c_low - c_hi) / max(c_hi, 1e-12)
gap_mid = abs(c_mid - c_hi) / max(c_hi, 1e-12)
check("J.0 K=500 is closer to K=5000 than K=50 (sample-mean converges)",
      gap_mid < gap_low, f"gap50→5000={gap_low:.3f}, gap500→5000={gap_mid:.3f}")
check("J.1 all three K give finite, positive cost",
      all(np.isfinite(x) and x > 0 for x in (c_low, c_mid, c_hi)))


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
