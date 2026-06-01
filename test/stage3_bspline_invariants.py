"""Stage 3 — local/bspline.py geometric invariants + scipy derivative oracle.

Random batched checks (seeded). Each property MUST hold for any B-spline:

  I.0 translation invariance:   ctrl + c     -> eval shifts by c
  I.1 linearity:                a*P1 + b*P2  -> eval scales/adds the same way
  I.2 convex-hull property:     eval(t) ∈ axis-wise [min(ctrl), max(ctrl)]
  I.3 C^{p-1} continuity at internal knots (uniform knots, multiplicity 1)
  I.4 derivatives vs scipy.BSpline.derivative (strict oracle, all orders ≤ p)
  I.5 basis non-negative + partition of unity (Σ N_i(t) = 1)
  I.6 LSQ + clamp pins both endpoints (200 random paths)
  I.7 spline → sample → LSQ refit recovers the spline (M >> N+1)

Run:  python3 test/stage3_bspline_invariants.py
"""
import importlib.util
import os
import numpy as np
import scipy.interpolate as si

BS_PATH = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local", "bspline.py")
spec = importlib.util.spec_from_file_location("bspline_standalone", os.path.abspath(BS_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
UniformBSpline = mod.UniformBSpline

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- I.0 translation invariance
print("\n--- I.0 translation invariance (50 random configs) ---")
rng = np.random.default_rng(11)
bad, worst = 0, 0.0
for _ in range(50):
    p = int(rng.choice([3, 4, 5]))
    n_ctrl = int(rng.integers(p + 2, p + 20))
    dt = float(rng.uniform(0.2, 1.5))
    ctrl = rng.standard_normal((n_ctrl, 3))
    c = rng.standard_normal(3)
    bs1 = UniformBSpline(ctrl, p, dt)
    bs2 = UniformBSpline(ctrl + c, p, dt)
    ts = np.linspace(bs1.t_start, bs1.t_end, 20)
    err = float(np.max(np.abs((bs2.eval(ts) - bs1.eval(ts)) - c)))
    if err > 1e-10:
        bad += 1
    worst = max(worst, err)
check("I.0 ctrl + c shifts eval by c", bad == 0, f"worst|err|={worst:.2e}, bad={bad}/50")


# ---------------------------------------------------------------- I.1 linearity
print("\n--- I.1 linearity in ctrl (50 random configs) ---")
bad, worst = 0, 0.0
for _ in range(50):
    p = int(rng.choice([3, 4, 5]))
    n_ctrl = int(rng.integers(p + 2, p + 20))
    dt = float(rng.uniform(0.2, 1.5))
    P1 = rng.standard_normal((n_ctrl, 3))
    P2 = rng.standard_normal((n_ctrl, 3))
    a, b = float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))
    bs1 = UniformBSpline(P1, p, dt)
    bs2 = UniformBSpline(P2, p, dt)
    bsmix = UniformBSpline(a * P1 + b * P2, p, dt)
    ts = np.linspace(bs1.t_start, bs1.t_end, 20)
    err = float(np.max(np.abs(bsmix.eval(ts) - (a * bs1.eval(ts) + b * bs2.eval(ts)))))
    if err > 1e-9:
        bad += 1
    worst = max(worst, err)
check("I.1 eval is linear in ctrl", bad == 0, f"worst|err|={worst:.2e}, bad={bad}/50")


# ---------------------------------------------------------------- I.2 convex hull
print("\n--- I.2 convex-hull property (50 random configs) ---")
bad, worst = 0, 0.0
for _ in range(50):
    p = int(rng.choice([3, 4, 5]))
    n_ctrl = int(rng.integers(p + 2, p + 30))
    ctrl = rng.standard_normal((n_ctrl, 3)) * 3
    bs = UniformBSpline(ctrl, p, dt=1.0)
    ts = np.linspace(bs.t_start, bs.t_end, 50)
    vals = bs.eval(ts)
    lo = ctrl.min(axis=0) - 1e-10
    hi = ctrl.max(axis=0) + 1e-10
    breach = float(max(np.max(vals - hi), np.max(lo - vals)))
    if breach > 1e-9:
        bad += 1
    worst = max(worst, breach)
check("I.2 eval inside axis-wise ctrl box", bad == 0, f"worst breach={worst:.2e}, bad={bad}/50")


# ---------------------------------------------------------------- I.3 C^{p-1} continuity
print("\n--- I.3 C^{p-1} continuity at internal knots (40 random splines) ---")
# at each internal knot, derivatives of order 1..p-1 must be continuous
bad, worst = 0, 0.0
for _ in range(40):
    p = int(rng.choice([3, 5]))
    n_ctrl = int(rng.integers(p + 4, p + 15))
    dt = float(rng.uniform(0.3, 1.2))
    ctrl = rng.standard_normal((n_ctrl, 3))
    bs = UniformBSpline(ctrl, p, dt)
    internals = bs.knots[bs.p + 1 : bs.n + 1]
    eps = 1e-7
    for tk in internals:
        for order in range(1, p):
            left = bs.eval_deriv(float(tk - eps), order)
            right = bs.eval_deriv(float(tk + eps), order)
            err = float(np.max(np.abs(left - right)))
            if err > 1e-3:
                bad += 1
            worst = max(worst, err)
check("I.3 derivatives continuous up to order (p-1) at internal knots", bad == 0, f"worst|err|={worst:.2e}")


# ---------------------------------------------------------------- I.4 scipy derivative oracle
print("\n--- I.4 derivatives vs scipy.BSpline.derivative (30 configs × orders 1..p) ---")
bad, worst = 0, 0.0
for _ in range(30):
    p = int(rng.choice([3, 4, 5]))
    n_ctrl = int(rng.integers(p + 2, p + 15))
    dt = float(rng.uniform(0.3, 1.0))
    ctrl = rng.standard_normal((n_ctrl, 3)) * 2
    bs = UniformBSpline(ctrl, p, dt)
    oracle = si.BSpline(bs.knots, bs.ctrl, bs.p)
    # avoid the very edges where scipy's derivative extension can disagree
    margin = 0.5 * dt
    ts = np.linspace(bs.t_start + margin, bs.t_end - margin, 12)
    for order in range(1, p + 1):
        err = float(np.max(np.abs(bs.eval_deriv(ts, order) - oracle.derivative(order)(ts))))
        if err > 1e-9:
            bad += 1
        worst = max(worst, err)
check("I.4 eval_deriv == scipy.BSpline.derivative (orders 1..p)", bad == 0, f"worst|err|={worst:.2e}")


# ---------------------------------------------------------------- I.5 basis non-negativity + partition
print("\n--- I.5 basis non-negativity + partition of unity (30 configs × 50 t each) ---")
bad_neg, bad_sum, worst_neg, worst_sum = 0, 0, 0.0, 0.0
for _ in range(30):
    p = int(rng.choice([3, 4, 5]))
    n_ctrl = int(rng.integers(p + 2, p + 15))
    bs = UniformBSpline(np.zeros((n_ctrl, 3)), p, dt=1.0)
    for t in np.linspace(bs.t_start, bs.t_end, 50):
        _, vals = bs.nonzero_basis(float(t))
        if vals.min() < -1e-12:
            bad_neg += 1
            worst_neg = max(worst_neg, float(-vals.min()))
        s_err = abs(float(np.sum(vals)) - 1.0)
        if s_err > 1e-10:
            bad_sum += 1
        worst_sum = max(worst_sum, s_err)
check("I.5a basis non-negative", bad_neg == 0, f"worst neg={worst_neg:.2e}")
check("I.5b basis sums to 1", bad_sum == 0, f"worst|sum-1|={worst_sum:.2e}")


# ---------------------------------------------------------------- I.6 LSQ + clamp pin endpoints (batched)
print("\n--- I.6 LSQ + clamp_endpoints pins both ends (200 random paths) ---")
bad, worst = 0, 0.0
for _ in range(200):
    M = int(rng.integers(30, 200))
    p = 5
    n_ctrl = int(rng.integers(2 * (p + 1), 30))  # need 2*(p+1) ctrl pts for both-end clamp
    path = np.cumsum(rng.standard_normal((M, 3)), axis=0)
    bs = UniformBSpline.fit_path(path, n_ctrl, p, dt=1.0, clamp_endpoints=True)
    err_s = float(np.max(np.abs(bs.eval(bs.t_start) - path[0])))
    err_e = float(np.max(np.abs(bs.eval(bs.t_end) - path[-1])))
    if err_s > 1e-9 or err_e > 1e-9:
        bad += 1
    worst = max(worst, err_s, err_e)
check("I.6 clamp_endpoints pins both ends exactly", bad == 0, f"worst|err|={worst:.2e}, bad={bad}/200")


# ---------------------------------------------------------------- I.7 LSQ self-consistency
print("\n--- I.7 LSQ self-consistency: spline → samples → refit (30 random) ---")
bad, worst = 0, 0.0
for _ in range(30):
    p = int(rng.choice([3, 4, 5]))
    n_ctrl = int(rng.integers(p + 2, p + 12))
    dt = float(rng.uniform(0.3, 1.2))
    ctrl_true = rng.standard_normal((n_ctrl, 3))
    bs_true = UniformBSpline(ctrl_true, p, dt)
    M = max(200, 10 * n_ctrl)
    ts = np.linspace(bs_true.t_start, bs_true.t_end, M)
    samples = bs_true.eval(ts)
    bs_fit = UniformBSpline.fit_path(samples, n_ctrl, p, dt=dt)
    err = float(np.max(np.abs(bs_fit.eval(ts) - samples)))
    if err > 1e-8:
        bad += 1
    worst = max(worst, err)
check("I.7 refit from own samples recovers the spline", bad == 0, f"worst|err|={worst:.2e}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
