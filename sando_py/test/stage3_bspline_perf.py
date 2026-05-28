"""Stage 3 — local/bspline.py performance gates.

Thresholds are intentionally generous (machine-dependent); failure here means
an algorithmic regression (something went quadratic / O(N²) / hangs), not a
microbenchmark fail. Caps at ~30 s per case.

  P.0 large spline (N+1 = 500 ctrl pts) evaluated at 1000 t  — under cap
  P.1 LSQ fit M = 2000 path → 100 ctrl pts                   — under cap
  P.2 high degree p = 7 (build + eval + derivative)          — under cap
  P.3 derivative orders 1..p on a moderate spline finite     — under cap
  P.4 dense derivative evaluation 10k samples                — under cap

Run:  python3 test/stage3_bspline_perf.py
"""
import importlib.util
import os
import time
import numpy as np

BS_PATH = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local", "bspline.py")
spec = importlib.util.spec_from_file_location("bspline_standalone", os.path.abspath(BS_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
UniformBSpline = mod.UniformBSpline

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

CAP_SECS = 30.0
rng = np.random.default_rng(2026)


# ---------------------------------------------------------------- P.0
print("\n--- P.0 large spline: N+1=500 ctrl, evaluate 1000 t ---")
ctrl = rng.standard_normal((500, 3))
bs = UniformBSpline(ctrl, degree=5, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 1000)
t0 = time.time()
vals = bs.eval(ts)
elapsed = time.time() - t0
check(f"P.0 eval 500-ctrl × 1000 t in < {CAP_SECS}s",
      elapsed < CAP_SECS and np.all(np.isfinite(vals)), f"{elapsed:.2f}s")


# ---------------------------------------------------------------- P.1
print("\n--- P.1 LSQ fit M=2000 → 100 ctrl ---")
path = np.cumsum(rng.standard_normal((2000, 3)) * 0.05, axis=0)
t0 = time.time()
bs = UniformBSpline.fit_path(path, num_ctrl=100, degree=5, dt=1.0, clamp_endpoints=True)
elapsed = time.time() - t0
check(f"P.1 LSQ M=2000 → 100 ctrl in < {CAP_SECS}s",
      elapsed < CAP_SECS and np.all(np.isfinite(bs.ctrl)), f"{elapsed:.2f}s")


# ---------------------------------------------------------------- P.2
print("\n--- P.2 high degree p=7 ---")
ctrl = rng.standard_normal((30, 3))
t0 = time.time()
bs = UniformBSpline(ctrl, degree=7, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 200)
vals = bs.eval(ts)
d3 = bs.eval_deriv(ts, order=3)
elapsed = time.time() - t0
check(f"P.2 p=7 eval + 3rd deriv in < {CAP_SECS}s",
      elapsed < CAP_SECS and np.all(np.isfinite(vals)) and np.all(np.isfinite(d3)),
      f"{elapsed:.2f}s")


# ---------------------------------------------------------------- P.3
print("\n--- P.3 derivative orders 1..p all finite ---")
ctrl = rng.standard_normal((20, 3))
bs = UniformBSpline(ctrl, degree=5, dt=1.0)
ts = np.linspace(bs.t_start + 0.5, bs.t_end - 0.5, 100)
t0 = time.time()
all_finite = True
for order in range(1, bs.p + 1):
    d = bs.eval_deriv(ts, order=order)
    if not np.all(np.isfinite(d)):
        all_finite = False
        break
elapsed = time.time() - t0
check(f"P.3 all derivative orders 1..p finite in < {CAP_SECS}s",
      all_finite and elapsed < CAP_SECS, f"{elapsed:.2f}s")


# ---------------------------------------------------------------- P.4
print("\n--- P.4 dense derivative evaluation 10k samples ---")
bs = UniformBSpline(rng.standard_normal((50, 3)), degree=5, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 10000)
t0 = time.time()
d2 = bs.eval_deriv(ts, order=2)
elapsed = time.time() - t0
check(f"P.4 10k 2nd-derivative samples in < {CAP_SECS}s",
      np.all(np.isfinite(d2)) and elapsed < CAP_SECS, f"{elapsed:.2f}s")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    raise SystemExit(1)
