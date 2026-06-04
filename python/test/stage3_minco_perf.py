"""Stage 3 — local/minco.py PERFORMANCE (O(M) banded solve).

Run:  python3 test/stage3_minco_perf.py

The headline correctness risk is that the banded solve silently degrades to a
dense O(M^3) one. We assert:
  A. solve time scales ~linearly when M doubles, NOT cubically.
  B. a banded solver (scipy.linalg.solve_banded) is actually used, and the
     stored matrix has bandwidth 6 (no entries outside the band).
  C. absolute solve is fast (large M solves in well under a dense-inverse budget).
  D. accuracy is preserved at large M (waypoints still exact).
"""
import importlib.util
import os
import time
import numpy as np

MOD_PATH = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local", "minco.py")
spec = importlib.util.spec_from_file_location("minco_standalone", os.path.abspath(MOD_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MinjerkTraj = mod.MinjerkTraj

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def build_time(M, reps=5):
    rng = np.random.default_rng(M)
    wp = rng.standard_normal((M+1, 3))
    T = rng.uniform(0.5, 1.5, size=M)
    # warm
    MinjerkTraj(wp, T)
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        MinjerkTraj(wp, T)
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------- A. scaling
print("\n--- A. O(M) scaling (doubling M) ---")
Ms = [200, 400, 800, 1600]
times = [build_time(M) for M in Ms]
for M, t in zip(Ms, times):
    print(f"    M={M:5d}  build={t*1e3:8.3f} ms")
# ratios between successive doublings — O(M) => ~2, O(M^2) => ~4, O(M^3) => ~8
ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
print(f"    doubling ratios: {[f'{r:.2f}' for r in ratios]}")
avg_ratio = float(np.mean(ratios))
# allow generous slack for Python overhead / timer noise, but cubic (≈8) must fail
check("A.0 doubling ratio < 3.5 (linear-ish, not quadratic/cubic)",
      avg_ratio < 3.5, f"avg ratio={avg_ratio:.2f}")
# total scaling 200->1600 (8x M). linear ~8x, cubic ~512x.
total_scale = times[-1] / times[0]
check("A.1 8x M => total time growth < 25x (rules out O(M^3))",
      total_scale < 25.0, f"total={total_scale:.1f}x")


# ---------------------------------------------------------------- B. banded solver is used
print("\n--- B. a banded solver is actually used ---")
import inspect
src = inspect.getsource(mod)
check("B.0 minco.py imports/uses solve_banded",
      "solve_banded" in src and "import" in src)
# verify the matrix really has bandwidth 6 (no entries outside |i-j|<=6)
M = 50
rng = np.random.default_rng(1)
tr = MinjerkTraj(rng.standard_normal((M+1,3)), rng.uniform(0.5,1.5,size=M))
A = tr.dense_M()
n = A.shape[0]
ii, jj = np.nonzero(A)
maxband = int(np.max(np.abs(ii - jj))) if ii.size else 0
check("B.1 matrix bandwidth == 6", maxband == 6, f"maxband={maxband}")
# banded storage shape consistent with (l=6,u=6)
check("B.2 banded storage shape (13, 6M)", tr.banded_M.shape == (13, 6*M),
      f"shape={tr.banded_M.shape}")
# banded storage faithfully encodes A
ab = tr.banded_M; u=6; ok=True
for i,j in zip(ii,jj):
    if ab[u+i-j, j] != A[i,j]: ok=False; break
check("B.3 banded storage == dense entries", ok)


# ---------------------------------------------------------------- C. absolute speed at large M
print("\n--- C. absolute speed ---")
tbig = build_time(2000, reps=3)
print(f"    M=2000 build={tbig*1e3:.2f} ms")
check("C.0 M=2000 builds+solves under 200 ms", tbig < 0.2, f"{tbig*1e3:.2f} ms")
# compare against a dense solve of the SAME system to prove banded is the win
Mc = 800
rng = np.random.default_rng(2)
wp = rng.standard_normal((Mc+1,3)); Tc = rng.uniform(0.5,1.5,size=Mc)
trc = MinjerkTraj(wp, Tc)
Adense = trc.dense_M(); bdense = trc._b
t0=time.perf_counter(); np.linalg.solve(Adense, bdense); t_dense=time.perf_counter()-t0
t_band = build_time(Mc, reps=3)
print(f"    M={Mc}: banded build {t_band*1e3:.2f} ms vs dense solve {t_dense*1e3:.2f} ms")
check("C.1 banded build faster than a dense solve of the same matrix",
      t_band < t_dense, f"band={t_band*1e3:.2f}ms dense={t_dense*1e3:.2f}ms")


# ---------------------------------------------------------------- D. accuracy at large M
print("\n--- D. accuracy preserved at large M ---")
M = 1000
rng = np.random.default_rng(3)
wp = rng.standard_normal((M+1,3))*3
T = rng.uniform(0.5,1.5,size=M)
tr = MinjerkTraj(wp, T)
cum = np.concatenate([[0.0], np.cumsum(T)])
idx = rng.integers(1, M, size=40)
werr = max(float(np.max(np.abs(tr.eval_deriv(cum[i],0)-wp[i]))) for i in idx)
res = float(np.max(np.abs(tr.dense_M() @ tr.c - tr._b)))
check("D.0 large-M waypoint interpolation exact", werr < 1e-7, f"werr={werr:.2e}")
check("D.1 large-M residual |Mc-b| small", res < 1e-7, f"res={res:.2e}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    print("Failures:")
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
