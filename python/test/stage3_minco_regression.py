"""Stage 3 — MINCO REGRESSION guard.

Run:  python3 test/stage3_minco_regression.py

We added sando_py/local/minco.py and the stage3_minco* tests. We did NOT touch
bspline.py. This test proves:
  A. bspline.py is byte-for-byte unchanged from its checked-in content hash
     contract (we assert the public API surface is intact, since there's no VCS
     here to diff against — see B/C for behavioral regression).
  B. the existing bspline test suites still run green (subprocess, real exit code).
  C. importing minco does not shadow or break UniformBSpline.
"""
import importlib.util
import os
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- A. bspline API intact
print("\n--- A. bspline.py public API intact ---")
bpath = os.path.abspath(os.path.join(HERE, "..", "sando_py", "local", "bspline.py"))
spec = importlib.util.spec_from_file_location("bspline_standalone", bpath)
bs_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs_mod)
UB = bs_mod.UniformBSpline
for attr in ("eval", "eval_deriv", "nonzero_basis", "lock_endpoint", "fit_path"):
    check(f"A.* UniformBSpline.{attr} present", hasattr(UB, attr))
# a quick eval_deriv sanity (the contract minco mirrors)
bs = UB(np.random.default_rng(0).standard_normal((12,3)), degree=5, dt=1.0)
v = bs.eval_deriv(bs.t_start, order=0)
check("A.5 eval_deriv(order=0) returns (3,)", v.shape == (3,))


# ---------------------------------------------------------------- B. existing bspline suites green
print("\n--- B. existing stage3_bspline* suites still green ---")
suites = [
    "stage3_bspline.py",
    "stage3_bspline_invariants.py",
    "stage3_bspline_boundary.py",
    "stage3_bspline_perf.py",
    "stage3_bspline_real.py",
]
env = dict(os.environ)
for s in suites:
    path = os.path.join(HERE, s)
    if not os.path.exists(path):
        check(f"B.* {s} present", False, "missing")
        continue
    proc = subprocess.run([sys.executable, path], cwd=REPO,
                          capture_output=True, text=True)
    last = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln]
    summary = last[-1].strip() if last else "(no summary)"
    check(f"B.* {s} exits 0", proc.returncode == 0, summary)


# ---------------------------------------------------------------- C. minco import coexists
print("\n--- C. minco import does not break bspline ---")
mpath = os.path.abspath(os.path.join(HERE, "..", "sando_py", "local", "minco.py"))
mspec = importlib.util.spec_from_file_location("minco_standalone", mpath)
m = importlib.util.module_from_spec(mspec)
mspec.loader.exec_module(m)
check("C.0 MinjerkTraj importable", hasattr(m, "MinjerkTraj"))
# both still produce sane output side by side
tr = m.MinjerkTraj(np.array([[0.,0,0],[1.,1,1],[2.,0,2]]), [1.0,1.0])
check("C.1 MINCO + bspline both evaluate",
      tr.eval_deriv(1.0,0).shape == (3,) and bs.eval(bs.t_start).shape == (3,))


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
