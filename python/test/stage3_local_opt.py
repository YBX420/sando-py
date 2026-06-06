"""Stage 3 — local/local_opt.py functional baseline.

Coverage:
  A. straight path, no obstacles → cost stays tiny, endpoints exact
  B. straight path with a sphere on it → obstacle cost drops a lot vs init
  C. endpoints stay exactly = path[0] / path[-1] after optimisation
  D. cost is non-increasing: final_cost ≤ init_cost
  E. dt stays above dt_min and inside a sane range (no blow-up, no collapse)
  F. multiple obstacles: all of them get dodged (final obs_cost ≪ init obs_cost)
  G. vel / accel stay near vmax / amax (penalty is doing its job)
  H. plan() input validation (path too short, etc.)
  I. info dict shape (the upper-layer contract)

Run:  python3 test/stage3_local_opt.py
"""
import importlib.util
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams, default_config        # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.cost import obstacle_cost                              # noqa: E402
from sando_py.local.local_opt import plan, OptParams                       # noqa: E402

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


def straight_path(M=30, length=10.0):
    return np.column_stack([np.linspace(0, length, M), np.zeros(M), np.zeros(M)])


cfg = default_config()
# Tests use a leaner opt (smaller K + maxiter) to keep the suite fast;
# stage3_local_opt_perf.py covers full-budget timing.
opt = OptParams(maxiter=40, K=100, vmax=3.0, amax=3.0)


# ---------------------------------------------------------------- A. no obstacles
print("\n--- A. straight path, no obstacles ---")
bs, info = plan(straight_path(), [], cfg, opt_params=opt)
check("A.0 converged on a trivial scene", info["converged"], info["message"])
check("A.1 obstacle_cost = 0 after plan (no obstacles)",
      obstacle_cost(bs, [], cfg, K=300) == 0.0)
check("A.2 final_cost ≤ init_cost", info["final_cost"] <= info["init_cost"] + 1e-9,
      f"init={info['init_cost']:.3e}, final={info['final_cost']:.3e}")
check("A.3 endpoints exact (no-obstacle case)",
      np.allclose(bs.eval(bs.t_start), [0, 0, 0]) and np.allclose(bs.eval(bs.t_end), [10, 0, 0]))


# ---------------------------------------------------------------- B. sphere on the path
print("\n--- B. straight path with sphere on it ---")
path = straight_path()
sphere_path = SphereObstacle([5.0, 0.0, 0.0], 0.6, class_name="human")
init_obs = obstacle_cost(
    __import__("sando_py.local.bspline", fromlist=["UniformBSpline"]).UniformBSpline.fit_path(
        path, num_ctrl=15, degree=5, dt=1.0, clamp_endpoints=True),
    [sphere_path], cfg, K=300)
bs, info = plan(path, [sphere_path], cfg, opt_params=opt)
final_obs = obstacle_cost(bs, [sphere_path], cfg, K=300)
# threshold for the lean opt (maxiter=40, K=100): drops by at least 3× —
# stage3_local_opt_perf.py verifies tighter convergence with full budget.
check("B.0 final obstacle cost << initial (sphere gets dodged)",
      final_obs < 0.3 * init_obs, f"init_obs={init_obs:.3e}, final_obs={final_obs:.3e}")
check("B.1 endpoints exactly preserved",
      np.allclose(bs.eval(bs.t_start), path[0]) and np.allclose(bs.eval(bs.t_end), path[-1]))


# ---------------------------------------------------------------- C. endpoint preservation (multiple runs)
print("\n--- C. endpoint preservation under random paths ---")
rng = np.random.default_rng(7)
bad = 0
worst = 0.0
n_trials = 5
for _ in range(n_trials):
    M = int(rng.integers(20, 40))
    p = np.cumsum(rng.standard_normal((M, 3)) * 0.5, axis=0)
    obs = [SphereObstacle(rng.uniform(-3, 3, 3) + p.mean(axis=0), 0.3, class_name="human")]
    bs, info = plan(p, obs, cfg, opt_params=OptParams(maxiter=30, K=100, vmax=3.0))
    es = float(np.max(np.abs(bs.eval(bs.t_start) - p[0])))
    ee = float(np.max(np.abs(bs.eval(bs.t_end) - p[-1])))
    if es > 1e-9 or ee > 1e-9:
        bad += 1
    worst = max(worst, es, ee)
check(f"C.0 endpoints exactly preserved across {n_trials} random runs", bad == 0,
      f"worst|err|={worst:.2e}, bad={bad}/{n_trials}")


# ---------------------------------------------------------------- D. cost monotonic (init → final)
print("\n--- D. final_cost ≤ init_cost ---")
bs, info = plan(straight_path(), [sphere_path], cfg, opt_params=opt)
check("D.0 final_cost ≤ init_cost",
      info["final_cost"] <= info["init_cost"],
      f"init={info['init_cost']:.3e}, final={info['final_cost']:.3e}")


# ---------------------------------------------------------------- E. dt stays in a sane range
print("\n--- E. dt stays > dt_min and didn't blow up ---")
bs, info = plan(straight_path(), [sphere_path], cfg, opt_params=opt)
dt_min = opt.dt_min
dt_max_sane = 50.0 * info["T_target"] / (info["num_ctrl"] - opt.degree)
check("E.0 dt > dt_min", info["dt_final"] > dt_min, f"dt_final={info['dt_final']:.3e}")
check("E.1 dt is finite (didn't blow up)", info["dt_final"] < dt_max_sane,
      f"dt_final={info['dt_final']:.3e}, sane_max={dt_max_sane:.3e}")


# ---------------------------------------------------------------- F. multi-obstacle dodge
print("\n--- F. multiple obstacles all get dodged ---")
multi = [SphereObstacle([3.0, 0.0, 0.0], 0.4, class_name="human"),
         SphereObstacle([7.0, 0.0, 0.0], 0.4, class_name="human"),
         AABBObstacle([4.9, -0.3, -0.3], [5.1, 0.3, 0.3], class_name="wall")]
from sando_py.local.bspline import UniformBSpline
bs0 = UniformBSpline.fit_path(straight_path(), num_ctrl=15, degree=5, dt=1.0,
                              clamp_endpoints=True)
init_obs = obstacle_cost(bs0, multi, cfg, K=400)
bs, info = plan(straight_path(), multi, cfg, opt_params=opt)
final_obs = obstacle_cost(bs, multi, cfg, K=400)
check("F.0 multi-obstacle: obstacle cost drops significantly",
      final_obs < 0.1 * init_obs, f"init={init_obs:.3e}, final={final_obs:.3e}")


# ---------------------------------------------------------------- G. vel / accel stay near limits
print("\n--- G. vel / accel near limits (penalty working) ---")
bs, info = plan(straight_path(), [sphere_path], cfg, opt_params=opt)
ts = np.linspace(bs.t_start, bs.t_end, 500)
v_max = float(np.max(np.linalg.norm(bs.eval_deriv(ts, 1), axis=1)))
a_max = float(np.max(np.linalg.norm(bs.eval_deriv(ts, 2), axis=1)))
# soft penalty (not a hard constraint) + lean opt budget; perf test asserts tighter limits.
check("G.0 max|v| stays within 2× vmax (lean opt)", v_max < 2.0 * opt.vmax,
      f"max|v|={v_max:.3f}, vmax={opt.vmax}")
check("G.1 max|a| stays within 2× amax (lean opt)", a_max < 2.0 * opt.amax,
      f"max|a|={a_max:.3f}, amax={opt.amax}")


# ---------------------------------------------------------------- H. input validation
print("\n--- H. input validation ---")
check("H.0 path with < 2 points raises",
      raises(lambda: plan(np.zeros((1, 3)), [], cfg)))
check("H.1 path with wrong ndim raises",
      raises(lambda: plan(np.zeros(10), [], cfg)))


# ---------------------------------------------------------------- I. info contract
print("\n--- I. info dict shape ---")
bs, info = plan(straight_path(), [], cfg, opt_params=opt)
required_keys = {"converged", "init_cost", "final_cost", "iter",
                 "dt0", "dt_final", "T_target", "T_final", "num_ctrl", "message"}
missing = required_keys - set(info.keys())
check("I.0 info has all required keys", not missing, f"missing={missing}")
check("I.1 info types are sane",
      isinstance(info["converged"], bool) and
      isinstance(info["iter"], int) and
      all(isinstance(info[k], float) for k in ("init_cost", "final_cost",
                                                "dt0", "dt_final", "T_target", "T_final")))


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
