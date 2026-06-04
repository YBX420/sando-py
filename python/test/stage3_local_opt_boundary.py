"""Stage 3 — local/local_opt.py extreme boundaries / degenerate inputs.

  B.0  empty obstacle list converges fast + low final cost
  B.1  straight path + no obstacles → optimised spline stays straight
  B.2  M = 2 path (just start + goal) doesn't crash
  B.3  maxiter = 2 (too few iterations) → info.converged = False, spline still returned
  B.4  vmax very low → T_target large, dt grows; no inf/nan
  B.5  vmax very high → vel penalty stays at 0
  B.6  num_ctrl too small (< 2*(p+1) + 1) → auto-promoted, no crash
  B.7  explicit num_ctrl honoured
  B.8  short path triggers wrong-ndim / too-short validation
  B.9  plan returns the spline as UniformBSpline + dt_final > dt_min
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams, default_config        # noqa: E402
from sando_py.local.obstacles import SphereObstacle                        # noqa: E402
from sando_py.local.cost import obstacle_cost                              # noqa: E402
from sando_py.local.local_opt import plan, OptParams                       # noqa: E402
from sando_py.local.bspline import UniformBSpline                          # noqa: E402

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


cfg = default_config()
LEAN = OptParams(maxiter=30, K=80, vmax=3.0, amax=3.0)


def straight(M=20, length=10.0):
    return np.column_stack([np.linspace(0, length, M), np.zeros(M), np.zeros(M)])


# ---------------------------------------------------------------- B.0 empty obstacles
print("\n--- B.0 empty obstacle list ---")
bs, info = plan(straight(), [], cfg, opt_params=LEAN)
check("B.0a empty obs: converges", info["converged"])
check("B.0b empty obs: final_cost is small", info["final_cost"] < 1e2,
      f"final_cost={info['final_cost']:.3e}")


# ---------------------------------------------------------------- B.1 straight path stays straight
print("\n--- B.1 straight path stays roughly straight ---")
bs, info = plan(straight(), [], cfg, opt_params=LEAN)
ts = np.linspace(bs.t_start, bs.t_end, 50)
pts = bs.eval(ts)
# spline y, z should stay near 0 (path was along x axis)
yz_drift = float(np.max(np.linalg.norm(pts[:, 1:], axis=1)))
check("B.1 straight-path drift < 0.5 m (no obstacle pushing)", yz_drift < 0.5,
      f"yz_drift={yz_drift:.3e}")


# ---------------------------------------------------------------- B.2 M=2 minimal path
print("\n--- B.2 M = 2 minimal path ---")
path2 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
bs, info = plan(path2, [], cfg, opt_params=LEAN)
ok = (np.all(np.isfinite(bs.ctrl))
      and np.allclose(bs.eval(bs.t_start), path2[0])
      and np.allclose(bs.eval(bs.t_end), path2[-1]))
check("B.2 M=2 path: finite spline + endpoints exact", ok)


# ---------------------------------------------------------------- B.3 maxiter = 2
print("\n--- B.3 too-small maxiter → converged=False but spline still returned ---")
bs, info = plan(straight(), [SphereObstacle([5, 0, 0], 0.4, class_name="human")],
                cfg, opt_params=OptParams(maxiter=2, K=80, vmax=3.0))
check("B.3a converged=False with maxiter=2", not info["converged"])
check("B.3b spline still finite + endpoints exact",
      np.all(np.isfinite(bs.ctrl))
      and np.allclose(bs.eval(bs.t_start), [0, 0, 0])
      and np.allclose(bs.eval(bs.t_end), [10, 0, 0]))


# ---------------------------------------------------------------- B.4 vmax very low
print("\n--- B.4 vmax very low ---")
slow_opt = OptParams(maxiter=20, K=80, vmax=0.1, amax=3.0)
bs, info = plan(straight(), [], cfg, opt_params=slow_opt)
check("B.4a vmax=0.1 yields large T_target", info["T_target"] > 50)
check("B.4b spline + info still finite", np.all(np.isfinite(bs.ctrl)) and np.isfinite(info["final_cost"]))


# ---------------------------------------------------------------- B.5 vmax very high
print("\n--- B.5 vmax very high → vel penalty inactive ---")
fast_opt = OptParams(maxiter=20, K=80, vmax=1.0e6, amax=1.0e6)
bs, info = plan(straight(), [], cfg, opt_params=fast_opt)
ts = np.linspace(bs.t_start, bs.t_end, 100)
v_mag = np.linalg.norm(bs.eval_deriv(ts, 1), axis=1)
check("B.5 max|v| < vmax (penalty stays at 0)", float(np.max(v_mag)) < fast_opt.vmax)


# ---------------------------------------------------------------- B.6 num_ctrl too small auto-promoted
print("\n--- B.6 num_ctrl below 2*(p+1)+1 gets auto-promoted ---")
bs, info = plan(straight(), [], cfg, num_ctrl=5, opt_params=LEAN)
check("B.6 num_ctrl auto-promoted to 2*(p+1)+1=13", info["num_ctrl"] == 13,
      f"got num_ctrl={info['num_ctrl']}")


# ---------------------------------------------------------------- B.7 explicit num_ctrl honoured
print("\n--- B.7 explicit num_ctrl honoured ---")
bs, info = plan(straight(), [], cfg, num_ctrl=20, opt_params=LEAN)
check("B.7 num_ctrl=20 honoured", info["num_ctrl"] == 20)


# ---------------------------------------------------------------- B.8 input validation
print("\n--- B.8 input validation ---")
check("B.8a M < 2 path raises",
      raises(lambda: plan(np.zeros((1, 3)), [], cfg, opt_params=LEAN)))
check("B.8b 1-D ndarray path raises",
      raises(lambda: plan(np.zeros(10), [], cfg, opt_params=LEAN)))


# ---------------------------------------------------------------- B.9 return type + dt
print("\n--- B.9 return type + dt > dt_min ---")
bs, info = plan(straight(), [], cfg, opt_params=LEAN)
check("B.9a returns UniformBSpline", isinstance(bs, UniformBSpline))
check("B.9b dt_final > dt_min", info["dt_final"] > LEAN.dt_min)


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
