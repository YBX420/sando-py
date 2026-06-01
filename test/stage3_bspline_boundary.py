"""Stage 3 — local/bspline.py extreme boundaries + degenerate inputs.

Cases:
  B.0  minimum num_ctrl == degree+1 (smallest valid spline) — builds + evals
  B.1  num_ctrl < degree+1 → ValueError
  B.2  degree=1 (linear) → eval at the inner knot equals the ctrl point there
  B.3  high degree p=7 builds + scipy-matches
  B.4  dt very small (1e-6) / very large (1e6) stays numerically stable
  B.5  constant path → spline is the same constant, all derivatives = 0
  B.6  collinear path → spline stays on the same line
  B.7  M=2 path (minimal) doesn't crash
  B.8  M < num_ctrl (under-determined LSQ) → finite ctrl, eval finite
  B.9  lock_endpoint with bad side → ValueError
  B.10 2D ctrl pts (d=2) work, eval has shape (K, 2)
  B.11 path with ndim != 2 → ValueError
  B.12 dt ≤ 0 / degree < 1 / ndim=1 ctrl → ValueError at construction
  B.13 eval at t = ±inf clips silently (no NaN)
  B.14 fit_path with clamp_endpoints requires num_ctrl ≥ 2*(p+1) — regression
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

def raises(fn, exc=ValueError):
    try:
        fn()
        return False
    except exc:
        return True


# ---------------------------------------------------------------- B.0 minimum num_ctrl
print("\n--- B.0 minimum num_ctrl == degree+1 ---")
for p in (3, 5, 7):
    ctrl = np.random.default_rng(p).standard_normal((p + 1, 3))
    bs = UniformBSpline(ctrl, degree=p, dt=1.0)
    val = bs.eval(bs.t_start + 0.5 * bs.T)
    finite = np.all(np.isfinite(val))
    check(f"B.0 p={p}: smallest spline (p+1 ctrl) evaluates finite", finite)


# ---------------------------------------------------------------- B.1 num_ctrl < degree+1
print("\n--- B.1 too-few ctrl pts raises ---")
check("B.1 num_ctrl < degree+1 raises", raises(lambda: UniformBSpline(np.zeros((5, 3)), degree=5, dt=1.0)))


# ---------------------------------------------------------------- B.2 degree=1 piecewise linear
print("\n--- B.2 degree=1 is piecewise linear ---")
# for p=1, internal knots are knots[2..n-1]; basis at knots[k] selects ctrl[k-1]
# (uniform B-spline of degree 1 interpolates its own control points at the knots
# strictly inside the valid domain)
ctrl_lin = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0]], dtype=float)
bs = UniformBSpline(ctrl_lin, degree=1, dt=1.0)
# valid domain: knots[1] .. knots[n+1=5] → t in [0, 4]
# at t = knots[k] (k=1..n), eval == ctrl[k-1]? Actually for uniform deg-1, eval at knots[i] == ctrl[i-1].
# We'll spot-check: eval(t = knots[2]) should equal ctrl[1].
val = bs.eval(float(bs.knots[2]))
check("B.2 degree=1 spline interpolates ctrl at the inner knot", np.allclose(val, ctrl_lin[1]),
      f"got {val}, want {ctrl_lin[1]}")


# ---------------------------------------------------------------- B.3 high degree (p=7)
print("\n--- B.3 high degree p=7 ---")
rng = np.random.default_rng(73)
ctrl = rng.standard_normal((15, 3))
bs = UniformBSpline(ctrl, degree=7, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 25)
mine = bs.eval(ts)
them = si.BSpline(bs.knots, bs.ctrl, bs.p)(ts)
check("B.3 p=7 matches scipy oracle", np.allclose(mine, them, atol=1e-9),
      f"max|err|={float(np.max(np.abs(mine - them))):.2e}")


# ---------------------------------------------------------------- B.4 extreme dt
print("\n--- B.4 extreme dt values ---")
for dt in (1e-6, 1.0, 1e6):
    bs = UniformBSpline(rng.standard_normal((10, 3)), degree=5, dt=dt)
    val = bs.eval(0.5 * (bs.t_start + bs.t_end))
    check(f"B.4 dt={dt:g}: eval finite", np.all(np.isfinite(val)))


# ---------------------------------------------------------------- B.5 constant path
print("\n--- B.5 constant path → constant spline ---")
const = np.array([3.14, -2.7, 1.0])
path = np.tile(const, (80, 1))
bs = UniformBSpline.fit_path(path, num_ctrl=14, degree=5, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 50)
err_pos = float(np.max(np.abs(bs.eval(ts) - const)))
err_v = float(np.max(np.abs(bs.eval_deriv(ts, 1))))
check("B.5a constant path → eval == const everywhere", err_pos < 1e-9, f"max|err|={err_pos:.2e}")
check("B.5b constant path → velocity ≈ 0", err_v < 1e-9, f"max|v|={err_v:.2e}")


# ---------------------------------------------------------------- B.6 collinear path
print("\n--- B.6 collinear path → spline stays on the line ---")
a = np.array([1.0, 2.0, -1.0])
direction = np.array([1.0, 1.0, 0.5])
direction = direction / np.linalg.norm(direction)
M = 80
s = np.linspace(0, 10, M)[:, None]
path = a + s * direction
bs = UniformBSpline.fit_path(path, num_ctrl=14, degree=5, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 50)
vals = bs.eval(ts)
# distance to the line through a with direction `direction`
off = vals - a
proj = (off @ direction)[:, None] * direction
perp = off - proj
max_perp = float(np.max(np.linalg.norm(perp, axis=1)))
check("B.6 collinear: perpendicular distance to line ≈ 0", max_perp < 1e-9, f"max⊥={max_perp:.2e}")


# ---------------------------------------------------------------- B.7 M=2 path
print("\n--- B.7 M=2 (minimal) path doesn't crash ---")
path2 = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, -2.0]])
try:
    bs = UniformBSpline.fit_path(path2, num_ctrl=12, degree=5, dt=1.0)
    val = bs.eval(0.5 * (bs.t_start + bs.t_end))
    ok = np.all(np.isfinite(val))
except Exception as e:
    ok = False
    val = e
check("B.7 M=2 path: builds + evals finite", ok, f"got {val}")


# ---------------------------------------------------------------- B.8 M < num_ctrl
print("\n--- B.8 M < num_ctrl (under-determined LSQ) → finite ctrl ---")
path_small = rng.standard_normal((5, 3))
bs = UniformBSpline.fit_path(path_small, num_ctrl=12, degree=5, dt=1.0)
ts = np.linspace(bs.t_start, bs.t_end, 30)
ok = np.all(np.isfinite(bs.ctrl)) and np.all(np.isfinite(bs.eval(ts)))
check("B.8 under-determined LSQ: ctrl + eval finite", ok)


# ---------------------------------------------------------------- B.9 bad lock_endpoint side
print("\n--- B.9 lock_endpoint side validation ---")
bs = UniformBSpline(rng.standard_normal((10, 3)), degree=5, dt=1.0)
check("B.9 lock_endpoint('middle', ...) raises", raises(lambda: bs.lock_endpoint("middle", np.zeros(3))))


# ---------------------------------------------------------------- B.10 d=2 ctrl pts
print("\n--- B.10 2D control points work ---")
ctrl2 = rng.standard_normal((10, 2))
bs = UniformBSpline(ctrl2, degree=5, dt=1.0)
val = bs.eval(0.5 * (bs.t_start + bs.t_end))
check("B.10a d=2: eval shape (2,)", val.shape == (2,), f"shape={val.shape}")
oracle = si.BSpline(bs.knots, bs.ctrl, bs.p)
ts = np.linspace(bs.t_start, bs.t_end, 20)
check("B.10b d=2: matches scipy oracle", np.allclose(bs.eval(ts), oracle(ts)))


# ---------------------------------------------------------------- B.11 path ndim != 2
print("\n--- B.11 path ndim validation ---")
check("B.11 1-D path raises", raises(lambda: UniformBSpline.fit_path(np.zeros(10), num_ctrl=8, degree=5, dt=1.0)))


# ---------------------------------------------------------------- B.12 construction param validation
print("\n--- B.12 construction param validation ---")
check("B.12a dt ≤ 0 raises", raises(lambda: UniformBSpline(np.zeros((10, 3)), degree=5, dt=0.0)))
check("B.12b dt < 0 raises", raises(lambda: UniformBSpline(np.zeros((10, 3)), degree=5, dt=-1.0)))
check("B.12c degree < 1 raises", raises(lambda: UniformBSpline(np.zeros((10, 3)), degree=0, dt=1.0)))
check("B.12d ndim=1 ctrl raises", raises(lambda: UniformBSpline(np.zeros(10), degree=5, dt=1.0)))


# ---------------------------------------------------------------- B.13 eval ± inf
print("\n--- B.13 eval at ±inf clips ---")
bs = UniformBSpline(rng.standard_normal((10, 3)), degree=5, dt=1.0)
v_neg = bs.eval(-np.inf)
v_pos = bs.eval(np.inf)
ok = np.all(np.isfinite(v_neg)) and np.all(np.isfinite(v_pos))
check("B.13a eval(±inf) finite (clipped)", ok)
check("B.13b eval(-inf) == eval(t_start)", np.allclose(v_neg, bs.eval(bs.t_start)))
check("B.13c eval(+inf) == eval(t_end)", np.allclose(v_pos, bs.eval(bs.t_end)))


# ---------------------------------------------------------------- B.14 clamp_endpoints needs 2*(p+1) ctrl pts
print("\n--- B.14 clamp_endpoints regression: needs num_ctrl ≥ 2*(p+1) ---")
path = rng.standard_normal((60, 3))
# p=5 needs ≥ 12 ctrl pts to clamp both ends without overlap
check(
    "B.14a num_ctrl=11 (= 2*(p+1)-1) raises with clamp_endpoints=True",
    raises(lambda: UniformBSpline.fit_path(path, num_ctrl=11, degree=5, dt=1.0, clamp_endpoints=True)),
)
# also should NOT raise without clamp
try:
    UniformBSpline.fit_path(path, num_ctrl=11, degree=5, dt=1.0, clamp_endpoints=False)
    ok = True
except Exception:
    ok = False
check("B.14b num_ctrl=11 without clamp_endpoints: OK", ok)
# and num_ctrl == 12 should be OK with clamp
try:
    bs = UniformBSpline.fit_path(path, num_ctrl=12, degree=5, dt=1.0, clamp_endpoints=True)
    ok = (np.allclose(bs.eval(bs.t_start), path[0]) and np.allclose(bs.eval(bs.t_end), path[-1]))
except Exception:
    ok = False
check("B.14c num_ctrl=12 with clamp: pins both ends", ok)


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
