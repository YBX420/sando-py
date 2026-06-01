"""Stage 3 — local/bspline.py verification.

Standalone — loads bspline.py directly, no ROS / package __init__.
Uses scipy.interpolate.BSpline as an oracle for eval correctness (not for LSQ).

Run:  python3 test/stage3_bspline.py

Coverage:
  A. De Boor eval vs scipy oracle (random ctrl + random t, 5 configs)
  B. Domain boundaries t_start / t_end don't crash; vectorised eval = pointwise
  C. nonzero_basis: partition of unity, reconstructs eval(t)
  D. Derivatives match central finite differences (orders 1, 2, 3)
  E. lock_endpoint pins position + zeroes derivatives at the boundary
  F. LSQ fits — straight line, circular arc, random 3D polyline
  G. LSQ + clamp_endpoints — endpoints strictly pinned, fit still reasonable
  H. Param sweep — degrees 3 and 5, several num_ctrl, scipy oracle matches
"""
import importlib.util
import os
import numpy as np

# scipy is an oracle here, not part of the implementation
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


def scipy_bspline(bs):
    return si.BSpline(bs.knots, bs.ctrl, bs.p)


# ---------------------------------------------------------------- A. eval vs scipy
print("\n--- A. De Boor vs scipy oracle ---")
rng = np.random.default_rng(0)
for cfg_i, (n_ctrl, p, dt) in enumerate(
    [(10, 5, 0.5), (8, 5, 1.0), (12, 3, 0.3), (20, 5, 0.7), (7, 4, 1.0)]
):
    ctrl = rng.standard_normal((n_ctrl, 3)) * 3.0
    bs = UniformBSpline(ctrl, degree=p, dt=dt)
    oracle = scipy_bspline(bs)
    ts = np.linspace(bs.t_start, bs.t_end, 25)
    mine = np.array([bs.eval(float(t)) for t in ts])
    them = oracle(ts)
    err = float(np.max(np.abs(mine - them)))
    check(f"A.{cfg_i} eval matches scipy (n={n_ctrl}, p={p}, dt={dt})", err < 1e-9, f"max|err|={err:.2e}")


# ---------------------------------------------------------------- B. boundaries + vectorisation
print("\n--- B. boundaries / vectorisation ---")
ctrl = rng.standard_normal((12, 3))
bs = UniformBSpline(ctrl, degree=5, dt=1.0)
ok_lo = np.all(np.isfinite(bs.eval(bs.t_start)))
ok_hi = np.all(np.isfinite(bs.eval(bs.t_end)))
check("B.0 eval(t_start) finite", ok_lo)
check("B.1 eval(t_end) finite", ok_hi)
ts = np.linspace(bs.t_start, bs.t_end, 50)
vec = bs.eval(ts)
pt = np.array([bs.eval(float(t)) for t in ts])
check("B.2 batched eval == pointwise", np.allclose(vec, pt), f"max|err|={np.max(np.abs(vec-pt)):.2e}")
# clamping behaviour — out-of-domain t silently clipped (eval at boundary)
left = bs.eval(bs.t_start - 1.5)
right = bs.eval(bs.t_end + 1.5)
check("B.3 t < t_start clips to t_start", np.allclose(left, bs.eval(bs.t_start)))
check("B.4 t > t_end clips to t_end", np.allclose(right, bs.eval(bs.t_end)))


# ---------------------------------------------------------------- C. nonzero basis
print("\n--- C. nonzero_basis ---")
sum_err = 0.0
recon_err = 0.0
for t in np.linspace(bs.t_start, bs.t_end, 30):
    k, N = bs.nonzero_basis(float(t))
    sum_err = max(sum_err, abs(np.sum(N) - 1.0))
    p = bs.p
    recon = np.zeros(3)
    for j, v in enumerate(N):
        recon = recon + v * bs.ctrl[k - p + j]
    recon_err = max(recon_err, float(np.max(np.abs(recon - bs.eval(float(t))))))
check("C.0 partition of unity (sum N_i(t)=1)", sum_err < 1e-10, f"max|sum-1|={sum_err:.2e}")
check("C.1 nonzero basis reconstructs eval(t)", recon_err < 1e-10, f"max|err|={recon_err:.2e}")


# ---------------------------------------------------------------- D. derivatives vs finite difference
print("\n--- D. derivatives vs central diff ---")
ctrl = rng.standard_normal((14, 3)) * 2.0
bs = UniformBSpline(ctrl, degree=5, dt=0.8)
# evaluate well inside the domain so central diff isn't clipped at boundaries
ts = np.linspace(bs.t_start + 0.5, bs.t_end - 0.5, 11)
for order in (1, 2, 3):
    h = 5e-4
    fd = np.zeros((len(ts), 3))
    for i, t in enumerate(ts):
        # central difference for order-th derivative
        if order == 1:
            fd[i] = (bs.eval(float(t + h)) - bs.eval(float(t - h))) / (2 * h)
        elif order == 2:
            fd[i] = (bs.eval(float(t + h)) - 2 * bs.eval(float(t)) + bs.eval(float(t - h))) / (h * h)
        elif order == 3:
            fd[i] = (
                bs.eval(float(t + 2 * h)) - 2 * bs.eval(float(t + h))
                + 2 * bs.eval(float(t - h)) - bs.eval(float(t - 2 * h))
            ) / (2 * h ** 3)
    analytic = bs.eval_deriv(ts, order=order)
    err = float(np.max(np.abs(fd - analytic)))
    # tolerance loosens with order (h^2 truncation amplified by 1/h^k)
    tol = {1: 1e-5, 2: 1e-3, 3: 5e-1}[order]
    check(f"D.{order} eval_deriv(order={order}) ≈ central diff", err < tol, f"max|err|={err:.2e}")
# higher than p is identically zero
zhi = bs.eval_deriv(ts, order=bs.p + 1)
check("D.4 deriv(order>p) is identically zero", np.allclose(zhi, 0.0))


# ---------------------------------------------------------------- E. lock_endpoint
print("\n--- E. lock_endpoint ---")
ctrl = rng.standard_normal((15, 3))
bs = UniformBSpline(ctrl, degree=5, dt=1.0)
X = np.array([7.0, -2.0, 3.5])
bs.lock_endpoint("start", X)
err_pos = float(np.max(np.abs(bs.eval(bs.t_start) - X)))
v0 = bs.eval_deriv(bs.t_start, order=1)
a0 = bs.eval_deriv(bs.t_start, order=2)
check("E.0 start position pinned", err_pos < 1e-12, f"|err|={err_pos:.2e}")
check("E.1 start velocity = 0", float(np.max(np.abs(v0))) < 1e-10, f"|v|={float(np.max(np.abs(v0))):.2e}")
check("E.2 start accel = 0", float(np.max(np.abs(a0))) < 1e-10, f"|a|={float(np.max(np.abs(a0))):.2e}")
Y = np.array([-1.0, 0.0, 9.0])
bs.lock_endpoint("end", Y)
err_end = float(np.max(np.abs(bs.eval(bs.t_end) - Y)))
vT = bs.eval_deriv(bs.t_end, order=1)
aT = bs.eval_deriv(bs.t_end, order=2)
check("E.3 end position pinned", err_end < 1e-12)
check("E.4 end velocity = 0", float(np.max(np.abs(vT))) < 1e-10)
check("E.5 end accel = 0", float(np.max(np.abs(aT))) < 1e-10)
# re-locking another point at the same side must take precedence
Z = np.array([1.0, 1.0, 1.0])
bs.lock_endpoint("start", Z)
check("E.6 re-lock overrides previous lock", np.allclose(bs.eval(bs.t_start), Z))


# ---------------------------------------------------------------- F. LSQ fit
print("\n--- F. LSQ fit ---")
# F.0 straight line
M = 80
path_line = np.column_stack([
    np.linspace(0, 10, M),
    np.linspace(0, 5, M),
    np.linspace(0, -2, M),
])
fit_line = UniformBSpline.fit_path(path_line, num_ctrl=12, degree=5, dt=1.0)
ts = np.linspace(fit_line.t_start, fit_line.t_end, M)
err = float(np.max(np.linalg.norm(fit_line.eval(ts) - path_line, axis=1)))
check("F.0 straight line: max||err|| < 1e-9", err < 1e-9, f"max||err||={err:.2e}")

# F.1 circular arc (xy plane), 3/4 turn
M = 200
theta = np.linspace(0, 1.5 * np.pi, M)
path_arc = np.column_stack([np.cos(theta), np.sin(theta), 0.1 * theta])
fit_arc = UniformBSpline.fit_path(path_arc, num_ctrl=24, degree=5, dt=1.0)
ts = np.linspace(fit_arc.t_start, fit_arc.t_end, M)
err = float(np.max(np.linalg.norm(fit_arc.eval(ts) - path_arc, axis=1)))
# 24 ctrl pts for 3/4 turn → ~0.20 rad / pt, easily sub-mm in unit circle
check("F.1 arc (3/4 turn, 24 ctrl): max||err|| < 5e-3", err < 5e-3, f"max||err||={err:.2e}")

# F.2 random polyline (mimics A* output)
rng2 = np.random.default_rng(123)
M = 60
path_rand = np.cumsum(rng2.standard_normal((M, 3)) * 0.5, axis=0)
fit_rand = UniformBSpline.fit_path(path_rand, num_ctrl=30, degree=5, dt=1.0)
ts = np.linspace(fit_rand.t_start, fit_rand.t_end, M)
err = float(np.max(np.linalg.norm(fit_rand.eval(ts) - path_rand, axis=1)))
# 30 ctrl pts for 60 samples → spline can't follow every wiggle; want < typical step
step = float(np.max(np.linalg.norm(np.diff(path_rand, axis=0), axis=1)))
check("F.2 random polyline: max err < typical step length", err < step, f"err={err:.2e}, step={step:.2e}")


# ---------------------------------------------------------------- G. LSQ + clamp endpoints
print("\n--- G. LSQ + clamp endpoints ---")
# Straight line, fully clamped — endpoints should be EXACT, interior nearly so
fit_c = UniformBSpline.fit_path(path_line, num_ctrl=12, degree=5, dt=1.0, clamp_endpoints=True)
err_start = float(np.max(np.abs(fit_c.eval(fit_c.t_start) - path_line[0])))
err_end = float(np.max(np.abs(fit_c.eval(fit_c.t_end) - path_line[-1])))
ts = np.linspace(fit_c.t_start, fit_c.t_end, len(path_line))
err_mid = float(np.max(np.linalg.norm(fit_c.eval(ts) - path_line, axis=1)))
check("G.0 clamped start exact", err_start < 1e-12, f"|err|={err_start:.2e}")
check("G.1 clamped end exact", err_end < 1e-12, f"|err|={err_end:.2e}")
# straight line stays exact even under clamp (boundary zeros are consistent with linearity → no, vel(start)=const ≠ 0)
# So mid error is *not* zero; just check it's bounded by the line length scale.
line_len = float(np.linalg.norm(path_line[-1] - path_line[0]))
check("G.2 clamped fit error stays bounded", err_mid < 0.5 * line_len, f"mid_err={err_mid:.2e}, line_len={line_len:.2f}")
# curved path under clamp — endpoints still strictly pinned, mid error reasonable
fit_c2 = UniformBSpline.fit_path(path_arc, num_ctrl=24, degree=5, dt=1.0, clamp_endpoints=True)
ts = np.linspace(fit_c2.t_start, fit_c2.t_end, len(path_arc))
mid = float(np.max(np.linalg.norm(fit_c2.eval(ts) - path_arc, axis=1)))
arc_scale = float(np.max(np.linalg.norm(path_arc, axis=1)))
check("G.3 arc clamped: start pinned", np.allclose(fit_c2.eval(fit_c2.t_start), path_arc[0]))
check("G.4 arc clamped: end pinned", np.allclose(fit_c2.eval(fit_c2.t_end), path_arc[-1]))
check("G.5 arc clamped: mid error < scale", mid < arc_scale, f"mid={mid:.2e}, scale={arc_scale:.2f}")


# ---------------------------------------------------------------- H. parameter sweep — scipy oracle
print("\n--- H. param sweep vs scipy ---")
sweep_ok = True
worst = 0.0
rng3 = np.random.default_rng(7)
for trial in range(30):
    p = int(rng3.choice([3, 4, 5]))
    n_ctrl = int(rng3.integers(p + 2, p + 20))
    dt = float(rng3.uniform(0.1, 1.5))
    ctrl = rng3.standard_normal((n_ctrl, 3)) * 3
    bs = UniformBSpline(ctrl, degree=p, dt=dt)
    oracle = scipy_bspline(bs)
    ts = np.linspace(bs.t_start, bs.t_end, 17)
    mine = bs.eval(ts)
    them = oracle(ts)
    err = float(np.max(np.abs(mine - them)))
    if err > 1e-9:
        sweep_ok = False
    worst = max(worst, err)
check("H.0 30 random configs all match scipy", sweep_ok, f"worst|err|={worst:.2e}")


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
