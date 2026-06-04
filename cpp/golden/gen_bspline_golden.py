"""Golden generator for bspline.py (UniformBSpline) C++ port."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.local.bspline import UniformBSpline

rng = np.random.default_rng(303)
def fmt(a): return " ".join(repr(float(x)) for x in np.asarray(a, float).reshape(-1))

lines = []
# --- eval / eval_deriv / nonzero_basis cases ---
for p in [1, 2, 3, 5]:
    Nplus1 = p + 1 + rng.integers(1, 5)
    ctrl = rng.uniform(-4, 4, (Nplus1, 3))
    dt = float(rng.uniform(0.3, 1.5))
    bs = UniformBSpline(ctrl, degree=p, dt=dt)
    K = 30
    ts = np.linspace(bs.t_start, bs.t_end, K)
    e0 = bs.eval(ts); e1 = bs.eval_deriv(ts, 1); e2 = bs.eval_deriv(ts, 2)
    # nonzero_basis at a few interior times
    nb_t = np.linspace(bs.t_start + 1e-6, bs.t_end - 1e-6, 6)
    spans = []; basis = []
    for t in nb_t:
        k, vals = bs.nonzero_basis(float(t)); spans.append(k); basis.append(vals)
    lines += ["SPLINE", f"P {p}", f"DT {repr(dt)}", f"NCTRL {Nplus1}", f"CTRL {fmt(ctrl)}",
              f"TSTART {repr(float(bs.t_start))}", f"TEND {repr(float(bs.t_end))}",
              f"K {K}", f"TS {fmt(ts)}", f"E0 {fmt(e0)}", f"E1 {fmt(e1)}", f"E2 {fmt(e2)}",
              f"NBN 6", f"NBT {fmt(nb_t)}", f"NBSPAN {fmt(spans)}", f"NBVAL {fmt(np.array(basis))}", "END"]

# --- fit_path cases ---
for clamp in [False, True]:
    p = 5; num_ctrl = 2 * (p + 1) + (2 if clamp else 0)
    path = rng.uniform(-3, 3, (20, 3))
    bs = UniformBSpline.fit_path(path, num_ctrl=num_ctrl, degree=p, dt=0.8, clamp_endpoints=clamp)
    K = 25; ts = np.linspace(bs.t_start, bs.t_end, K); ev = bs.eval(ts)
    lines += ["FIT", f"P {p}", f"DT 0.8", f"NUMCTRL {num_ctrl}", f"CLAMP {1 if clamp else 0}",
              f"M {path.shape[0]}", f"PATH {fmt(path)}",
              f"CTRLOUT {fmt(bs.ctrl)}", f"K {K}", f"TS {fmt(ts)}", f"EV {fmt(ev)}", "END"]

out = os.path.join(os.path.dirname(__file__), "bspline_cases.txt")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}")
