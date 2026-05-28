"""Stage 3 — local/bspline.py on REAL global-pipeline paths.

Drives the global stage (voxel_map + HGPPlanner) on 3D scenarios, takes the
post-processed A* polyline, fits a quintic uniform B-spline with both ends
clamped, and checks:

  R.0 endpoints strictly pinned to A* start / goal
  R.1 spline stays close to the A* polyline (chord-to-polyline ≤ tol·seg_max)
       — using true geometric distance, not index-aligned error, since LSQ
         parameterises t uniformly while A* segment arc-lengths vary
  R.2 spline 4-th derivative everywhere finite (quintic = C^4)
  R.3 batched random 3D maps (seed=2026): every successful run obeys R.0–R.2

Run:  python3 test/stage3_bspline_real.py
"""
import importlib.util
import os
import sys
import numpy as np

# global-stage modules via the normal package path
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC       # noqa: E402

# load bspline.py directly (no package __init__ needed)
BS_PATH = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local", "bspline.py")
spec = importlib.util.spec_from_file_location("bspline_standalone", os.path.abspath(BS_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
UniformBSpline = mod.UniformBSpline

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------- helpers ------------------------------------------------------
def blank(dim, res=1.0):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.zeros(3)
    m.cmap = np.full(int(np.prod(dim)), VAL_FREE, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False
    m.initialized = True
    return m


def planner():
    p = HGPPlanner("astar", False, 5, 10, 50, 3000, 0, 0, 20, 0,
                   los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
    p.set_max_expand(200000)
    return p


def run_astar(m, start_w, goal_w):
    p = planner()
    p.set_map_util(m)
    ok, _ = p.plan(start_w, np.zeros(3), goal_w, 0.0, eps=1.0)
    if not ok:
        return None
    return np.array([np.asarray(pt, dtype=float) for pt in p.get_path()])


def chord_dist_to_polyline(point, poly):
    """Min distance from `point` to the polyline (segments poly[i]→poly[i+1])."""
    best = np.inf
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 == 0:
            d = float(np.linalg.norm(point - a))
        else:
            tt = float((point - a) @ ab) / L2
            tt = min(1.0, max(0.0, tt))
            d = float(np.linalg.norm(point - (a + tt * ab)))
        best = min(best, d)
    return best


def fit_and_assert(tag, path, *, num_ctrl=None, degree=5, dt=1.0, sample_K=120, tol_mult=2.0):
    """Fit + clamp endpoints, return per-check flags + details."""
    n_ctrl = max(2 * (degree + 1), int(round(len(path) / 2)))
    n_ctrl = max(n_ctrl, num_ctrl or 0)
    bs = UniformBSpline.fit_path(path, num_ctrl=n_ctrl, degree=degree, dt=dt, clamp_endpoints=True)
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    seg_max = float(seg_lens.max())
    err_s = float(np.max(np.abs(bs.eval(bs.t_start) - path[0])))
    err_e = float(np.max(np.abs(bs.eval(bs.t_end) - path[-1])))
    pin = err_s < 1e-10 and err_e < 1e-10
    ts_dense = np.linspace(bs.t_start, bs.t_end, sample_K)
    a4 = bs.eval_deriv(ts_dense, order=4)
    a4_max = float(np.max(np.abs(a4)))
    a4_finite = np.all(np.isfinite(a4))
    chord_max = max(chord_dist_to_polyline(p, path) for p in bs.eval(ts_dense))
    chord_ok = chord_max < tol_mult * seg_max + 1e-9
    return {
        "pin": pin, "a4_finite": a4_finite, "chord_ok": chord_ok,
        "err_s": err_s, "err_e": err_e, "seg_max": seg_max,
        "a4_max": a4_max, "chord_max": chord_max, "n_ctrl": n_ctrl,
    }


# ---------------------------------------------------------------- R.A. open 3D scene
print("\n--- R.A. open 3D scene: straight diagonal ---")
m = blank([8, 8, 5], res=1.0)
path = run_astar(m, np.array([0.5, 0.5, 0.5]), np.array([6.5, 6.5, 3.5]))
assert path is not None, "open scene A* must succeed"
r = fit_and_assert("open", path)
check("R.A.0 endpoints pinned", r["pin"], f"|s|={r['err_s']:.2e}, |e|={r['err_e']:.2e}")
check("R.A.1 spline stays near A* polyline", r["chord_ok"],
      f"chord_max={r['chord_max']:.2e}, seg_max={r['seg_max']:.2f}")
check("R.A.2 4-th derivative finite", r["a4_finite"], f"|a4|_max={r['a4_max']:.2e}")


# ---------------------------------------------------------------- R.B. 3D scene with a pillar
print("\n--- R.B. 3D scene with a pillar (forces detour) ---")
m = blank([10, 10, 6], res=1.0)
flat = m.cmap.reshape([10, 10, 6], order="F")
flat[5, 5, :] = VAL_OCC                 # vertical pillar in the centre
flat[5, 4, :] = VAL_OCC
flat[4, 5, :] = VAL_OCC
m.cmap = flat.reshape(-1, order="F")
path = run_astar(m, np.array([0.5, 0.5, 0.5]), np.array([8.5, 8.5, 4.5]))
assert path is not None, "pillar scene A* must succeed"
r = fit_and_assert("pillar", path, tol_mult=2.5)  # detour adds a bit of slack
check("R.B.0 endpoints pinned", r["pin"], f"|s|={r['err_s']:.2e}, |e|={r['err_e']:.2e}")
check("R.B.1 spline stays near A* polyline", r["chord_ok"],
      f"chord_max={r['chord_max']:.2e}, seg_max={r['seg_max']:.2f}")
check("R.B.2 4-th derivative finite", r["a4_finite"], f"|a4|_max={r['a4_max']:.2e}")


# ---------------------------------------------------------------- R.C. batched random 3D
print("\n--- R.C. batched random 3D maps (seed=2026) ---")
rng = np.random.default_rng(2026)
trials_run, all_pin, all_a4, all_chord = 0, 0, 0, 0
fails = []
for t in range(40):
    dim = [int(rng.integers(14, 22)), int(rng.integers(14, 22)), int(rng.integers(5, 9))]
    m = blank(dim, res=1.0)
    flat = m.cmap.reshape(dim, order="F")
    occ_mask = rng.random(dim) < 0.08
    flat[occ_mask] = VAL_OCC
    m.cmap = flat.reshape(-1, order="F")
    # force start and goal to opposite octants so the path is long enough to fit
    s = np.array([int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, max(2, dim[2] // 2)))])
    g = np.array([int(rng.integers(dim[0] - 3, dim[0])),
                  int(rng.integers(dim[1] - 3, dim[1])),
                  int(rng.integers(max(2, dim[2] // 2), dim[2]))])
    m.cmap[m.lin_index(*s)] = VAL_FREE
    m.cmap[m.lin_index(*g)] = VAL_FREE
    sw = m.int_to_float(s); gw = m.int_to_float(g)
    path = run_astar(m, sw, gw)
    if path is None or len(path) < 4:
        continue
    trials_run += 1
    r = fit_and_assert(f"rand{t}", path, tol_mult=3.0)  # random scenes can be jagged
    all_pin += int(r["pin"])
    all_a4 += int(r["a4_finite"])
    all_chord += int(r["chord_ok"])
    if not (r["pin"] and r["a4_finite"] and r["chord_ok"]):
        fails.append((t, r))

check("R.C.0 endpoints pinned on every run", all_pin == trials_run,
      f"{all_pin}/{trials_run}")
check("R.C.1 spline near polyline on every run", all_chord == trials_run,
      f"{all_chord}/{trials_run}")
check("R.C.2 4-th derivative finite on every run", all_a4 == trials_run,
      f"{all_a4}/{trials_run}")
print(f"  (random scenes solved + fitted: {trials_run})")
if fails:
    for t, r in fails[:4]:
        print(f"  trial {t}: {r}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    raise SystemExit(1)
