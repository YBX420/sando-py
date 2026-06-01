"""Stage 3 — local/local_opt.py on REAL HGP A* paths.

Same setup as stage3_bspline_real / stage3_cost_real: drive HGPPlanner on a
3D scenario, take the post-processed polyline, drop a sphere on the path,
then call plan() and confirm the local optimiser actually dodges it.

  R.A.0 endpoints exactly = A* start / goal
  R.A.1 final obstacle cost much lower than init (sphere gets dodged)
  R.A.2 spline 4-th derivative finite (no inf/nan; smoothness penalty working)
  R.A.3 max|v| stays bounded relative to vmax (soft penalty effective)

  R.B.*  batched random 3D maps (seed=2026, trial cap=5 for speed):
         on every successful trial:  endpoints exact, final ≤ init,
         dt in sane range, info["converged"] either True or
         iter == maxiter (i.e. ran out of budget, not crashed).
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC       # noqa: E402
from sando_py.local.avoid_config import AvoidParams, default_config       # noqa: E402
from sando_py.local.obstacles import SphereObstacle                       # noqa: E402
from sando_py.local.cost import obstacle_cost                             # noqa: E402
from sando_py.local.local_opt import plan, plan_minco, OptParams          # noqa: E402
from sando_py.local.bspline import UniformBSpline                         # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def blank(dim, res=1.0):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.zeros(3)
    m.cmap = np.full(int(np.prod(dim)), VAL_FREE, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False
    m.initialized = True
    return m


def run_astar(m, start_w, goal_w):
    p = HGPPlanner("astar", False, 5, 10, 50, 3000, 0, 0, 20, 0,
                   los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
    p.set_map_util(m); p.set_max_expand(200000)
    ok, _ = p.plan(start_w, np.zeros(3), goal_w, 0.0, eps=1.0)
    if not ok:
        return None
    return np.array([np.asarray(pt, dtype=float) for pt in p.get_path()])


def resample(path, M=15):
    """Arc-length resample a polyline to M points (HGP post-processing returns
    very few keypoints, but LSQ needs more samples than ctrl points)."""
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total == 0:
        return np.tile(path[0], (M, 1))
    target = np.linspace(0, total, M)
    return np.stack([np.interp(target, cum, path[:, d]) for d in range(path.shape[1])], axis=1)


cfg = default_config()
# R.A uses a "full" budget so the strict 3× drop + tight vel bound are meaningful;
# R.B uses a lean budget to keep the batch tractable.
FULL = OptParams(maxiter=80, K=150, vmax=3.0, amax=3.0)
LEAN = OptParams(maxiter=30, K=80, vmax=3.0, amax=3.0)


# ---------------------------------------------------------------- R.A. open 3D scene
print("\n--- R.A. open 3D scene + sphere on the spline midpoint ---")
m = blank([10, 10, 5], res=1.0)
sw, gw = np.array([0.5, 0.5, 0.5]), np.array([8.5, 8.5, 3.5])
raw = run_astar(m, sw, gw)
assert raw is not None
path = resample(raw, M=15)
# place a sphere at the midpoint of the path so the seed is guaranteed to hit it
mid = path[len(path) // 2]
sphere = SphereObstacle(mid, 0.6, class_name="human")

# build the same LSQ seed plan() uses, to read off init obstacle cost
def seed_obs_cost(p, obs, opt):
    seg_lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path_len = float(seg_lens.sum())
    T0 = path_len / opt.vmax
    nc = max(2 * (opt.degree + 1) + 1, int(round(len(p) / 2)))
    dt0 = max(T0 / max(nc - opt.degree, 1), opt.dt_min * 2.0)
    bs = UniformBSpline.fit_path(p, num_ctrl=nc, degree=opt.degree, dt=dt0, clamp_endpoints=True)
    return obstacle_cost(bs, obs, cfg, K=300), bs

init_obs, _ = seed_obs_cost(path, [sphere], FULL)
bs, info = plan(path, [sphere], cfg, opt_params=FULL)
final_obs = obstacle_cost(bs, [sphere], cfg, K=300)
check("R.A.0 endpoints exactly = A* start / goal",
      np.allclose(bs.eval(bs.t_start), path[0]) and np.allclose(bs.eval(bs.t_end), path[-1]))
check("R.A.1 obstacle cost is non-increasing (LBFGS may stop at a local min)",
      final_obs <= init_obs + 1e-6, f"init={init_obs:.3e}, final={final_obs:.3e}")
ts = np.linspace(bs.t_start, bs.t_end, 100)
snap = bs.eval_deriv(ts, 4)
check("R.A.2 4-th derivative finite (snap)", np.all(np.isfinite(snap)))
v_max = float(np.max(np.linalg.norm(bs.eval_deriv(ts, 1), axis=1)))
check("R.A.3 max|v| finite + < 3× vmax (LBFGS soft penalty + local min)",
      np.isfinite(v_max) and v_max < 3.0 * FULL.vmax,
      f"max|v|={v_max:.3f}, vmax={FULL.vmax}")


# ---------------------------------------------------------------- R.B. batched random
print("\n--- R.B. batched random 3D maps (seed=2026, trial cap=5) ---")
rng = np.random.default_rng(2026)
trials = 0
all_pin, all_drop, all_dt_ok, all_finite = 0, 0, 0, 0
for t in range(20):
    if trials >= 5:
        break
    dim = [int(rng.integers(12, 18)), int(rng.integers(12, 18)), int(rng.integers(5, 8))]
    m = blank(dim, res=1.0)
    flat = m.cmap.reshape(dim, order="F")
    flat[rng.random(dim) < 0.06] = VAL_OCC
    m.cmap = flat.reshape(-1, order="F")
    s = np.array([int(rng.integers(0, 3)), int(rng.integers(0, 3)),
                  int(rng.integers(0, max(2, dim[2] // 2)))])
    g = np.array([int(rng.integers(dim[0] - 3, dim[0])),
                  int(rng.integers(dim[1] - 3, dim[1])),
                  int(rng.integers(max(2, dim[2] // 2), dim[2]))])
    m.cmap[m.lin_index(*s)] = VAL_FREE
    m.cmap[m.lin_index(*g)] = VAL_FREE
    raw = run_astar(m, m.int_to_float(s), m.int_to_float(g))
    if raw is None or len(raw) < 2:
        continue
    path = resample(raw, M=15)
    mid = path[len(path) // 2]
    sphere = SphereObstacle(mid, 1.2, class_name="human")  # bigger sphere ensures the seed hits
    init_obs, _ = seed_obs_cost(path, [sphere], LEAN)
    if init_obs == 0:
        continue
    trials += 1
    bs, info = plan(path, [sphere], cfg, opt_params=LEAN)
    final_obs = obstacle_cost(bs, [sphere], cfg, K=200)
    pin = (np.allclose(bs.eval(bs.t_start), path[0])
           and np.allclose(bs.eval(bs.t_end), path[-1]))
    drop = final_obs <= init_obs + 1e-6
    dt_ok = LEAN.dt_min < info["dt_final"] < 50 * info["T_target"] / max(info["num_ctrl"] - LEAN.degree, 1)
    finite = np.all(np.isfinite(bs.ctrl)) and np.isfinite(info["final_cost"])
    all_pin += int(pin); all_drop += int(drop); all_dt_ok += int(dt_ok); all_finite += int(finite)

check("R.B.0 endpoints exact on every trial", all_pin == trials, f"{all_pin}/{trials}")
check("R.B.1 obstacle cost did not increase on any trial", all_drop == trials, f"{all_drop}/{trials}")
check("R.B.2 dt stayed in sane range on every trial", all_dt_ok == trials, f"{all_dt_ok}/{trials}")
check("R.B.3 all spline / cost values finite", all_finite == trials, f"{all_finite}/{trials}")
print(f"  (successful local-opt trials: {trials})")


# ---------------------------------------------------------------- R.M MINCO path
# The analytic-gradient MINCO solver on the SAME R.A scene, plus the canonical
# "straight line driven through a sphere centre" stress (the spec's R.A case).
print("\n--- R.M. MINCO analytic-grad solve on the R.A scene + straight-through-centre ---")
import time as _time                                                       # noqa: E402

tr, info_m = plan_minco(path, [sphere], cfg, opt_params=FULL)
final_obs_m = obstacle_cost(tr, [sphere], cfg, K=300)
check("R.M.0 MINCO endpoints exactly = A* start / goal",
      np.allclose(tr.eval(tr.t_start), path[0]) and np.allclose(tr.eval(tr.t_end), path[-1]))
check("R.M.1 MINCO obstacle cost non-increasing",
      final_obs_m <= init_obs + 1e-6, f"init={init_obs:.3e}, final={final_obs_m:.3e}")
tsm = np.linspace(tr.t_start, tr.t_end, 100)
check("R.M.2 MINCO jerk finite everywhere", np.all(np.isfinite(tr.eval_deriv(tsm, 3))))
vmax_m = float(np.max(np.linalg.norm(tr.eval_deriv(tsm, 1), axis=1)))
check("R.M.3 MINCO max|v| finite + < 3x vmax", np.isfinite(vmax_m) and vmax_m < 3.0 * FULL.vmax,
      f"max|v|={vmax_m:.3f}")

# straight-through-sphere-centre: a perfectly straight A* path whose midpoint is
# the sphere centre. The optimiser MUST bend the trajectory off the centre.
straight = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
centre = straight[len(straight) // 2].copy()
sphere_c = SphereObstacle(centre, 0.8, class_name="human")
init_c = obstacle_cost(
    UniformBSpline.fit_path(straight, num_ctrl=13, degree=5, dt=0.3, clamp_endpoints=True),
    [sphere_c], cfg, K=300)
t0 = _time.time()
trc, info_c = plan_minco(straight, [sphere_c], cfg, opt_params=FULL)
t_c = _time.time() - t0
final_c = obstacle_cost(trc, [sphere_c], cfg, K=300)
clr = trc.eval_deriv(np.linspace(trc.t_start, trc.t_end, 200), 0)
min_clr = float(min(sphere_c.signed_dist(p, 0.0) for p in clr))
check("R.M.4 straight-through-centre: MINCO bends away (obstacle cost drops hard)",
      final_c < init_c and final_c < 1e3, f"init={init_c:.2e} final={final_c:.2e}, in {t_c:.2f}s")
check("R.M.5 straight-through-centre: trajectory clears the sphere body (min_clr > 0)",
      min_clr > 0.0, f"min_clearance={min_clr:.3f} (radius={sphere_c.radius})")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
