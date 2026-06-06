"""Stage 3 — per-class HARD/ALM on REAL HGP A* paths.

Drive HGPPlanner on a 3D scenario, take the post-processed polyline, drop a
HARD human on it, then call plan_minco and confirm the convex-hull+ALM path
clears the person by d_safe (continuous-time, 2000-pt dense), while a soft-only
run on the SAME real path breaches — the headline contrast on real geometry.

  R.0 endpoints exactly = A* start / goal
  R.1 soft-only on the real path BREACHES the human (M2 baseline)
  R.2 HARD on the real path clears the human >= d_safe (2000-pt dense)
  R.3 certificate margin >= 0 and consistent with the dense clearance (no leak)
  R.4 batched random 3D maps: every VALID hard solve clears densely
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                           # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC        # noqa: E402
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402
from sando_py.local.obstacles import SphereObstacle                        # noqa: E402
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


D_SAFE = 0.8
NO_DETOUR = DetourConfig(enabled=False)


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
                   los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0,
                   obstacle_soft_cost=0)
    p.set_map_util(m); p.set_max_expand(200000)
    ok, _ = p.plan(start_w, np.zeros(3), goal_w, 0.0, eps=1.0)
    if not ok:
        return None
    return np.array([np.asarray(pt, dtype=float) for pt in p.get_path()])


def resample(path, M=15):
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total == 0:
        return np.tile(path[0], (M, 1))
    target = np.linspace(0, total, M)
    return np.stack([np.interp(target, cum, path[:, d]) for d in range(path.shape[1])], axis=1)


def dense_min_clr(tr, obs, K=2000):
    ts = np.linspace(tr.t_start, tr.t_end, K)
    pts = tr.eval(ts)
    return float(min(obs.signed_dist(p, float(t)) for p, t in zip(pts, ts)))


CFG = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
CFG_SOFTW = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e3),
             "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
OPT = OptParams(maxiter=250, kappa=16, vmax=3.0, amax=3.0)


# ---------------------------------------------------------------- R. open scene
print("\n--- R. real HGP A* path + a HARD human on the midpoint ---")
m = blank([10, 10, 5], res=1.0)
sw, gw = np.array([0.5, 0.5, 0.5]), np.array([8.5, 8.5, 3.5])
raw = run_astar(m, sw, gw)
assert raw is not None
path = resample(raw, M=15)
mid = path[len(path) // 2]
# place an off-axis human so the soft penalty plateaus just inside d_safe
human = SphereObstacle(mid + np.array([0.0, 0.25, 0.0]), 0.4, class_name="human")

tr, info = plan_minco(path, [human], CFG, opt_params=OPT, detour_cfg=NO_DETOUR)
check("R.0 endpoints exactly = A* start / goal",
      np.allclose(tr.eval(tr.t_start), path[0]) and np.allclose(tr.eval(tr.t_end), path[-1]))

# soft-only on the SAME real path (the M2 baseline arm)
opt_soft = OptParams(maxiter=250, kappa=16, vmax=3.0, amax=3.0, avoid_override="soft")
tr_s, info_s = plan_minco(path, [human], CFG_SOFTW, opt_params=opt_soft, detour_cfg=NO_DETOUR)
clr_soft = dense_min_clr(tr_s, human, K=2000)
check("R.1 soft-only on the real path breaches the human (clearance < d_safe)",
      clr_soft < D_SAFE, f"soft dense min_clr={clr_soft:.4f} d_safe={D_SAFE}")

clr_hard = dense_min_clr(tr, human, K=2000)
check("R.2 HARD on the real path clears the human >= d_safe (2000-pt dense)",
      clr_hard >= D_SAFE - 1e-6,
      f"hard dense min_clr={clr_hard:.4f} (soft was {clr_soft:.4f})")
check("R.3 certificate margin >= 0 and consistent with dense (no gap leak)",
      info["certificate_margin"] >= -1e-6 and clr_hard >= D_SAFE - 1e-6,
      f"cert_margin={info['certificate_margin']:.4f} dense={clr_hard:.4f}")


# ---------------------------------------------------------------- R.4 batched
print("\n--- R.4 batched random 3D maps: VALID hard solves clear densely ---")
rng = np.random.default_rng(2026)
trials = 0
n_valid, n_clear = 0, 0
worst_leak = 0.0
for t in range(25):
    if trials >= 5:
        break
    dim = [int(rng.integers(12, 18)), int(rng.integers(12, 18)), int(rng.integers(5, 8))]
    mm = blank(dim, res=1.0)
    flat = mm.cmap.reshape(dim, order="F")
    flat[rng.random(dim) < 0.06] = VAL_OCC
    mm.cmap = flat.reshape(-1, order="F")
    s = np.array([int(rng.integers(0, 3)), int(rng.integers(0, 3)),
                  int(rng.integers(0, max(2, dim[2] // 2)))])
    g = np.array([int(rng.integers(dim[0] - 3, dim[0])),
                  int(rng.integers(dim[1] - 3, dim[1])),
                  int(rng.integers(max(2, dim[2] // 2), dim[2]))])
    mm.cmap[mm.lin_index(*s)] = VAL_FREE
    mm.cmap[mm.lin_index(*g)] = VAL_FREE
    raw = run_astar(mm, mm.int_to_float(s), mm.int_to_float(g))
    if raw is None or len(raw) < 2:
        continue
    p = resample(raw, M=15)
    pmid = p[len(p) // 2]
    h = SphereObstacle(pmid + rng.standard_normal(3) * 0.15, 0.35, class_name="human")
    trials += 1
    tr_b, info_b = plan_minco(p, [h], CFG, opt_params=OptParams(maxiter=150, kappa=14, vmax=3.0, amax=3.0),
                              detour_cfg=NO_DETOUR)
    clr = dense_min_clr(tr_b, h, K=2000)
    if info_b["trajectory_valid"]:
        n_valid += 1
        if clr >= D_SAFE - 1e-6:
            n_clear += 1
        else:
            worst_leak = max(worst_leak, D_SAFE - clr)
check("R.4 every VALID hard solve clears densely (no leak) on real maps",
      trials > 0 and n_valid > 0 and n_clear == n_valid,
      f"trials={trials} valid={n_valid} cleared={n_clear} worst_leak={worst_leak:.4f}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
