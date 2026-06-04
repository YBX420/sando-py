"""Stage 3 — local/cost.py on REAL global-pipeline paths.

Same setup as stage3_bspline_real: drive HGPPlanner on 3D scenarios, fit a
quintic uniform B-spline (clamped at A* start/goal), and evaluate the
per-class avoidance cost in three regimes (obstacle far / near / on the
path) plus a batched random sweep.

  R.A.0  empty obstacle list → cost = 0
  R.A.1  obstacle planted on the spline → cost > 0
  R.A.2  obstacle far from the path → cost = 0
  R.A.3  pulling the sphere closer increases cost monotonically
  R.A.4  hard-class weight ≫ soft-class weight at same geometry → hard > soft

  R.B.*  batched random 3D maps (seed=2026): on each successful run,
         cost is finite, non-negative, > 0 when the obstacle sits inside the
         spline's safety band.
"""
import importlib.util
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC       # noqa: E402
from sando_py.local.avoid_config import AvoidParams                       # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.cost import obstacle_cost                             # noqa: E402

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local")
spec = importlib.util.spec_from_file_location(
    "bspline_standalone", os.path.abspath(os.path.join(LOCAL_DIR, "bspline.py")))
_bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(_bs)
UniformBSpline = _bs.UniformBSpline

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


def make_spline(path, degree=5, dt=1.0):
    n_ctrl = max(2 * (degree + 1), int(round(len(path) / 2)))
    return UniformBSpline.fit_path(path, num_ctrl=n_ctrl, degree=degree, dt=dt, clamp_endpoints=True)


cfg = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}


# ---------------------------------------------------------------- R.A. open 3D scene
print("\n--- R.A. open 3D scene (diagonal) ---")
m = blank([10, 10, 5], res=1.0)
sw, gw = np.array([0.5, 0.5, 0.5]), np.array([8.5, 8.5, 3.5])
path = run_astar(m, sw, gw)
assert path is not None
bs = make_spline(path)
# midpoint of the spline (roughly)
mid = bs.eval(0.5 * (bs.t_start + bs.t_end))

# R.A.0 empty
check("R.A.0 empty obstacle list → cost = 0", obstacle_cost(bs, [], cfg) == 0.0)

# R.A.1 obstacle on the path
obs_on = [SphereObstacle(mid, 0.3, class_name="human")]
c_on = obstacle_cost(bs, obs_on, cfg, K=300)
check("R.A.1 obstacle on the spline → cost > 0", c_on > 0, f"cost={c_on:.3e}")

# R.A.2 obstacle far from the path
obs_far = [SphereObstacle(mid + np.array([0.0, 50.0, 0.0]), 0.3, class_name="human")]
c_far = obstacle_cost(bs, obs_far, cfg, K=300)
check("R.A.2 obstacle far from the path → cost = 0", c_far == 0.0)

# R.A.3 sliding the obstacle in along the perp axis monotonically increases cost
costs = []
for offset_y in (10.0, 3.0, 1.5, 0.5, 0.0):
    obs = [SphereObstacle(mid + np.array([0.0, offset_y, 0.0]), 0.3, class_name="human")]
    costs.append(obstacle_cost(bs, obs, cfg, K=300))
monotone = all(costs[i] <= costs[i + 1] + 1e-9 for i in range(len(costs) - 1))
check("R.A.3 cost monotone-nondecreasing as the obstacle slides toward the spline",
      monotone, f"costs={[f'{c:.2e}' for c in costs]}")

# R.A.4 hard vs soft at same geometry
cfg_soft = {"human": AvoidParams("human", "soft", 0.8, 10.0)}
cfg_hard = {"human": AvoidParams("human", "hard", 0.8, 1.0e4)}
c_soft = obstacle_cost(bs, obs_on, cfg_soft, K=300)
c_hard = obstacle_cost(bs, obs_on, cfg_hard, K=300)
check("R.A.4 hard-class cost >> soft-class cost at same geometry", c_hard > 100 * c_soft,
      f"soft={c_soft:.3e}, hard={c_hard:.3e}")


# ---------------------------------------------------------------- R.B. batched random 3D maps
print("\n--- R.B. batched random 3D maps (seed=2026) ---")
rng = np.random.default_rng(2026)
trials, all_finite, all_nonneg, all_hit = 0, 0, 0, 0
for t in range(50):
    dim = [int(rng.integers(16, 24)), int(rng.integers(16, 24)), int(rng.integers(6, 10))]
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
    path = run_astar(m, m.int_to_float(s), m.int_to_float(g))
    if path is None or len(path) < 4:
        continue
    bs = make_spline(path)
    # 1) far obstacle: cost must be 0
    far = [SphereObstacle(np.array([1e4, 1e4, 1e4]), 0.3, class_name="human")]
    c_far = obstacle_cost(bs, far, cfg, K=200)
    # 2) obstacle pinned at the spline midpoint: cost must be > 0 and finite
    midp = bs.eval(0.5 * (bs.t_start + bs.t_end))
    near = [SphereObstacle(midp, 0.4, class_name="human")]
    c_near = obstacle_cost(bs, near, cfg, K=200)
    trials += 1
    all_finite += int(np.isfinite(c_far) and np.isfinite(c_near))
    all_nonneg += int(c_far >= 0 and c_near >= 0)
    all_hit += int(c_far == 0.0 and c_near > 0.0)

check("R.B.0 all costs finite", all_finite == trials, f"{all_finite}/{trials}")
check("R.B.1 all costs ≥ 0", all_nonneg == trials, f"{all_nonneg}/{trials}")
check("R.B.2 far=0, on-path>0 on every run", all_hit == trials, f"{all_hit}/{trials}")
print(f"  (successful random trials: {trials})")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    raise SystemExit(1)
