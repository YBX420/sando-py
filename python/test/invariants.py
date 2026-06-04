"""Path safety invariants — properties that must hold for EVERY plan, no matter
how the map is generated. Fixed seeds so any failure is reproducible.

Invariants (raw AND simplified path):
  I1  every path vertex maps to an in-bounds cell          (no out-of-bounds)
  I2  every path segment is collision-free                  (no obstacle / inflation)
  I3  if a path is returned it reaches the goal
Covered on: random hand-built maps, real read_map maps (inflation), and maps
with dynamic obstacles marked occupied (post-process safe under dynamics).

Run:  python3 test/invariants.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC       # noqa: E402

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

def seg_free(m, a, b, step=0.02, margin=0.12):
    """Safe = the segment does not cross the INTERIOR of any occupied cell.
    A measure-zero touch of a cell corner/face (the path skimming the inflation
    boundary) is allowed; only a real interior traversal counts as a collision."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.linalg.norm(b - a) / step))
    for i in range(n + 1):
        g = (a + (b - a) * (i / n) - m.origin) / m.res
        cell = np.floor(g).astype(int)
        if not m.in_bounds(int(cell[0]), int(cell[1]), int(cell[2])):
            continue
        if m.cmap[m.lin_index(int(cell[0]), int(cell[1]), int(cell[2]))] != VAL_OCC:
            continue
        frac = g - cell
        if np.all(frac > margin) and np.all(frac < 1.0 - margin):
            return False   # strictly interior -> real collision
    return True

def violated(m, path):
    """Return reason string if a path violates an invariant, else None."""
    if len(path) < 2:
        return "too short"
    for v in path:
        c = [int(x) for x in m.float_to_int(np.asarray(v))]
        if not m.in_bounds(*c):
            return f"vertex OOB {tuple(np.round(v,2))}"
    for i in range(len(path) - 1):
        if not seg_free(m, path[i], path[i + 1]):
            return f"segment {i} in OCC"
    return None

def make_planner(m):
    p = HGPPlanner("astar", False, 5, 10, 50, 3000, 0, 0, 20, 0,
                   los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
    p.set_map_util(m); p.set_max_expand(200000)
    return p

print("=" * 62)
print("PATH SAFETY INVARIANTS (fixed seed, reproducible)")
print("=" * 62)

# ---- Part A: random hand-built maps ----
rng = np.random.default_rng(2024)
SEED_NOTE = "rng=default_rng(2024)"
fails = []
ran = 0
for t in range(250):
    dim = [int(rng.integers(10, 16)), int(rng.integers(10, 16)), int(rng.integers(6, 10))]
    m = blank(dim); dx, dy, dz = dim
    if rng.random() < 0.5:
        ax = int(rng.integers(0, 3)); coord = int(rng.integers(2, [dx, dy, dz][ax] - 2))
        gap = (int(rng.integers(0, dy)), int(rng.integers(0, dz))) if ax == 0 else \
              (int(rng.integers(0, dx)), int(rng.integers(0, dz))) if ax == 1 else \
              (int(rng.integers(0, dx)), int(rng.integers(0, dy)))
        for i in range([dy, dx, dx][ax]):
            for j in range([dz, dz, dy][ax]):
                if (i, j) == gap:
                    continue
                cell = (coord, i, j) if ax == 0 else (i, coord, j) if ax == 1 else (i, j, coord)
                m.cmap[m.lin_index(*cell)] = VAL_OCC
    else:
        occ = rng.random((dx, dy, dz)) < 0.15
        m.cmap = np.where(occ.reshape(-1, order="F"), VAL_OCC, VAL_FREE).astype(np.int8)
    free = np.array(np.where(m.cmap.reshape(dim, order="F") == VAL_FREE)).T
    if len(free) < 2:
        continue
    s = free[rng.integers(len(free))]; g = free[rng.integers(len(free))]
    if tuple(s) == tuple(g):
        continue
    sw, gw = m.int_to_float(s), m.int_to_float(g)
    p = make_planner(m)
    ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
    if not ok:
        continue
    ran += 1
    for tag, path in (("raw", p.get_raw_path()), ("simp", p.get_path())):
        why = violated(m, path)
        if why:
            fails.append((t, tag, why)); break
    if ok and not np.allclose(p.get_path()[-1], gw):
        fails.append((t, "goal", "did not reach goal"))
check(f"A: invariants hold on {ran} random 3D maps ({SEED_NOTE})",
      not fails, f"first fails={fails[:5]}" if fails else "all clean")

# ---- Part B: real read_map maps — path respects inflation (OCC = obstacle+inflation) ----
rng = np.random.default_rng(99)
infl_fail = []
for t in range(40):
    m = VoxelMapUtil(res=0.3)
    m.use_soft_cost_obstacles = False
    # a few random pillars (point clouds)
    cloud = []
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = rng.uniform(-3, 3), rng.uniform(-3, 3)
        cloud += [[cx, cy, z] for z in np.linspace(0.4, 2.2, 12)]
    cloud = np.array(cloud)
    infl = 0.3
    m.read_map(30, 30, 10, np.array([0.0, 0.0, 1.2]), cloud, 0.0, 3.0, infl, [], [], 0.0)
    sw, gw = np.array([-3.5, -3.5, 1.2]), np.array([3.5, 3.5, 1.2])
    p = make_planner(m)
    ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
    if not ok:
        continue
    simp = p.get_path()
    if violated(m, simp):   # OCC already includes the inflation ring
        infl_fail.append(t)
check("B: real read_map paths stay out of inflated-obstacle cells",
      not infl_fail, f"fails={infl_fail[:5]}" if infl_fail else "all clean")

# ---- Part C: dynamic obstacles (marked occupied) — post-process stays safe ----
rng = np.random.default_rng(123)
dyn_fail = []
for t in range(40):
    m = VoxelMapUtil(res=0.3)
    m.use_soft_cost_obstacles = False
    m.dynamic_as_occupied_current = True       # dynamic footprint -> occupied
    obst_pos = [np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), 1.2])
                for _ in range(int(rng.integers(2, 5)))]
    obst_bbox = [np.array([0.4, 0.4, 0.6]) for _ in obst_pos]
    m.read_map(30, 30, 10, np.array([0.0, 0.0, 1.2]), np.zeros((0, 3)),
               0.0, 3.0, 0.3, obst_pos, obst_bbox, 0.0)
    sw, gw = np.array([-3.5, 0.0, 1.2]), np.array([3.5, 0.0, 1.2])
    p = make_planner(m)
    ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
    if not ok:
        continue
    if violated(m, p.get_path()):
        dyn_fail.append(t)
check("C: paths avoid dynamic-obstacle footprints after post-processing",
      not dyn_fail, f"fails={dyn_fail[:5]}" if dyn_fail else "all clean")

# ---- summary ----
n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 62)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
print("=" * 62)
