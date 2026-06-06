"""Extreme / boundary conditions — the planner must never crash and must give
sane answers at the edges of its input domain.

Run:  python3 test/boundary.py
"""
import os
import sys
import traceback

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC       # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def mk(dim, res=1.0, origin=(0.0, 0.0, 0.0)):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.array(origin, dtype=np.float64)
    m.cmap = np.full(int(np.prod(dim)), VAL_FREE, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False
    m.initialized = True
    return m

def seg_free(m, a, b, step=0.02, margin=0.12):
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
            return False
    return True

def plan_safe(m, sw, gw):
    """Run plan; return (crashed, ok, path)."""
    try:
        p = HGPPlanner("astar", False, 5, 10, 50, 3000, 0, 0, 20, 0,
                       los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
        p.set_map_util(m); p.set_max_expand(300000)
        ok, _ = p.plan(np.asarray(sw, float), np.zeros(3), np.asarray(gw, float), 0.0, eps=1.0)
        return False, ok, p.get_path()
    except Exception:
        traceback.print_exc()
        return True, False, []

def path_ok(m, path):
    return len(path) >= 2 and all(seg_free(m, path[i], path[i + 1]) for i in range(len(path) - 1))

print("=" * 60)
print("EXTREME / BOUNDARY CONDITIONS")
print("=" * 60)

# B1 start/goal pressed against a wall
m = mk([12, 12, 6])
for y in range(12):
    for z in range(6):
        m.cmap[m.lin_index(6, y, z)] = VAL_OCC
m.cmap[m.lin_index(6, 5, 3)] = VAL_FREE   # single gap
crashed, ok, path = plan_safe(m, m.int_to_float(np.array([5, 5, 3])),
                              m.int_to_float(np.array([7, 5, 3])))
check("B1: start/goal next to wall, gap -> safe path", (not crashed) and ok and path_ok(m, path))

# B2 start/goal at map corners
m = mk([10, 10, 6])
crashed, ok, path = plan_safe(m, m.int_to_float(np.array([0, 0, 0])),
                              m.int_to_float(np.array([9, 9, 5])))
check("B2: corner-to-corner -> safe path", (not crashed) and ok and path_ok(m, path))

# B3 negative origin / negative world coords
m = mk([14, 14, 8], res=0.3, origin=(-2.5, -2.5, -0.5))
sw = m.int_to_float(np.array([1, 1, 1])); gw = m.int_to_float(np.array([12, 12, 6]))
crashed, ok, path = plan_safe(m, sw, gw)
check("B3: negative origin/coords -> safe path", (not crashed) and ok and path_ok(m, path)
      and sw[0] < 0 and sw[1] < 0)

# B4 tiny map
m = mk([3, 3, 3])
crashed, ok, path = plan_safe(m, m.int_to_float(np.array([0, 0, 0])),
                              m.int_to_float(np.array([2, 2, 2])))
check("B4: tiny 3x3x3 map -> safe path", (not crashed) and ok and path_ok(m, path))

# B5 unsolvable (goal sealed inside an OCC shell)
m = mk([12, 12, 8])
for x in range(7, 10):
    for y in range(4, 8):
        for z in range(2, 6):
            on_shell = x in (7, 9) or y in (4, 7) or z in (2, 5)
            if on_shell:
                m.cmap[m.lin_index(x, y, z)] = VAL_OCC
m.cmap[m.lin_index(8, 5, 3)] = VAL_FREE  # sealed pocket
crashed, ok, path = plan_safe(m, m.int_to_float(np.array([1, 1, 1])),
                              m.int_to_float(np.array([8, 5, 3])))
check("B5: goal sealed -> reports NO path, no crash", (not crashed) and (not ok))

# B6 start == goal
m = mk([8, 8, 4])
sw = m.int_to_float(np.array([3, 3, 2]))
crashed, ok, path = plan_safe(m, sw, sw)
check("B6: start == goal -> no crash, trivial path", (not crashed) and (len(path) >= 1))

# B7 start inside an obstacle (pathological input) -> must not crash
m = mk([10, 10, 6])
m.cmap[m.lin_index(3, 3, 2)] = VAL_OCC
crashed, ok, path = plan_safe(m, m.int_to_float(np.array([3, 3, 2])),
                              m.int_to_float(np.array([8, 8, 4])))
check("B7: start in obstacle -> no crash", not crashed)

# B8 voxel-boundary coordinates (points exactly on cell edges)
m = mk([10, 10, 6], res=1.0, origin=(0.0, 0.0, 0.0))
crashed, ok, path = plan_safe(m, np.array([1.0, 1.0, 1.0]), np.array([8.0, 8.0, 5.0]))
check("B8: coords exactly on voxel boundaries -> no crash, safe path",
      (not crashed) and ok and path_ok(m, path))

# B9 goal outside map bounds -> graceful (no crash)
m = mk([10, 10, 6])
crashed, ok, path = plan_safe(m, m.int_to_float(np.array([1, 1, 1])),
                              np.array([99.0, 99.0, 99.0]))
check("B9: goal out of bounds -> no crash", not crashed)

n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
if n_pass != len(results):
    for nm, ok, d in results:
        if not ok:
            print(f"  FAIL: {nm}  {d}")
print("=" * 60)
