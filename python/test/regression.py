"""Regression tests — each of the 5 bugs found & fixed during verification,
pinned so a future code change can't silently bring them back.

Run:  python3 test/regression.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.graph_search import GraphSearch                       # noqa: E402
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC, VAL_UNK  # noqa: E402

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
    """Safe = segment does not cross the INTERIOR of any occupied cell."""
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

print("=" * 60)
print("REGRESSION — the 5 fixed bugs")
print("=" * 60)

# ---- R1: float_to_int uses floor (write==read; no -0.5 mismatch) ----
m = VoxelMapUtil(res=0.3)
m.origin = np.array([-1.55, -2.3, 0.4])
ok = True
for p in (np.array([0.0, 0.0, 1.0]), np.array([-1.2, -2.0, 0.5]),
          np.array([2.37, 1.04, 0.9]), np.array([-0.9, -2.29, 1.7])):
    if not np.array_equal(m.float_to_int(p), np.floor((p - m.origin) / m.res).astype(np.int64)):
        ok = False
check("R1: float_to_int == floor((p-origin)/res) (write/read same convention)", ok)

# ---- R2: post-processing won't simplify a path THROUGH a thin wall gap ----
m = blank([16, 16, 12])
for y in range(16):
    for z in range(12):
        if not (y == 8 and z == 10):
            m.cmap[m.lin_index(8, y, z)] = VAL_OCC
sw, gw = m.int_to_float(np.array([2, 8, 2])), m.int_to_float(np.array([14, 8, 2]))
p = HGPPlanner("astar", False, 5, 10, 50, 4000, 0, 0, 20, 0,
               los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
p.set_map_util(m); p.set_max_expand(300000)
ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
simp = p.get_path()
check("R2: simplified path through thin wall+gap stays collision-free",
      ok and all(seg_free(m, simp[i], simp[i + 1]) for i in range(len(simp) - 1)))

# ---- R3: A* does not corner-cut (diagonal grazing an occupied shoulder) ----
m = blank([4, 4, 2])
m.cmap[m.lin_index(2, 1, 0)] = VAL_OCC          # shoulder of the (1,1,0)->(2,2,0) diagonal
gs = GraphSearch(m, eps=1.0, global_planner="astar", heat_weight=0.0)
gs.plan(np.array([1, 1, 0]), np.array([2, 2, 0]), 0.0, np.zeros(3))
raw = gs.get_path_world()
no_graze = all(not m.is_blocked(np.asarray(raw[i]), np.asarray(raw[i + 1]))
               for i in range(len(raw) - 1))
took_direct = any(tuple(np.round(raw[i], 1)) == (1.5, 1.5, 0.5) and
                  tuple(np.round(raw[i + 1], 1)) == (2.5, 2.5, 0.5) for i in range(len(raw) - 1))
check("R3: A* path has no corner-cutting step (no segment grazes obstacle)",
      no_graze and not took_direct, f"steps={len(raw)-1}")

# ---- R4: is_blocked (DDA) catches a thin graze that point-sampling missed ----
m = blank([14, 14, 6])
m.cmap[m.lin_index(4, 8, 0)] = VAL_OCC
a, b = np.array([4.5, 9.5, 0.5]), np.array([7.5, 3.5, 5.5])   # grazes (4,8,0) briefly
check("R4: is_blocked (DDA) detects the thin-graze segment", m.is_blocked(a, b),
      f"occ cell (4,8,0) on segment")

# ---- R5: static heat reaches UNKNOWN cells on a fresh (un-carved) map ----
m = VoxelMapUtil(res=0.3)
m.dynamic_heat_enabled = False                  # isolate static heat
pillar = np.array([[1.0, 0.0, z] for z in np.linspace(0.4, 2.4, 25)])
m.read_map(24, 20, 10, np.array([0.0, 0.0, 1.2]), pillar, 0.0, 3.0, 0.3, [], [], 0.0)
unknown_with_heat = int(np.sum((m.cmap == VAL_UNK) & (m.heat > 0)))
check("R5: static heat is written onto UNKNOWN cells (apply_on_unknown)",
      unknown_with_heat > 0, f"{unknown_with_heat} unknown cells heated")

# ---- summary ----
n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
if n_pass != len(results):
    for name, ok, d in results:
        if not ok:
            print(f"  FAIL: {name}  {d}")
print("=" * 60)
