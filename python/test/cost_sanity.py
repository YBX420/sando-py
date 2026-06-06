"""Cost-function sanity: when a weight changes, the path behaves as expected
(monotonically), not randomly.

  C1 heat_weight up   -> path keeps further from heat (lower heat exposure)
  C2 w_unknown up     -> path prefers known cells (fewer unknown cells traversed)
  C3 inflation up     -> path keeps more clearance from obstacles

Run:  python3 test/cost_sanity.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.graph_search import GraphSearch                          # noqa: E402
from sando_py.hgp.hgp_planner import HGPPlanner                             # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC, VAL_UNK # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def monotone_noninc(xs, tol=1e-6):
    return all(xs[i] >= xs[i + 1] - tol for i in range(len(xs) - 1))

print("=" * 60)
print("COST-FUNCTION SANITY")
print("=" * 60)

# ---- C1: heat_weight up -> lower heat exposure ----
m = VoxelMapUtil(res=1.0)
m.dim = np.array([22, 22, 3]); m.origin = np.zeros(3)
m.cmap = np.full(22 * 22 * 3, VAL_FREE, np.int8)
m.heat = np.zeros(22 * 22 * 3, np.float32)
m.use_soft_cost_obstacles = False; m.initialized = True
center = np.array([10.5, 10.5, 1.5])
for x in range(22):
    for y in range(22):
        for z in range(3):
            if np.linalg.norm(np.array([x, y, z]) + 0.5 - center) <= 2.5:
                m.heat[m.lin_index(x, y, z)] = 4.0
si, gi = np.array([2, 10, 1]), np.array([19, 10, 1])

def heat_exposure(weight):
    gs = GraphSearch(m, eps=1.0, global_planner="astar_heat", heat_weight=weight)
    gs.plan(si, gi, 0.0, np.zeros(3), max_expand=400000, timeout_ms=5000)
    path = gs.get_path_world()
    tot = 0.0
    for i in range(len(path) - 1):
        a, b = np.asarray(path[i]), np.asarray(path[i + 1])
        n = max(1, int(np.linalg.norm(b - a) / 0.5))
        for k in range(n + 1):
            c = [int(v) for v in m.float_to_int(a + (b - a) * (k / n))]
            tot += m.get_heat(*c)
    return tot

exps = [heat_exposure(w) for w in (0.0, 2.0, 4.0, 8.0, 16.0)]
print(f"heat exposure vs weight [0,2,4,8,16]: {[round(e,1) for e in exps]}")
check("C1: heat exposure is non-increasing as heat_weight rises", monotone_noninc(exps))
check("C1: high heat_weight gives far less exposure than zero", exps[-1] < exps[0] - 1e-3)

# ---- C2: w_unknown up -> fewer unknown cells traversed ----
dim = [9, 5, 1]
m = VoxelMapUtil(res=1.0); m.dim = np.array(dim); m.origin = np.zeros(3)
m.cmap = np.full(int(np.prod(dim)), VAL_UNK, np.int8)   # default UNKNOWN
m.heat = np.zeros(int(np.prod(dim)), np.float32)
m.use_soft_cost_obstacles = False; m.initialized = True
# carve a FREE detour: column x=0 (y0..2), row y=0 (all x), column x=8 (y0..2)
for y in range(3):
    m.cmap[m.lin_index(0, y, 0)] = VAL_FREE
    m.cmap[m.lin_index(8, y, 0)] = VAL_FREE
for x in range(9):
    m.cmap[m.lin_index(x, 0, 0)] = VAL_FREE
si, gi = np.array([0, 2, 0]), np.array([8, 2, 0])

def unknown_count(w_unk):
    gs = GraphSearch(m, eps=1.0, global_planner="astar", w_unknown=w_unk, heat_weight=0.0)
    gs.plan(si, gi, 0.0, np.zeros(3), max_expand=200000, timeout_ms=3000)
    cells = [tuple(int(v) for v in m.float_to_int(p)) for p in gs.get_path_world()]
    return sum(1 for c in cells if m.cmap[m.lin_index(*c)] == VAL_UNK)

uks = [unknown_count(w) for w in (0.0, 0.5, 2.0)]
print(f"unknown cells in path vs w_unknown [0,0.5,2]: {uks}")
check("C2: fewer unknown cells traversed as w_unknown rises", monotone_noninc([float(u) for u in uks]))
check("C2: high w_unknown avoids unknown vs w_unknown=0", uks[-1] < uks[0])

# ---- C3: inflation up -> more clearance from obstacle ----
def min_clear(infl):
    m = VoxelMapUtil(res=0.3); m.use_soft_cost_obstacles = False
    pillar = np.array([[1.0, 0.0, z] for z in np.linspace(0.4, 2.2, 16)])
    m.read_map(30, 30, 10, np.array([0.0, 0.0, 1.2]), pillar, 0.0, 3.0, infl, [], [], 0.0)
    p = HGPPlanner("astar", False, 5, 10, 50, 4000, 0, 0, 20, 0,
                   los_cells=0, min_len=0.3, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
    p.set_map_util(m); p.set_max_expand(400000)
    ok, _ = p.plan(np.array([-2.5, 0.0, 1.2]), np.zeros(3), np.array([4.0, 0.0, 1.2]), 0.0, eps=1.0)
    if not ok:
        return float("inf")
    path = p.get_path()
    md = float("inf")
    for i in range(len(path) - 1):
        a, b = np.asarray(path[i]), np.asarray(path[i + 1])
        n = max(1, int(np.linalg.norm(b - a) / 0.1))
        for k in range(n + 1):
            q = a + (b - a) * (k / n)
            md = min(md, float(np.hypot(q[0] - 1.0, q[1] - 0.0)))
    return md

clears = [min_clear(infl) for infl in (0.1, 0.3, 0.6)]
print(f"min clearance to pillar vs inflation [0.1,0.3,0.6]: {[round(c,2) for c in clears]}")
check("C3: clearance is non-decreasing as inflation rises",
      all(clears[i] <= clears[i + 1] + 1e-6 for i in range(len(clears) - 1)))
check("C3: bigger inflation gives more clearance than small", clears[-1] > clears[0] + 1e-3)

n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
if n_pass != len(results):
    for nm, ok, d in results:
        if not ok:
            print(f"  FAIL: {nm}  {d}")
print("=" * 60)
