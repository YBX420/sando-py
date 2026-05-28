"""Stage 2b verification: hgp_planner post-processing (LOS shortcut + simplify).

The post-processing turns the raw zig-zag A* path into a few waypoints (what
local_opt will consume). What MUST hold (the "best" bar for this sector):
  1. SAFETY: every simplified segment is still collision-free (a shortcut must
     never cut a corner through an obstacle) — checked in 3D, sampled finer than
     the planner's own LOS step.
  2. endpoints preserved exactly (start, goal).
  3. it actually simplifies (fewer waypoints than the raw path).
  4. open space straightens to a near-straight line.

Run:  python3 test/stage2b_hgp_postprocess.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner            # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC  # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def make_map(dim, res=1.0):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.zeros(3)
    m.cmap = np.full(int(np.prod(dim)), VAL_FREE, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False
    m.initialized = True
    return m

def make_planner(m, los_cells=0):
    p = HGPPlanner(global_planner="astar", verbose=False, v_max=5.0, a_max=10.0, j_max=50.0,
                   timeout_duration_ms=5000, w_unknown=0.0, w_align=0.0, decay_len_cells=20.0,
                   w_side=0.0, los_cells=los_cells, min_len=0.5, min_turn=10.0,
                   heat_weight=0.0, obstacle_soft_cost=0.0)
    p.set_map_util(m)
    p.set_max_expand(500000)
    return p

def seg_free(m, a, b, step=0.02, margin=0.12):
    """Safe = segment does not cross the INTERIOR of any occupied cell (a
    measure-zero corner/face skim at the inflation boundary is allowed)."""
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

def path_free(m, path):
    return all(seg_free(m, path[i], path[i + 1]) for i in range(len(path) - 1))

print("=" * 60)
print("STAGE 2b — hgp_planner post-processing (3D) verification")
print("=" * 60)

# ---- Test 1: open 3D space straightens to a near-straight line ----
m = make_map([14, 14, 10])
s_cell, g_cell = np.array([1, 1, 1]), np.array([12, 12, 8])
sw, gw = m.int_to_float(s_cell), m.int_to_float(g_cell)
p = make_planner(m)
ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
raw, simp = p.get_raw_path(), p.get_path()
print(f"\nopen: raw={len(raw)} pts -> simplified={len(simp)} pts")
check("open: plan ok", ok and len(simp) >= 2)
check("open: simplified has fewer points than raw", len(simp) < len(raw))
check("open: straightens to <=3 waypoints", len(simp) <= 3, f"{len(simp)} pts")

# ---- Test 2: 3D obstacle block -> shortcut MUST stay collision-free ----
m = make_map([16, 16, 10])
# solid block: x in 7..9, y in 0..11, z in 0..6  (detour over the top z>=7 or around y>=12)
for x in (7, 8, 9):
    for y in range(0, 12):
        for z in range(0, 7):
            m.cmap[m.lin_index(x, y, z)] = VAL_OCC
s_cell, g_cell = np.array([2, 4, 3]), np.array([13, 4, 3])  # straight line pierces the block
sw, gw = m.int_to_float(s_cell), m.int_to_float(g_cell)
p = make_planner(m, los_cells=0)   # thinnest LOS = most aggressive shortcut = strictest safety test
ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
raw, simp = p.get_raw_path(), p.get_path()
print(f"\nblock: raw={len(raw)} pts -> simplified={len(simp)} pts")
check("block: plan ok & reaches goal", ok and np.allclose(simp[-1], gw))
check("block: SAFETY — every simplified segment collision-free (3D, fine sample)",
      path_free(m, simp))
check("block: raw path itself collision-free (sanity)", path_free(m, raw))
check("block: still simplified (fewer pts than raw)", len(simp) < len(raw))
# the straight start->goal line is NOT free (proves the block really blocks it)
check("block: straight start->goal would hit block (so detour was required)",
      not seg_free(m, sw, gw))

# ---- Test 3: endpoints preserved exactly ----
check("endpoints: simplified starts at start", np.allclose(simp[0], sw),
      f"{np.round(simp[0],2).tolist()} vs {np.round(sw,2).tolist()}")
check("endpoints: simplified ends at goal", np.allclose(simp[-1], gw),
      f"{np.round(simp[-1],2).tolist()} vs {np.round(gw,2).tolist()}")

# ---- Test 4: a 2nd obstacle layout (gap up high) — safety again ----
m = make_map([16, 16, 12])
wall_x = 8
gap_y, gap_z = 8, 10
for y in range(16):
    for z in range(12):
        if not (y == gap_y and z == gap_z):
            m.cmap[m.lin_index(wall_x, y, z)] = VAL_OCC
s_cell, g_cell = np.array([2, 8, 2]), np.array([14, 8, 2])
sw, gw = m.int_to_float(s_cell), m.int_to_float(g_cell)
p = make_planner(m, los_cells=0)
ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
simp = p.get_path()
check("wall+gap: plan ok & reaches goal", ok and np.allclose(simp[-1], gw))
check("wall+gap: SAFETY — simplified path collision-free (3D)", path_free(m, simp))

# ---- summary ----
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {n_fail} failed (of {len(results)})")
if n_fail:
    print("FAILED:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}  {detail}")
print("=" * 60)
