"""Stage 2 verification: global A* (graph_search.py).

All cases are genuine 3D (start/goal differ in x,y,z; gaps and heat live in 3D).
Maps are hand-built (cmap set directly) for full control — no read_map, no ROS.

Run:  python3 test/stage2_astar.py
Checks (by FUNCTION, not "matches C++"):
  1. empty 3D space  -> path found, cost ~ Euclidean (no wandering)
  2. wall with a gap (gap on a different z) -> path threads the gap, stays free
  3. fully sealed wall -> reports NO path (no crash, no fake path)
  4. heat field on -> path bends away from the hot blob; bigger weight bends more
  5. optimality -> A* cost == brute-force Dijkstra cost on random 3D maps
"""
import heapq
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.graph_search import GraphSearch          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC  # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

# ---------------------------------------------------------------- map builders
def make_map(dim, res=1.0):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.zeros(3)
    m.cmap = np.full(int(np.prod(dim)), VAL_FREE, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False   # OCC truly blocks
    m.initialized = True
    return m

def set_occ(m, x, y, z):
    if m.in_bounds(x, y, z):
        m.cmap[m.lin_index(x, y, z)] = VAL_OCC

def make_gs(m, eps=1.0, planner="astar", heat_weight=0.0):
    return GraphSearch(m, eps=eps, global_planner=planner,
                       w_unknown=0.0, heat_weight=heat_weight, obstacle_soft_cost=0.0)

def run(gs, start, goal):
    ok = gs.plan(np.array(start), np.array(goal), 0.0, np.zeros(3),
                 max_expand=500000, timeout_ms=5000)
    path = [(s.x, s.y, s.z) for s in gs._path]
    sid = gs._coord_to_id(*goal)
    gstate = gs._hm[sid]
    gcost = gstate.g if (gstate is not None) else float("inf")
    return ok, path, gcost

def corner_cut(m, u, move):
    """Same no-corner-cutting rule the A* uses (for a matched comparison)."""
    from itertools import combinations
    nz = [(i, d) for i, d in enumerate(move) if d != 0]
    if len(nz) <= 1:
        return False
    for r in range(1, len(nz)):
        for combo in combinations(nz, r):
            s = [0, 0, 0]
            for i, d in combo:
                s[i] = d
            c = (u[0] + s[0], u[1] + s[1], u[2] + s[2])
            if (not m.in_bounds(*c)) or m.cmap[m.lin_index(*c)] == VAL_OCC:
                return True
    return False

def dijkstra(m, start, goal):
    """Brute-force optimal cost, same 26-connected euclidean edges, OCC blocked,
    no corner-cutting (matches A*)."""
    nbrs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)]
    start, goal = tuple(start), tuple(goal)
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == goal:
            return d
        for dx, dy, dz in nbrs:
            v = (u[0] + dx, u[1] + dy, u[2] + dz)
            if not m.in_bounds(*v):
                continue
            if m.cmap[m.lin_index(*v)] == VAL_OCC:
                continue
            if corner_cut(m, u, (dx, dy, dz)):
                continue
            nd = d + (dx * dx + dy * dy + dz * dz) ** 0.5
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float("inf")

def path_is_free(m, path):
    return all(m.cmap[m.lin_index(*c)] != VAL_OCC for c in path)

print("=" * 60)
print("STAGE 2 — global A* (3D) verification")
print("=" * 60)

# ---- Test 1: empty 3D -> near-straight ----
m = make_map([14, 14, 10])
start, goal = (1, 1, 1), (12, 12, 8)
ok, path, gcost = run(make_gs(m), start, goal)
euclid = float(np.linalg.norm(np.array(goal) - np.array(start)))
check("empty 3D: path found & reaches goal", ok and path and path[-1] == goal)
check("empty 3D: cost ~ Euclidean (<=1.15x, no wandering)",
      euclid <= gcost <= 1.15 * euclid, f"cost={gcost:.2f} euclid={euclid:.2f}")

# ---- Test 2: wall with a gap on a different z (forces 3D detour) ----
m = make_map([14, 14, 10])
wall_x = 7
gap = (wall_x, 7, 8)   # the only opening, up high
for y in range(14):
    for z in range(10):
        if (wall_x, y, z) != gap:
            set_occ(m, wall_x, y, z)
start, goal = (2, 7, 2), (12, 7, 2)   # both low; must climb to z=8 to pass
ok, path, gcost = run(make_gs(m), start, goal)
threads_gap = any(c[0] == wall_x for c in path) and all(
    (c[0] != wall_x) or (c == gap) for c in path)
check("wall+gap: path found & reaches goal", ok and path and path[-1] == goal)
check("wall+gap: path stays in free space", path_is_free(m, path))
check("wall+gap: path goes through the single gap cell", gap in path,
      f"gap={gap}")
check("wall+gap: crossing happens only at the gap", threads_gap)

# ---- Test 3: fully sealed wall -> no path ----
m = make_map([14, 14, 10])
for y in range(14):
    for z in range(10):
        set_occ(m, 7, y, z)
ok, path, gcost = run(make_gs(m), (2, 7, 5), (12, 7, 5))
reached = bool(path) and path[-1] == (12, 7, 5)
check("sealed wall: reports NO path", (not ok) and (not reached),
      f"ok={ok} reached_goal={reached}")

# ---- Test 4: heat field -> path bends away; bigger weight bends more ----
m = make_map([22, 22, 10])
start, goal = (2, 2, 2), (19, 19, 7)
center = np.array([10.5, 10.5, 4.5])     # on the straight line, in 3D
Rb = 2.5
for x in range(22):
    for y in range(22):
        for z in range(10):
            if np.linalg.norm(np.array([x, y, z]) - center) <= Rb:
                m.heat[m.lin_index(x, y, z)] = 5.0

def min_dist_to_center(path):
    return min(float(np.linalg.norm(np.array(c) - center)) for c in path)

_, path_off, _ = run(make_gs(m, planner="astar", heat_weight=0.0), start, goal)
_, path_lo, _ = run(make_gs(m, planner="astar_heat", heat_weight=5.0), start, goal)
_, path_hi, _ = run(make_gs(m, planner="astar_heat", heat_weight=50.0), start, goal)
d_off, d_lo, d_hi = (min_dist_to_center(p) for p in (path_off, path_lo, path_hi))
print(f"\nmin dist to hot center: heat_off={d_off:.2f} weight5={d_lo:.2f} weight50={d_hi:.2f}")
check("heat: path bends away vs no-heat", d_hi > d_off, f"{d_hi:.2f} > {d_off:.2f}")
check("heat: bigger weight bends more", d_hi >= d_lo >= d_off - 1e-9,
      f"{d_hi:.2f} >= {d_lo:.2f} >= {d_off:.2f}")
check("heat: hot path avoids the blob interior (min dist >= Rb-ish)",
      d_hi >= Rb - 1.0, f"min={d_hi:.2f} Rb={Rb}")

# ---- Test 5: optimality vs Dijkstra on random 3D maps ----
rng = np.random.default_rng(0)
all_match = True
trials = 0
for t in range(6):
    m = make_map([10, 10, 7])
    # random obstacles (~18%), keep start/goal free
    for x in range(10):
        for y in range(10):
            for z in range(7):
                if rng.random() < 0.18:
                    set_occ(m, x, y, z)
    start, goal = (0, 0, 0), (9, 9, 6)
    m.cmap[m.lin_index(*start)] = VAL_FREE
    m.cmap[m.lin_index(*goal)] = VAL_FREE
    ok, path, gcost = run(make_gs(m), start, goal)
    dcost = dijkstra(m, start, goal)
    reachable = dcost < float("inf")
    if not reachable:
        continue
    trials += 1
    match = ok and abs(gcost - dcost) < 1e-6
    all_match = all_match and match
    if not match:
        print(f"  trial {t}: A*={gcost:.4f} Dijkstra={dcost:.4f} ok={ok}")
check(f"optimality: A* cost == Dijkstra on {trials} reachable 3D maps", all_match)

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
