"""Stress / realistic verification — replaces the weak spot-checks with the
things that actually matter (and that hid a bug before).

Sections:
  A. post-processing SAFETY under random 3D maps (incl. thin walls w/ gaps)
  B. A* optimality WITH unknown cells (+w_unknown), vs brute-force Dijkstra
  C. A* on a REAL read_map output (obstacles + heat + unknown together)
  D. dynamic prediction: heat follows where the obstacle WILL be, not where it is

Run:  python3 test/stage_stress.py
"""
import heapq
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

NBRS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)]

def corner_cut(m, u, move):
    """Same no-corner-cutting rule the A* uses (matched comparison)."""
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

def blank(dim, fill=VAL_FREE, res=1.0):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.zeros(3)
    m.cmap = np.full(int(np.prod(dim)), fill, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False
    m.initialized = True
    return m

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

# ======================================================================
# A. post-processing SAFETY under random 3D maps
# ======================================================================
print("=" * 64)
print("A. post-processing safety — random 3D maps (200 trials)")
print("=" * 64)
rng = np.random.default_rng(1)
N = 200
unsafe = []
ran = 0
for t in range(N):
    dim = [int(rng.integers(10, 16)), int(rng.integers(10, 16)), int(rng.integers(6, 10))]
    m = blank(dim)
    dx, dy, dz = dim
    if rng.random() < 0.5:
        # thin wall (random axis) with a random gap -> the structure that hid the bug
        ax = int(rng.integers(0, 3))
        coord = int(rng.integers(2, [dx, dy, dz][ax] - 2))
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
        # scattered random blocks (~14%)
        occ = rng.random((dx, dy, dz)) < 0.14
        m.cmap = np.where(occ.reshape(-1, order="F"), VAL_OCC, VAL_FREE).astype(np.int8)
    # random free start/goal
    free = np.array(np.where(m.cmap.reshape(dim, order="F") == VAL_FREE)).T
    if len(free) < 2:
        continue
    s = free[rng.integers(len(free))]
    g = free[rng.integers(len(free))]
    if tuple(s) == tuple(g):
        continue
    sw, gw = m.int_to_float(s), m.int_to_float(g)
    p = HGPPlanner("astar", False, 5, 10, 50, 3000, 0, 0, 20, 0,
                   los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
    p.set_map_util(m); p.set_max_expand(200000)
    ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
    if not ok:
        continue
    ran += 1
    simp = p.get_path()
    if not all(seg_free(m, simp[i], simp[i + 1]) for i in range(len(simp) - 1)):
        unsafe.append(t)
        if np.allclose(simp[-1], gw) is False:
            pass
check(f"A: no simplified path clips an obstacle ({ran} solvable trials)",
      not unsafe, f"unsafe seeds={unsafe[:8]}" if unsafe else "all safe")

# ======================================================================
# B. A* optimality WITH unknown cells (+w_unknown) vs Dijkstra
# ======================================================================
print("\n" + "=" * 64)
print("B. A* optimality with UNKNOWN + w_unknown vs Dijkstra (30 trials)")
print("=" * 64)
def dijkstra(m, start, goal, w_unk):
    start, goal = tuple(start), tuple(goal)
    dist = {start: 0.0}; pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u == goal:
            return d
        for off in NBRS:
            v = (u[0] + off[0], u[1] + off[1], u[2] + off[2])
            if not m.in_bounds(*v):
                continue
            val = m.cmap[m.lin_index(*v)]
            if val == VAL_OCC:
                continue
            if corner_cut(m, u, off):
                continue
            step = (off[0]**2 + off[1]**2 + off[2]**2) ** 0.5
            w = step * (1.0 + w_unk) if val == VAL_UNK else step
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return float("inf")

rng = np.random.default_rng(7)
W = 0.3
mism = 0; tried = 0
for t in range(30):
    dim = [9, 9, 6]
    m = blank(dim)
    r = rng.random(dim)
    flat = m.cmap.reshape(dim, order="F")
    flat[r < 0.15] = VAL_OCC
    flat[(r >= 0.15) & (r < 0.55)] = VAL_UNK   # ~40% unknown
    m.cmap = flat.reshape(-1, order="F")
    s, g = (0, 0, 0), (8, 8, 5)
    m.cmap[m.lin_index(*s)] = VAL_FREE; m.cmap[m.lin_index(*g)] = VAL_FREE
    dcost = dijkstra(m, s, g, W)
    if dcost == float("inf"):
        continue
    tried += 1
    gs = GraphSearch(m, eps=1.0, global_planner="astar", w_unknown=W, heat_weight=0.0)
    ok = gs.plan(np.array(s), np.array(g), 0.0, np.zeros(3), max_expand=300000, timeout_ms=4000)
    acost = gs._hm[gs._coord_to_id(*g)].g if ok else float("inf")
    if not (ok and abs(acost - dcost) < 1e-6):
        mism += 1
        print(f"  mismatch t={t}: A*={acost:.4f} Dijkstra={dcost:.4f}")
check(f"B: A* cost == Dijkstra with unknown+w_unknown ({tried} trials)", mism == 0)

# ======================================================================
# C. A* on a REAL read_map output (obstacles + heat + unknown together)
# ======================================================================
print("\n" + "=" * 64)
print("C. A* on real read_map map (heat + obstacles + unknown)")
print("=" * 64)
m = VoxelMapUtil(res=0.3)
m.use_soft_cost_obstacles = False          # OCC hard-blocks for a clean assertion
m.static_heat_rmax_m = 2.0                  # wide halo so heat has room to steer
m.static_heat_default_radius_m = 2.0
# a pillar (point-cloud column) near the straight line start->goal
pillar = np.array([[1.0, 0.0, z] for z in np.linspace(0.4, 2.4, 30)]
                  + [[1.0 + dx, dy, z] for dx in (-0.1, 0.1) for dy in (-0.1, 0.1)
                     for z in np.linspace(0.4, 2.4, 10)])
m.read_map(30, 22, 10, np.array([0.0, 0.0, 1.2]), pillar, 0.0, 3.0, 0.3, [], [], 0.0)
n_unk = int(np.sum(m.cmap == VAL_UNK)); n_occ = int(np.sum(m.cmap == VAL_OCC))
print(f"real map: occ={n_occ} unknown={n_unk} (interior is mostly UNKNOWN, as in real use)")
sw, gw = np.array([-2.5, 0.0, 1.2]), np.array([4.0, 0.0, 1.2])  # straight line grazes the pillar

def run_planner(heat_w):
    p = HGPPlanner("astar_heat" if heat_w > 0 else "astar", False, 5, 10, 50, 4000,
                   w_unknown=0.0, w_align=0, decay_len_cells=20, w_side=0,
                   los_cells=0, min_len=0.3, min_turn=10.0,
                   heat_weight=heat_w, obstacle_soft_cost=0.0)
    p.set_map_util(m); p.set_max_expand(400000)
    ok, _ = p.plan(sw, np.zeros(3), gw, 0.0, eps=1.0)
    return ok, p.get_path()

def heat_exposure(path):
    tot = 0.0
    for i in range(len(path) - 1):
        a, b = np.asarray(path[i]), np.asarray(path[i + 1])
        n = max(1, int(np.linalg.norm(b - a) / (0.5 * m.res)))
        for k in range(n + 1):
            c = [int(v) for v in m.float_to_int(a + (b - a) * (k / n))]
            tot += m.get_heat(*c)
    return tot

def crosses_occ(path):
    return not all(seg_free(m, path[i], path[i + 1]) for i in range(len(path) - 1))

ok_off, path_off = run_planner(0.0)
ok_on, path_on = run_planner(40.0)
exp_off = heat_exposure(path_off) if ok_off else float("inf")
exp_on = heat_exposure(path_on) if ok_on else float("inf")
print(f"heat exposure: off={exp_off:.2f}  on={exp_on:.2f}")
check("C: A* runs on real (mostly-unknown) map & reaches goal", ok_on and ok_off)
check("C: path never crosses an occupied cell (heat-on & heat-off)",
      (not crosses_occ(path_on)) and (not crosses_occ(path_off)))
check("C: heat steers the path away (lower heat exposure with heat on)",
      exp_on < exp_off, f"on={exp_on:.2f} < off={exp_off:.2f}")

# ======================================================================
# D. dynamic prediction — heat follows where the obstacle WILL be
# ======================================================================
print("\n" + "=" * 64)
print("D. dynamic prediction: heat follows the predicted future path")
print("=" * 64)
m = VoxelMapUtil(res=0.3)
horizon = 2.0
times = np.linspace(0.0, horizon, 8)
# obstacle at x=0 now, predicted to move to x=+2 over the horizon (+x)
cur = np.array([0.0, 0.0, 1.2])
samples = [[np.array([2.0 * (tt / horizon), 0.0, 1.2]) for tt in times]]
m.dyn_pred_times = times
m.dyn_pred_samples = samples
m.read_map(30, 22, 10, np.array([0.0, 0.0, 1.2]), np.zeros((0, 3)),
           0.0, 3.0, 0.3, [cur], [np.array([0.2, 0.2, 0.2])], horizon)

def H(p):
    return m.get_heat(*[int(v) for v in m.float_to_int(np.asarray(p, float))])

h_now = H([0.0, 0.0, 1.2])      # current position
h_ahead = H([1.4, 0.0, 1.2])    # on the predicted future path (+x)
h_behind = H([-1.4, 0.0, 1.2])  # behind — obstacle never goes here
h_side = H([0.0, 1.4, 1.2])     # lateral — also never visited
print(f"heat: now={h_now:.3f}  ahead(future path)={h_ahead:.3f}  "
      f"behind={h_behind:.3f}  side={h_side:.3f}")
check("D: heat high at current obstacle position", h_now > 0.0)
check("D: heat extends ALONG the predicted future path (ahead > behind)",
      h_ahead > h_behind + 1e-6, f"ahead={h_ahead:.3f} behind={h_behind:.3f}")
check("D: heat does NOT bleed where obstacle never goes (behind ~ side ~ low)",
      h_behind < h_ahead and h_side < h_ahead, f"behind={h_behind:.3f} side={h_side:.3f}")

# ---- summary ----
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print("\n" + "=" * 64)
print(f"SUMMARY: {n_pass} passed, {n_fail} failed (of {len(results)})")
if n_fail:
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL: {name}  {detail}")
print("=" * 64)
