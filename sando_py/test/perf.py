"""Performance — the planner must return in acceptable time on big/dense/crowded
inputs (catch pathological blow-ups & hangs, not micro-benchmark).

Run:  python3 test/perf.py
"""
import os
import sys
import time

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.hgp_planner import HGPPlanner                          # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC       # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

BUDGET = 30.0   # generous upper bound — we only want to catch hangs / blow-ups

# ---- P1: large sparse map, A* corner-to-corner ----
dim = [80, 80, 16]   # ~102k cells
m = VoxelMapUtil(res=0.3); m.dim = np.array(dim); m.origin = np.zeros(3)
rng = np.random.default_rng(0)
occ = rng.random(dim) < 0.08
m.cmap = np.where(occ.reshape(-1, order="F"), VAL_OCC, VAL_FREE).astype(np.int8)
m.heat = np.zeros(int(np.prod(dim)), np.float32)
m.use_soft_cost_obstacles = False; m.initialized = True
m.cmap[m.lin_index(0, 0, 0)] = VAL_FREE
m.cmap[m.lin_index(dim[0] - 1, dim[1] - 1, dim[2] - 1)] = VAL_FREE
p = HGPPlanner("astar", False, 5, 10, 50, int(BUDGET * 1000), 0, 0, 20, 0,
               los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
p.set_map_util(m); p.set_max_expand(5000000)
t0 = time.perf_counter()
ok, _ = p.plan(m.int_to_float(np.array([0, 0, 0])),
               np.zeros(3), m.int_to_float(np.array(dim) - 1), 0.0, eps=1.0)
dt = time.perf_counter() - t0
print(f"P1 large {dim} (~{int(np.prod(dim)/1000)}k cells): {dt:.2f}s, ok={ok}")
check("P1: A* on large map returns within budget", dt < BUDGET, f"{dt:.2f}s")

# ---- P2: read_map with many dynamic obstacles (heat composition) ----
m = VoxelMapUtil(res=0.3); m.use_soft_cost_obstacles = False
rng = np.random.default_rng(1)
obst_pos = [np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), 1.2]) for _ in range(20)]
obst_bbox = [np.array([0.3, 0.3, 0.5]) for _ in obst_pos]
t0 = time.perf_counter()
m.read_map(40, 40, 10, np.array([0.0, 0.0, 1.2]), np.zeros((0, 3)),
           0.0, 3.0, 0.3, obst_pos, obst_bbox, 1.5)
dt = time.perf_counter() - t0
print(f"P2 read_map + 20 dynamic obstacles (heat): {dt:.2f}s")
check("P2: read_map with many dynamic obstacles returns within budget", dt < BUDGET, f"{dt:.2f}s")

# ---- P3: dense obstacles, A* must still return (likely no path) without hanging ----
dim = [40, 40, 10]
m = VoxelMapUtil(res=0.3); m.dim = np.array(dim); m.origin = np.zeros(3)
rng = np.random.default_rng(2)
occ = rng.random(dim) < 0.45   # very dense
m.cmap = np.where(occ.reshape(-1, order="F"), VAL_OCC, VAL_FREE).astype(np.int8)
m.heat = np.zeros(int(np.prod(dim)), np.float32)
m.use_soft_cost_obstacles = False; m.initialized = True
m.cmap[m.lin_index(0, 0, 0)] = VAL_FREE
m.cmap[m.lin_index(dim[0] - 1, dim[1] - 1, dim[2] - 1)] = VAL_FREE
p = HGPPlanner("astar", False, 5, 10, 50, int(BUDGET * 1000), 0, 0, 20, 0,
               los_cells=0, min_len=0.5, min_turn=10.0, heat_weight=0, obstacle_soft_cost=0)
p.set_map_util(m); p.set_max_expand(5000000)
t0 = time.perf_counter()
ok, _ = p.plan(m.int_to_float(np.array([0, 0, 0])),
               np.zeros(3), m.int_to_float(np.array(dim) - 1), 0.0, eps=1.0)
dt = time.perf_counter() - t0
print(f"P3 dense 45% obstacles {dim}: {dt:.2f}s, ok={ok}")
check("P3: A* on dense map returns within budget (no hang)", dt < BUDGET, f"{dt:.2f}s")

n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
print("(timings are machine-dependent; budget only catches hangs/blow-ups)")
print("=" * 60)
