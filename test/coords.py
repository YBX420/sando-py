"""Coordinate / unit consistency: world <-> voxel <-> path, across origin,
resolution, floor convention, and negative coordinates.

Run:  python3 test/coords.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.graph_search import GraphSearch                 # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC  # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def mk(dim, res, origin):
    m = VoxelMapUtil(res=res)
    m.dim = np.array(dim, dtype=np.int64)
    m.origin = np.array(origin, dtype=np.float64)
    m.cmap = np.full(int(np.prod(dim)), VAL_FREE, dtype=np.int8)
    m.heat = np.zeros(int(np.prod(dim)), dtype=np.float32)
    m.use_soft_cost_obstacles = False
    m.initialized = True
    return m

print("=" * 60)
print("COORDINATE / UNIT CONSISTENCY")
print("=" * 60)

configs = [
    ("unit res, zero origin", [10, 10, 10], 1.0, [0.0, 0.0, 0.0]),
    ("fine res, neg origin", [20, 18, 12], 0.25, [-3.3, 1.7, 0.5]),
    ("coarse res, neg origin", [12, 12, 8], 0.5, [-5.0, -5.0, -1.0]),
]
for name, dim, res, origin in configs:
    m = mk(dim, res, origin)
    # X1 int -> float -> int is exact (cell center floors back to its own cell)
    ok_rt = True
    for _ in range(200):
        cell = np.array([np.random.randint(0, dim[0]), np.random.randint(0, dim[1]),
                         np.random.randint(0, dim[2])])
        if not np.array_equal(m.float_to_int(m.int_to_float(cell)), cell):
            ok_rt = False
            break
    check(f"[{name}] int->float->int exact", ok_rt)
    # X2 float -> int -> float within res
    ok_w = True
    for _ in range(200):
        p = m.origin + np.random.rand(3) * (np.array(dim) * res)
        back = m.int_to_float(m.float_to_int(p))
        if np.any(np.abs(back - p) > res + 1e-9):
            ok_w = False
            break
    check(f"[{name}] float->int->float within res", ok_w)
    # X3 floor convention
    ok_floor = True
    for _ in range(200):
        p = m.origin + (np.random.rand(3) * 2 - 0.5) * (np.array(dim) * res)
        if not np.array_equal(m.float_to_int(p),
                              np.floor((p - m.origin) / res).astype(np.int64)):
            ok_floor = False
            break
    check(f"[{name}] float_to_int == floor((p-origin)/res)", ok_floor)
    # X7 lin_index unravel round-trip
    ok_lin = True
    for _ in range(200):
        x, y, z = (np.random.randint(0, dim[0]), np.random.randint(0, dim[1]),
                   np.random.randint(0, dim[2]))
        lin = m.lin_index(x, y, z)
        rx = lin % dim[0]
        ry = (lin // dim[0]) % dim[1]
        rz = lin // (dim[0] * dim[1])
        if (rx, ry, rz) != (x, y, z):
            ok_lin = False
            break
    check(f"[{name}] lin_index unravel round-trip", ok_lin)

# X6 path output consistency: raw path cells contiguous, world coords map back in-bounds
m = mk([16, 16, 8], 0.3, [-2.0, -2.0, 0.3])
m.cmap[m.lin_index(8, 5, 4)] = VAL_OCC
gs = GraphSearch(m, eps=1.0, global_planner="astar", heat_weight=0.0)
gs.plan(np.array([2, 2, 2]), np.array([13, 13, 6]), 0.0, np.zeros(3))
path = gs.get_path_world()
ints = [tuple(int(v) for v in m.float_to_int(p)) for p in path]
contiguous = all(max(abs(ints[i][k] - ints[i + 1][k]) for k in range(3)) <= 1
                 for i in range(len(ints) - 1))
inb = all(m.in_bounds(*c) for c in ints)
check("path world coords map back to in-bounds cells", inb and len(path) >= 2)
check("raw path cells are 26-connected contiguous (no teleport)", contiguous)

n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
if n_pass != len(results):
    for nm, ok, d in results:
        if not ok:
            print(f"  FAIL: {nm}  {d}")
print("=" * 60)
