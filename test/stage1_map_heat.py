"""Stage 1 verification: voxel_map.py (occupancy + heat).

Standalone — loads voxel_map.py directly, no ROS / gurobi / package __init__.
Run:  python3 test/stage1_map_heat.py
Checks:
  0. coordinate round-trip int_to_float(float_to_int(p)) ~ p
  1. empty map = UNKNOWN inside + occupied y-walls (documents behavior)
  2. static point-cloud obstacle -> occupied at its location, free far away
  3. write/read convention: floor (rasterize) vs -0.5 (query) consistency
  4. dynamic obstacle AABB -> occupied + dyn mask
  5. heat: high near obstacle, decreasing with distance, ~0 far, bounded
"""
import importlib.util
import os
import numpy as np

VM_PATH = os.path.join(os.path.dirname(__file__), "..", "sando_py", "hgp", "voxel_map.py")
spec = importlib.util.spec_from_file_location("voxel_map_standalone", os.path.abspath(VM_PATH))
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)
VoxelMapUtil = vm.VoxelMapUtil
VAL_FREE, VAL_OCC, VAL_UNK = vm.VAL_FREE, vm.VAL_OCC, vm.VAL_UNK

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def occ(m, p):
    c = m.float_to_int(np.asarray(p, float))
    return m.is_occupied(int(c[0]), int(c[1]), int(c[2]))
def heat_at(m, p):
    c = m.float_to_int(np.asarray(p, float))
    return m.get_heat(int(c[0]), int(c[1]), int(c[2]))

print("=" * 60)
print("STAGE 1 — voxel_map occupancy + heat verification")
print("=" * 60)

# ---- Test 0: coordinate round-trip ----
m0 = VoxelMapUtil(res=0.3)
m0.dim = np.array([20, 20, 20]); m0.origin = np.array([-3.0, -3.0, 0.0])
for p in (np.array([0.0, 0.0, 1.0]), np.array([0.45, -0.3, 0.6]), np.array([1.0, 1.0, 0.9])):
    back = m0.int_to_float(m0.float_to_int(p))
    check(f"roundtrip {p.tolist()}", np.all(np.abs(back - p) <= m0.res + 1e-9),
          f"back={np.round(back,3).tolist()}")

# ---- Build a real map ----
res = 0.3
center = np.array([0.0, 0.0, 1.0])
cells_x = cells_y = 40; cells_z = 10
z_ground, z_max, inflation = 0.0, 3.0, 0.3

# static obstacle: a small point-cloud cluster at (2,0,1)
blob = np.array([[2.0 + dx, 0.0 + dy, 1.0 + dz]
                 for dx in (-0.05, 0.0, 0.05)
                 for dy in (-0.05, 0.0, 0.05)
                 for dz in (-0.05, 0.0, 0.05)])
# dynamic obstacle at (-2,0,1)
dyn_pos = [np.array([-2.0, 0.0, 1.0])]
dyn_bbox = [np.array([0.3, 0.3, 0.3])]

m = VoxelMapUtil(res=res)
m.read_map(cells_x, cells_y, cells_z, center, blob, z_ground, z_max,
           inflation, dyn_pos, dyn_bbox, traj_max_time=1.0)

dimX, dimY, dimZ = (int(v) for v in m.dim)
print(f"\nmap dim = {[dimX, dimY, dimZ]}, origin = {np.round(m.origin,3).tolist()}, "
      f"#occ = {int(np.sum(m.cmap == VAL_OCC))}, #unk = {int(np.sum(m.cmap == VAL_UNK))}\n")

# ---- Test 1: empty interior is UNKNOWN, y-walls occupied ----
center_cell_unknown = (m.cmap[m.lin_index(dimX // 2, dimY // 2, dimZ // 2)] == VAL_UNK)
# pick a clearly-interior empty point far from both obstacles
check("interior empty cell is UNKNOWN (not free/occ)", center_cell_unknown,
      f"val={int(m.cmap[m.lin_index(dimX//2, dimY//2, dimZ//2)])}")
y_lo_world = m.origin[1] + 0.5 * res
y_hi_world = m.origin[1] + (dimY - 0.5) * res
check("y-min wall occupied", occ(m, [0.0, y_lo_world, 1.0]))
check("y-max wall occupied", occ(m, [0.0, y_hi_world, 1.0]))

# ---- Test 2: static obstacle occupied, far cell not occupied ----
check("static obstacle (2,0,1) occupied", occ(m, [2.0, 0.0, 1.0]))
check("far cell (0,0,1) NOT occupied", not occ(m, [0.0, 0.0, 1.0]))
# inflation: neighbor one cell away in +x should also be occupied (inflation=res)
check("static obstacle inflated by ~1 cell", occ(m, [2.0 + res, 0.0, 1.0]))

# ---- Test 3: write (floor) vs query (-0.5) convention ----
p = np.array([2.0, 0.0, 1.0])
raster_cell = np.floor((p - m.origin) / res).astype(int)
query_cell = m.float_to_int(p)
same = np.array_equal(raster_cell, query_cell)
# NOT a pass/fail: C++ also writes with floor() and reads with floatToInt(-0.5).
# Faithful to C++; the ~1-cell offset is masked by inflation >= 1 cell.
print(f"[NOTE] write(floor) vs query(-0.5): raster={raster_cell.tolist()} "
      f"query={query_cell.tolist()} -> "
      f"{'same' if same else 'OFF-BY-ONE (faithful to C++, masked by inflation>=1 cell)'}")

# ---- Test 4: dynamic obstacle AABB occupied + dyn mask ----
check("dynamic obstacle (-2,0,1) occupied", occ(m, [-2.0, 0.0, 1.0]))
dc = m.float_to_int(np.array([-2.0, 0.0, 1.0]))
lin = m.lin_index(int(dc[0]), int(dc[1]), int(dc[2]))
check("dynamic cell flagged in dyn_occ_mask", bool(m.dyn_occ_mask[lin]))

# ---- Test 5: heat behavior ----
h_center = heat_at(m, [-2.0, 0.0, 1.0])      # on the dynamic obstacle
h_near   = heat_at(m, [-1.4, 0.0, 1.0])      # 0.6 m away
h_far    = heat_at(m, [0.0, 0.0, 1.0])       # 2 m away (between obstacles)
print(f"\nheat: center={h_center:.3f}  near(0.6m)={h_near:.3f}  far(2m)={h_far:.3f}")
check("heat at obstacle > 0", h_center > 0.0, f"{h_center:.3f}")
check("heat decreases with distance (center > near > far)",
      h_center > h_near > h_far - 1e-9, f"{h_center:.3f} > {h_near:.3f} > {h_far:.3f}")
check("heat ~0 far from any obstacle", h_far < 1e-6, f"far={h_far:.4f}")
hmax = float(m.heat.max()) if m.heat.size else 0.0
cap = max(m.heat_Hmax, m.static_heat_Hmax)
check("heat bounded by Hmax", hmax <= cap + 1e-6, f"max={hmax:.3f} cap={cap:.3f}")
# static heat around the point-cloud obstacle (should exist near it, not on it only)
h_static_near = heat_at(m, [2.0 + 0.5, 0.0, 1.0])
check("static heat present near point-cloud obstacle", h_static_near > 0.0,
      f"{h_static_near:.3f}")

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
