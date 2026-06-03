"""Golden-data generator for VoxelMapUtil (voxel_map.py) C++ port verification.

不许欺骗、必须还原:这里用真 Python VoxelMapUtil 跑出「输入 -> 输出」金标准,
C++ 移植后读同样输入、算出来必须逐项对上(容差内),对不上就不算还原。

Each case feeds read_map(...) with a small set of box obstacles (obst_pos/obst_bbox
= half-extents) + params, then dumps:
  DIM / ORIGIN / RES,
  OCC at a set of sample cells (occupancy int value),
  HEAT at a set of sample cells (float heat),
  int_to_float / float_to_int round-trips at sample points.

dump 到 cpp/golden/voxel_map_cases.txt(纯文本,C++ 无需 JSON 依赖即可解析)。
跑: python cpp/golden/gen_voxel_map_golden.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from sando_py.hgp.voxel_map import VoxelMapUtil  # noqa: E402

rng = np.random.default_rng(20260603)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


def fmt_i(a):
    return " ".join(str(int(x)) for x in np.asarray(a).reshape(-1))


lines = []


# Full list of knobs the C++ side must mirror (name -> default), so the golden
# can carry the exact configuration used and the C++ test rebuilds identically.
KNOB_BOOL = [
    "use_heat_map", "dynamic_heat_enabled", "dynamic_as_occupied_current",
    "dynamic_as_occupied_future", "static_heat_enabled", "static_heat_boundary_only",
    "static_heat_apply_on_unknown", "static_heat_exclude_dynamic",
    "use_soft_cost_obstacles",
]
KNOB_INT = ["heat_p", "heat_q", "heat_num_samples", "static_heat_p"]
KNOB_FLOAT = [
    "heat_alpha0", "heat_alpha1", "heat_tau_ratio", "heat_gamma", "heat_Hmax",
    "dyn_base_inflation_m", "dyn_heat_tube_radius_m", "obst_max_vel",
    "static_heat_alpha", "static_heat_Hmax", "static_heat_rmax_m",
    "static_heat_default_radius_m", "obstacle_soft_cost",
]


def emit_case(vm, cfg, obst_pos, obst_bbox, cloud_occ):
    """Run read_map with cfg dict of read_map params + dumped knobs, then dump outputs."""
    # apply knob overrides BEFORE read_map (heat composition reads them)
    for k, v in cfg.get("knobs", {}).items():
        setattr(vm, k, v)

    vm.read_map(
        cfg["cells_x"], cfg["cells_y"], cfg["cells_z"],
        np.asarray(cfg["center_map"], dtype=np.float64),
        np.asarray(cloud_occ, dtype=np.float64) if len(cloud_occ) else np.zeros((0, 3)),
        cfg["z_ground"], cfg["z_max"], cfg["inflation"],
        [np.asarray(p, dtype=np.float64) for p in obst_pos],
        [np.asarray(b, dtype=np.float64) for b in obst_bbox],
        cfg["traj_max_time"],
    )

    dimX, dimY, dimZ = int(vm.dim[0]), int(vm.dim[1]), int(vm.dim[2])

    # ---- sample cells: deterministic grid spread + random cells (in-bounds) ----
    sample_cells = []
    # corners + center + some fractional positions
    fracs = [0.0, 0.25, 0.5, 0.75, 0.99]
    for fx in fracs:
        for fy in (0.1, 0.5, 0.9):
            for fz in (0.0, 0.5, 0.99):
                x = min(int(fx * dimX), dimX - 1)
                y = min(int(fy * dimY), dimY - 1)
                z = min(int(fz * dimZ), dimZ - 1)
                sample_cells.append((x, y, z))
    # random cells
    for _ in range(60):
        x = int(rng.integers(0, dimX))
        y = int(rng.integers(0, dimY))
        z = int(rng.integers(0, dimZ))
        sample_cells.append((x, y, z))
    # cells near each obstacle center
    for c in obst_pos:
        ci = vm.float_to_int(np.asarray(c, dtype=np.float64))
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                x = int(ci[0] + dx)
                y = int(ci[1] + dy)
                z = int(ci[2])
                if vm.in_bounds(x, y, z):
                    sample_cells.append((x, y, z))
    # dedup, keep in-bounds
    seen = set()
    uniq = []
    for c in sample_cells:
        if not vm.in_bounds(*c):
            continue
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    sample_cells = uniq

    occ_vals = [int(vm.cmap[vm.lin_index(*c)]) for c in sample_cells]
    heat_vals = [float(vm.get_heat(*c)) for c in sample_cells]

    # ---- round-trip sample points (world) for float_to_int / int_to_float ----
    # int_to_float on sample cells, and float_to_int on world points
    rt_cells = sample_cells[:25]
    i2f = [vm.int_to_float(np.array(c, dtype=np.int64)) for c in rt_cells]
    # float_to_int of arbitrary world points (origin-relative + random)
    rt_points = []
    for _ in range(25):
        wp = vm.origin + rng.uniform(-1.0, 1.0, 3) + rng.uniform(0, 5, 3)
        rt_points.append(wp)
    # also include obstacle centers as test points
    for c in obst_pos:
        rt_points.append(np.asarray(c, dtype=np.float64))
    f2i = [vm.float_to_int(p) for p in rt_points]

    lines.append("CASE")
    # ---- INPUTS so the C++ side rebuilds the map identically ----
    lines.append(f"RES {repr(float(vm.res))}")
    lines.append("PARAMS " + fmt([
        cfg["cells_x"], cfg["cells_y"], cfg["cells_z"],
        cfg["center_map"][0], cfg["center_map"][1], cfg["center_map"][2],
        cfg["z_ground"], cfg["z_max"], cfg["inflation"], cfg["traj_max_time"],
    ]))
    lines.append(f"NOBST {len(obst_pos)}")
    if obst_pos:
        lines.append("OBSTPOS " + fmt([v for c in obst_pos for v in c]))
        lines.append("OBSTBBOX " + fmt([v for b in obst_bbox for v in b]))
    else:
        lines.append("OBSTPOS")
        lines.append("OBSTBBOX")
    lines.append(f"NCLOUD {len(cloud_occ)}")
    if len(cloud_occ):
        lines.append("CLOUD " + fmt([v for c in cloud_occ for v in c]))
    else:
        lines.append("CLOUD")
    # knobs (the actual values on vm after overrides)
    lines.append("KBOOL " + " ".join(str(int(bool(getattr(vm, k)))) for k in KNOB_BOOL))
    lines.append("KINT " + " ".join(str(int(getattr(vm, k))) for k in KNOB_INT))
    lines.append("KFLOAT " + fmt([float(getattr(vm, k)) for k in KNOB_FLOAT]))
    # predicted times / samples
    has_times = (vm.dyn_pred_times is not None and len(vm.dyn_pred_times) >= 1)
    if has_times:
        t = np.asarray(vm.dyn_pred_times, dtype=float)
        lines.append(f"PREDTIMES {len(t)} " + fmt(t))
    else:
        lines.append("PREDTIMES 0")
    if vm.dyn_pred_samples is not None:
        # flatten as: for each obstacle k: count then count*3 floats
        flat = []
        meta = [len(vm.dyn_pred_samples)]
        for sk in vm.dyn_pred_samples:
            meta.append(len(sk))
            for s in sk:
                flat.extend([float(s[0]), float(s[1]), float(s[2])])
        lines.append("PREDSAMP " + " ".join(str(int(x)) for x in meta) +
                     ((" " + fmt(flat)) if flat else ""))
    else:
        lines.append("PREDSAMP -1")
    # ---- OUTPUTS ----
    lines.append(f"DIM {fmt_i([dimX, dimY, dimZ])}")
    lines.append(f"ORIGIN {fmt(vm.origin)}")
    lines.append(f"NCELL {len(sample_cells)}")
    lines.append(f"CELLS {fmt_i([v for c in sample_cells for v in c])}")
    lines.append(f"OCC {fmt_i(occ_vals)}")
    lines.append(f"HEAT {fmt(heat_vals)}")
    lines.append(f"NRT {len(rt_cells)}")
    lines.append(f"RTCELLS {fmt_i([v for c in rt_cells for v in c])}")
    lines.append(f"I2F {fmt([v for w in i2f for v in w])}")
    lines.append(f"NF2I {len(rt_points)}")
    lines.append(f"RTPOINTS {fmt([v for p in rt_points for v in p])}")
    lines.append(f"F2I {fmt_i([v for q in f2i for v in q])}")
    lines.append("END")


# =====================================================================
# Case configurations: diverse + random + edge cases
# =====================================================================
def base_knobs():
    return {}


# Case 1: simple — one box obstacle, default heat knobs
emit_case(
    VoxelMapUtil(res=0.3),
    dict(cells_x=20, cells_y=20, cells_z=8, center_map=[0.0, 0.0, 1.5],
         z_ground=0.0, z_max=3.0, inflation=0.3, traj_max_time=2.0, knobs={}),
    obst_pos=[[0.5, 0.5, 1.5]],
    obst_bbox=[[0.3, 0.3, 0.4]],
    cloud_occ=[],
)

# Case 2: several boxes + a point cloud, finer res, future cone on
cloud2 = []
for _ in range(120):
    cloud2.append([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.2, 2.8)])
emit_case(
    VoxelMapUtil(res=0.2),
    dict(cells_x=30, cells_y=30, cells_z=14, center_map=[0.3, -0.4, 1.4],
         z_ground=0.1, z_max=2.9, inflation=0.25, traj_max_time=3.0,
         knobs={"dynamic_as_occupied_future": True, "heat_gamma": 0.15}),
    obst_pos=[[-0.8, 0.6, 1.2], [1.0, -0.7, 1.6], [0.0, 0.0, 1.0]],
    obst_bbox=[[0.25, 0.25, 0.3], [0.4, 0.2, 0.5], [0.15, 0.35, 0.2]],
    cloud_occ=cloud2,
)

# Case 3: static heat stress — bigger rmax, larger default radius, p=3
emit_case(
    VoxelMapUtil(res=0.25),
    dict(cells_x=24, cells_y=24, cells_z=10, center_map=[1.0, 1.0, 1.2],
         z_ground=0.0, z_max=2.5, inflation=0.2, traj_max_time=1.5,
         knobs={"static_heat_rmax_m": 1.5, "static_heat_default_radius_m": 1.0,
                "static_heat_p": 3, "static_heat_alpha": 2.0,
                "static_heat_Hmax": 8.0}),
    obst_pos=[[1.2, 0.8, 1.2], [0.5, 1.3, 1.1]],
    obst_bbox=[[0.3, 0.3, 0.4], [0.2, 0.2, 0.3]],
    cloud_occ=[],
)

# Case 4: dynamic heat off, static only; boundary_only off
emit_case(
    VoxelMapUtil(res=0.3),
    dict(cells_x=18, cells_y=18, cells_z=8, center_map=[0.0, 0.0, 1.0],
         z_ground=0.0, z_max=2.0, inflation=0.3, traj_max_time=2.0,
         knobs={"dynamic_heat_enabled": False,
                "static_heat_boundary_only": False,
                "static_heat_exclude_dynamic": False}),
    obst_pos=[[0.3, -0.3, 1.0]],
    obst_bbox=[[0.4, 0.4, 0.5]],
    cloud_occ=[],
)

# Case 5: static heat off, dynamic only; soft cost obstacles off; more samples
emit_case(
    VoxelMapUtil(res=0.2),
    dict(cells_x=26, cells_y=26, cells_z=12, center_map=[-0.5, 0.5, 1.3],
         z_ground=0.0, z_max=2.6, inflation=0.2, traj_max_time=4.0,
         knobs={"static_heat_enabled": False,
                "use_soft_cost_obstacles": False,
                "heat_num_samples": 25,
                "heat_p": 3, "heat_q": 1,
                "heat_alpha0": 0.5, "heat_alpha1": 1.5,
                "heat_Hmax": 3.0, "obst_max_vel": 0.8,
                "dyn_heat_tube_radius_m": 0.7}),
    obst_pos=[[-0.5, 0.5, 1.3], [0.4, -0.2, 1.1]],
    obst_bbox=[[0.2, 0.3, 0.25], [0.3, 0.3, 0.3]],
    cloud_occ=[],
)

# Case 6: explicit dyn_pred_times + dyn_pred_samples (moving obstacles)
def case6():
    vm = VoxelMapUtil(res=0.25)
    # 2 obstacles, each with a predicted trajectory (5 time samples)
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    obst_pos = [[-0.6, 0.0, 1.2], [0.7, 0.5, 1.3]]
    obst_bbox = [[0.25, 0.25, 0.3], [0.2, 0.3, 0.25]]
    # predicted sample centers per obstacle
    samples = [
        [np.array([-0.6 + 0.3 * t, 0.1 * t, 1.2]) for t in times],
        [np.array([0.7 - 0.2 * t, 0.5 + 0.15 * t, 1.3]) for t in times],
    ]
    vm.dyn_pred_times = times
    vm.dyn_pred_samples = samples
    emit_case(
        vm,
        dict(cells_x=22, cells_y=22, cells_z=10, center_map=[0.0, 0.2, 1.3],
             z_ground=0.0, z_max=2.6, inflation=0.25, traj_max_time=2.0,
             knobs={"heat_gamma": 0.2, "heat_tau_ratio": 0.4}),
        obst_pos=obst_pos, obst_bbox=obst_bbox, cloud_occ=[],
    )


case6()

# Case 7: edge — z clamp hits ground (center near ground), tiny inflation
emit_case(
    VoxelMapUtil(res=0.3),
    dict(cells_x=16, cells_y=16, cells_z=10, center_map=[0.0, 0.0, 0.4],
         z_ground=0.0, z_max=3.0, inflation=0.05, traj_max_time=1.0, knobs={}),
    obst_pos=[[0.2, 0.1, 0.3]],
    obst_bbox=[[0.2, 0.2, 0.2]],
    cloud_occ=[],
)

# Case 8: edge — z clamp hits top, large inflation pad
emit_case(
    VoxelMapUtil(res=0.3),
    dict(cells_x=14, cells_y=14, cells_z=12, center_map=[0.0, 0.0, 2.7],
         z_ground=0.0, z_max=3.0, inflation=0.6, traj_max_time=2.0, knobs={}),
    obst_pos=[[0.0, 0.0, 2.6]],
    obst_bbox=[[0.3, 0.3, 0.3]],
    cloud_occ=[],
)

# Case 9: no obstacles at all (only walls + UNKNOWN), heat fields still allocated
emit_case(
    VoxelMapUtil(res=0.3),
    dict(cells_x=12, cells_y=12, cells_z=6, center_map=[0.0, 0.0, 1.0],
         z_ground=0.0, z_max=2.0, inflation=0.3, traj_max_time=2.0, knobs={}),
    obst_pos=[],
    obst_bbox=[],
    cloud_occ=[],
)

# Case 10: random — randomized params + several random box obstacles + cloud
for trial in range(4):
    res = float(rng.uniform(0.18, 0.35))
    cx = int(rng.integers(16, 32))
    cy = int(rng.integers(16, 32))
    cz = int(rng.integers(6, 16))
    cmap_center = [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)),
                   float(rng.uniform(1.0, 1.8))]
    infl = float(rng.uniform(0.1, 0.4))
    tmax = float(rng.uniform(1.0, 4.0))
    nobst = int(rng.integers(1, 4))
    op = []
    ob = []
    for _ in range(nobst):
        op.append([float(rng.uniform(-1.2, 1.2)), float(rng.uniform(-1.2, 1.2)),
                   float(rng.uniform(0.8, 2.0))])
        ob.append([float(rng.uniform(0.1, 0.5)), float(rng.uniform(0.1, 0.5)),
                   float(rng.uniform(0.1, 0.5))])
    ncloud = int(rng.integers(0, 100))
    cl = []
    for _ in range(ncloud):
        cl.append([float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)),
                   float(rng.uniform(0.1, 2.8))])
    knobs = {
        "heat_alpha0": float(rng.uniform(0.1, 0.6)),
        "heat_alpha1": float(rng.uniform(0.5, 1.5)),
        "heat_p": int(rng.integers(1, 4)),
        "heat_q": int(rng.integers(1, 4)),
        "heat_gamma": float(rng.uniform(0.0, 0.3)),
        "heat_Hmax": float(rng.uniform(1.0, 4.0)),
        "heat_num_samples": int(rng.integers(5, 20)),
        "obst_max_vel": float(rng.uniform(0.3, 1.0)),
        "dyn_heat_tube_radius_m": float(rng.uniform(0.3, 0.8)),
        "static_heat_rmax_m": float(rng.uniform(0.6, 1.4)),
        "static_heat_default_radius_m": float(rng.uniform(0.4, 1.0)),
        "static_heat_p": int(rng.integers(1, 4)),
        "static_heat_alpha": float(rng.uniform(0.5, 2.0)),
        "static_heat_Hmax": float(rng.uniform(3.0, 8.0)),
        "dynamic_as_occupied_future": bool(rng.integers(0, 2)),
        "static_heat_boundary_only": bool(rng.integers(0, 2)),
        "static_heat_exclude_dynamic": bool(rng.integers(0, 2)),
        "static_heat_apply_on_unknown": bool(rng.integers(0, 2)),
    }
    emit_case(
        VoxelMapUtil(res=res),
        dict(cells_x=cx, cells_y=cy, cells_z=cz, center_map=cmap_center,
             z_ground=0.0, z_max=3.0, inflation=infl, traj_max_time=tmax, knobs=knobs),
        obst_pos=op, obst_bbox=ob, cloud_occ=cl,
    )

out = os.path.join(os.path.dirname(__file__), "voxel_map_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len([l for l in lines if l == 'CASE'])} cases)")
