"""Golden-data generator for VoxelMapUtil geometric query helpers (voxel_map.py)
C++ port verification (voxel_map_helpers.hpp).

不许欺骗、必须还原:用真 Python VoxelMapUtil 跑出「输入 -> 输出」金标准,C++ 移植后
读同样输入、用 read_map 重建同一张图,再对这些 helper 的结果逐项对上:
  set_free_voxel_and_surroundings  -> carve, then OCC at sample cells (int) + the n used
  find_closest_free_point          -> world point (3 floats) per sample point
  is_blocked (DDA)                 -> bool (0/1) per sample segment, val=100 and val=0
  line_of_sight_capsule            -> bool (0/1) per (segment, radius)
  cells_world(value)               -> count of cells (int) for value in {0,100,-1}
  has_non_free_neighbor            -> bool (0/1) per sample point

Each case rebuilds the SAME read_map(...) used by gen_voxel_map_golden.py (box
obstacles + params + knobs), then samples diverse + random + edge-case queries.

dump 到 cpp/golden/voxel_map_helpers_cases.txt(纯文本,C++ 无需 JSON 即可解析)。
跑: python cpp/golden/gen_voxel_map_helpers_golden.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.hgp.voxel_map import VoxelMapUtil  # noqa: E402

rng = np.random.default_rng(20260603)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


def fmt_i(a):
    return " ".join(str(int(x)) for x in np.asarray(a).reshape(-1))


lines = []

# Same knob lists as gen_voxel_map_golden.py so the C++ side rebuilds identically.
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


def build_vm(res, cfg, obst_pos, obst_bbox, cloud_occ):
    vm = VoxelMapUtil(res=res)
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
    return vm


def world_box(vm):
    """Return (lo, hi) world-space bounds of the map's cell-center range."""
    lo = vm.int_to_float(np.array([0, 0, 0], dtype=np.int64))
    hi = vm.int_to_float(np.array([vm.dim[0] - 1, vm.dim[1] - 1, vm.dim[2] - 1],
                                  dtype=np.int64))
    return lo, hi


def emit_case(res, cfg, obst_pos, obst_bbox, cloud_occ, precarve=None):
    # precarve: optional list of (center, d) applied via set_free_voxel_and_surroundings
    # BEFORE the read-only queries, so genuine FREE cells exist (exercises is_blocked
    # val=0 != val=100, and find_closest_free_point returning a real free center).
    precarve = precarve or []
    vm = build_vm(res, cfg, obst_pos, obst_bbox, cloud_occ)
    for (pc_c, pc_d) in precarve:
        vm.set_free_voxel_and_surroundings(np.asarray(pc_c, dtype=np.float64), float(pc_d))
    dimX, dimY, dimZ = int(vm.dim[0]), int(vm.dim[1]), int(vm.dim[2])
    lo, hi = world_box(vm)

    # ---------------------------------------------------------------
    # Sample world points: obstacle centers, walls, random in/near bounds,
    # plus some clearly-out-of-bounds points (closest-free-point edge cases).
    # ---------------------------------------------------------------
    sample_pts = []
    for c in obst_pos:
        sample_pts.append(np.asarray(c, dtype=np.float64))
        sample_pts.append(np.asarray(c, dtype=np.float64) + np.array([0.4, 0.0, 0.0]))
        sample_pts.append(np.asarray(c, dtype=np.float64) + np.array([0.0, -0.5, 0.3]))
    # map center-ish and corners
    center = vm.origin + 0.5 * (np.array([dimX, dimY, dimZ]) * vm.res)
    sample_pts.append(center)
    for fx in (0.1, 0.5, 0.9):
        for fy in (0.1, 0.5, 0.9):
            for fz in (0.2, 0.8):
                wp = vm.origin + np.array([fx * dimX, fy * dimY, fz * dimZ]) * vm.res
                sample_pts.append(wp)
    # random points within / slightly beyond bounds
    span = (hi - lo)
    for _ in range(40):
        t = rng.uniform(-0.15, 1.15, 3)
        sample_pts.append(lo + t * span)
    # a few far-out-of-bounds points (closest-free search may fail -> returns point)
    sample_pts.append(hi + np.array([10.0, 10.0, 10.0]))
    sample_pts.append(lo - np.array([8.0, 8.0, 8.0]))

    sample_pts = [np.asarray(p, dtype=np.float64) for p in sample_pts]

    # find_closest_free_point per sample point
    fcfp = [vm.find_closest_free_point(p) for p in sample_pts]
    # has_non_free_neighbor per sample point
    hnfn = [int(bool(vm.has_non_free_neighbor(p))) for p in sample_pts]

    # ---------------------------------------------------------------
    # Sample segments for is_blocked / line_of_sight_capsule.
    # Mix: through obstacles, along walls, free-space diagonals, degenerate (a==b),
    # axis-aligned, corner-grazing, partly-out-of-bounds.
    # ---------------------------------------------------------------
    segs = []  # list of (p1, p2)
    # obstacle-piercing segments (center to opposite free space)
    for c in obst_pos:
        c = np.asarray(c, dtype=np.float64)
        segs.append((c + np.array([-1.0, 0.0, 0.0]), c + np.array([1.0, 0.0, 0.0])))
        segs.append((c + np.array([0.0, -1.0, 0.0]), c + np.array([0.0, 1.0, 0.0])))
        segs.append((c + np.array([-0.8, -0.8, -0.3]), c + np.array([0.8, 0.8, 0.3])))
        segs.append((c, c + np.array([0.0, 0.0, 0.5])))  # short
    # corner-graze: diagonal that skims cell corners
    diag = lo + 0.5 * span
    segs.append((diag, diag + np.array([1.0, 1.0, 1.0])))
    segs.append((diag, diag + np.array([1.0, 1.0, 0.0])))
    segs.append((diag, diag + np.array([1.0, 0.0, 1.0])))
    # axis-aligned across the whole map (will hit y-walls)
    midz = vm.origin[2] + 0.5 * dimZ * vm.res
    segs.append((np.array([lo[0], lo[1], midz]), np.array([hi[0], lo[1], midz])))  # along y-min wall row
    segs.append((np.array([center[0], lo[1] - 1.0, midz]),
                 np.array([center[0], hi[1] + 1.0, midz])))  # crosses both y walls
    # degenerate a==b
    segs.append((center, center.copy()))
    # random segments in/near bounds
    for _ in range(30):
        p1 = lo + rng.uniform(-0.1, 1.1, 3) * span
        p2 = lo + rng.uniform(-0.1, 1.1, 3) * span
        segs.append((p1, p2))
    # partly out of bounds
    segs.append((lo - np.array([2.0, 2.0, 2.0]), center))
    segs.append((center, hi + np.array([3.0, 0.5, 1.0])))

    segs = [(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
            for a, b in segs]

    # is_blocked with val=100 (occupied) and val=0 (free-threshold: blocks on >=0,
    # i.e. anything not strictly negative/unknown). Two val regimes stress the code.
    blk100 = [int(bool(vm.is_blocked(a, b, 100))) for a, b in segs]
    blk0 = [int(bool(vm.is_blocked(a, b, 0))) for a, b in segs]

    # line_of_sight_capsule with several radii
    radii = [0, 1, 2]
    los = []  # flatten: for each seg, for each radius
    for a, b in segs:
        for rcell in radii:
            los.append(int(bool(vm.line_of_sight_capsule(a, b, rcell))))

    # cells_world counts (cheap, exact integer cross-check of iteration/order-independence)
    nfree = len(vm.cells_world(0))
    nocc = len(vm.cells_world(100))
    nunk = len(vm.cells_world(-1))

    # ---------------------------------------------------------------
    # set_free_voxel_and_surroundings: on a FRESH rebuild, carve around a couple
    # of centers with a few d values; dump n used + resulting OCC at sample cells.
    # ---------------------------------------------------------------
    carve_specs = []  # (center, d)
    if obst_pos:
        carve_specs.append((np.asarray(obst_pos[0], dtype=np.float64), 0.6))
    carve_specs.append((center.copy(), 0.45))
    carve_specs.append((vm.origin + np.array([dimX, dimY, dimZ]) * vm.res * 0.3, 1.0))

    carve_results = []  # list of (n, [sample cells], [occ after])
    for ccenter, d in carve_specs:
        vm2 = build_vm(res, cfg, obst_pos, obst_bbox, cloud_occ)
        n_used = int(round(d / vm2.res + 0.5))
        # sample cells in a window around the carve center to verify the cube was freed
        cx, cy, cz = vm2.float_to_int(ccenter)
        cells = []
        for dx in range(-(n_used + 1), n_used + 2, 1):
            for dy in (-(n_used + 1), 0, n_used + 1):
                for dz in (-(n_used + 1), 0, n_used + 1):
                    x, y, z = int(cx + dx), int(cy + dy), int(cz + dz)
                    if vm2.in_bounds(x, y, z):
                        cells.append((x, y, z))
        # dedup
        seen = set()
        uc = []
        for c in cells:
            if c in seen:
                continue
            seen.add(c)
            uc.append(c)
        cells = uc
        vm2.set_free_voxel_and_surroundings(ccenter, d)
        occ_after = [int(vm2.cmap[vm2.lin_index(*c)]) for c in cells]
        carve_results.append((ccenter, d, n_used, cells, occ_after))

    # ===============================================================
    # Emit
    # ===============================================================
    lines.append("CASE")
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
    lines.append("KBOOL " + " ".join(str(int(bool(getattr(vm, k)))) for k in KNOB_BOOL))
    lines.append("KINT " + " ".join(str(int(getattr(vm, k))) for k in KNOB_INT))
    lines.append("KFLOAT " + fmt([float(getattr(vm, k)) for k in KNOB_FLOAT]))
    has_times = (vm.dyn_pred_times is not None and len(vm.dyn_pred_times) >= 1)
    if has_times:
        t = np.asarray(vm.dyn_pred_times, dtype=float)
        lines.append(f"PREDTIMES {len(t)} " + fmt(t))
    else:
        lines.append("PREDTIMES 0")
    if vm.dyn_pred_samples is not None:
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
    lines.append(f"DIM {fmt_i([dimX, dimY, dimZ])}")
    lines.append(f"ORIGIN {fmt(vm.origin)}")

    # ---- precarve (applied to the queried map BEFORE sampling) ----
    lines.append(f"NPRECARVE {len(precarve)}")
    for (pc_c, pc_d) in precarve:
        lines.append("PRECARVE " + fmt(list(np.asarray(pc_c, dtype=float)) + [float(pc_d)]))

    # ---- helper outputs ----
    # find_closest_free_point + has_non_free_neighbor share the sample_pts list
    lines.append(f"NPTS {len(sample_pts)}")
    lines.append("PTS " + fmt([v for p in sample_pts for v in p]))
    lines.append("FCFP " + fmt([v for w in fcfp for v in w]))
    lines.append("HNFN " + fmt_i(hnfn))

    # segments
    lines.append(f"NSEG {len(segs)}")
    lines.append("SEGS " + fmt([v for (a, b) in segs for v in list(a) + list(b)]))
    lines.append("BLK100 " + fmt_i(blk100))
    lines.append("BLK0 " + fmt_i(blk0))
    lines.append("RADII " + fmt_i(radii))
    lines.append("LOS " + fmt_i(los))

    # cells_world counts
    lines.append(f"CWCOUNTS {nfree} {nocc} {nunk}")

    # carve specs/results
    lines.append(f"NCARVE {len(carve_results)}")
    for (ccenter, d, n_used, cells, occ_after) in carve_results:
        lines.append("CARVE_CENTER " + fmt(ccenter))
        lines.append(f"CARVE_D {repr(float(d))}")
        lines.append(f"CARVE_N {n_used}")
        lines.append(f"CARVE_NCELL {len(cells)}")
        lines.append("CARVE_CELLS " + fmt_i([v for c in cells for v in c]))
        lines.append("CARVE_OCC " + fmt_i(occ_after))
    lines.append("END")


# =====================================================================
# Case configurations — mirror gen_voxel_map_golden.py + extra geometry stress.
# =====================================================================
# Case 1: simple — one box obstacle
emit_case(
    0.3,
    dict(cells_x=20, cells_y=20, cells_z=8, center_map=[0.0, 0.0, 1.5],
         z_ground=0.0, z_max=3.0, inflation=0.3, traj_max_time=2.0, knobs={}),
    obst_pos=[[0.5, 0.5, 1.5]],
    obst_bbox=[[0.3, 0.3, 0.4]],
    cloud_occ=[],
)

# Case 1b: SAME map but pre-carved free regions so FREE cells exist. This makes
# is_blocked(val=0) differ from is_blocked(val=100), and find_closest_free_point
# actually return real free-cell centers (not the out-of-radius fallback).
emit_case(
    0.3,
    dict(cells_x=20, cells_y=20, cells_z=8, center_map=[0.0, 0.0, 1.5],
         z_ground=0.0, z_max=3.0, inflation=0.3, traj_max_time=2.0, knobs={}),
    obst_pos=[[0.5, 0.5, 1.5]],
    obst_bbox=[[0.3, 0.3, 0.4]],
    cloud_occ=[],
    precarve=[([0.0, 0.0, 1.5], 1.2), ([0.5, 0.5, 1.5], 0.5)],
)

# Case 2: several boxes + a point cloud, finer res
cloud2 = []
for _ in range(120):
    cloud2.append([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.2, 2.8)])
emit_case(
    0.2,
    dict(cells_x=30, cells_y=30, cells_z=14, center_map=[0.3, -0.4, 1.4],
         z_ground=0.1, z_max=2.9, inflation=0.25, traj_max_time=3.0,
         knobs={"dynamic_as_occupied_future": True, "heat_gamma": 0.15}),
    obst_pos=[[-0.8, 0.6, 1.2], [1.0, -0.7, 1.6], [0.0, 0.0, 1.0]],
    obst_bbox=[[0.25, 0.25, 0.3], [0.4, 0.2, 0.5], [0.15, 0.35, 0.2]],
    cloud_occ=cloud2,
)

# Case 3: static heat stress params (geometry queries unaffected by heat, but
# rebuild identical) — bigger inflation cube
emit_case(
    0.25,
    dict(cells_x=24, cells_y=24, cells_z=10, center_map=[1.0, 1.0, 1.2],
         z_ground=0.0, z_max=2.5, inflation=0.2, traj_max_time=1.5,
         knobs={"static_heat_rmax_m": 1.5, "static_heat_default_radius_m": 1.0,
                "static_heat_p": 3, "static_heat_alpha": 2.0,
                "static_heat_Hmax": 8.0}),
    obst_pos=[[1.2, 0.8, 1.2], [0.5, 1.3, 1.1]],
    obst_bbox=[[0.3, 0.3, 0.4], [0.2, 0.2, 0.3]],
    cloud_occ=[],
)

# Case 4: future cone on — big occupied blobs, lots of blocked segments
emit_case(
    0.3,
    dict(cells_x=18, cells_y=18, cells_z=8, center_map=[0.0, 0.0, 1.0],
         z_ground=0.0, z_max=2.0, inflation=0.3, traj_max_time=2.0,
         knobs={"dynamic_as_occupied_future": True}),
    obst_pos=[[0.3, -0.3, 1.0]],
    obst_bbox=[[0.4, 0.4, 0.5]],
    cloud_occ=[],
)

# Case 5: edge — z clamp hits ground (center near ground), tiny inflation
emit_case(
    0.3,
    dict(cells_x=16, cells_y=16, cells_z=10, center_map=[0.0, 0.0, 0.4],
         z_ground=0.0, z_max=3.0, inflation=0.05, traj_max_time=1.0, knobs={}),
    obst_pos=[[0.2, 0.1, 0.3]],
    obst_bbox=[[0.2, 0.2, 0.2]],
    cloud_occ=[],
)

# Case 6: edge — z clamp hits top, large inflation pad
emit_case(
    0.3,
    dict(cells_x=14, cells_y=14, cells_z=12, center_map=[0.0, 0.0, 2.7],
         z_ground=0.0, z_max=3.0, inflation=0.6, traj_max_time=2.0, knobs={}),
    obst_pos=[[0.0, 0.0, 2.6]],
    obst_bbox=[[0.3, 0.3, 0.3]],
    cloud_occ=[],
)

# Case 7: no obstacles at all (only y-walls + UNKNOWN). is_blocked val=0 should
# treat UNKNOWN(-1) cells as NOT blocked, FREE(0)/OCC(100) as blocked.
emit_case(
    0.3,
    dict(cells_x=12, cells_y=12, cells_z=6, center_map=[0.0, 0.0, 1.0],
         z_ground=0.0, z_max=2.0, inflation=0.3, traj_max_time=2.0, knobs={}),
    obst_pos=[],
    obst_bbox=[],
    cloud_occ=[],
)

# Case 8: dense point cloud -> many occupied cells; stresses DDA through clutter
cloud8 = []
for _ in range(400):
    cloud8.append([rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8), rng.uniform(0.3, 2.6)])
emit_case(
    0.22,
    dict(cells_x=28, cells_y=28, cells_z=12, center_map=[0.0, 0.0, 1.4],
         z_ground=0.0, z_max=3.0, inflation=0.22, traj_max_time=2.5, knobs={}),
    obst_pos=[[-0.6, 0.4, 1.3]],
    obst_bbox=[[0.25, 0.25, 0.3]],
    cloud_occ=cloud8,
)

# Case 8b: dense cloud map, pre-carved a corridor of free cells through the clutter
# -> exercises DDA through mixed FREE/OCC/UNKNOWN with val=0 vs val=100 differing.
emit_case(
    0.22,
    dict(cells_x=28, cells_y=28, cells_z=12, center_map=[0.0, 0.0, 1.4],
         z_ground=0.0, z_max=3.0, inflation=0.22, traj_max_time=2.5, knobs={}),
    obst_pos=[[-0.6, 0.4, 1.3]],
    obst_bbox=[[0.25, 0.25, 0.3]],
    cloud_occ=cloud8,
    precarve=[([0.0, 0.0, 1.4], 1.5), ([0.8, -0.5, 1.4], 0.9)],
)

# Case 9..12: random — randomized params + several random box obstacles + cloud
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
    ncloud = int(rng.integers(0, 120))
    cl = []
    for _ in range(ncloud):
        cl.append([float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)),
                   float(rng.uniform(0.1, 2.8))])
    knobs = {
        "heat_p": int(rng.integers(1, 4)),
        "heat_q": int(rng.integers(1, 4)),
        "obst_max_vel": float(rng.uniform(0.3, 1.0)),
        "dynamic_as_occupied_future": bool(rng.integers(0, 2)),
    }
    emit_case(
        res,
        dict(cells_x=cx, cells_y=cy, cells_z=cz, center_map=cmap_center,
             z_ground=0.0, z_max=3.0, inflation=infl, traj_max_time=tmax, knobs=knobs),
        obst_pos=op, obst_bbox=ob, cloud_occ=cl,
    )

out = os.path.join(os.path.dirname(__file__), "voxel_map_helpers_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len([l for l in lines if l == 'CASE'])} cases)")
