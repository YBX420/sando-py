"""Golden-data generator for HGPManager (hgp_manager.py) C++ port verification.

不许欺骗、必须还原:这里用真 Python HGPManager(set_parameters / update_map / solve_hgp /
create_more_vertexes / free_start / free_goal)跑「输入 -> 输出」金标准。
C++ 移植后用同样 Parameters + 同样 update_map 输入 + 同样 start/start_vel/goal 跑
solve_hgp,以及同样的 polyline 跑 create_more_vertexes、同样的 free_start/free_goal
carving,得到的结果必须逐项对上(processed/raw waypoints world abs 2e-4 + cell-index
exact;create_more_vertexes world abs 2e-4;free carve 后 cell 占据/空闲状态 exact)。

Three families of cases dumped to cpp/golden/hgp_manager_cases.txt:

  SOLVE cases  (KIND=0):  build a real HGPManager via set_parameters(par) then
      update_map(wdx,wdy,wdz, center, cloud_occ, None, obst_pos, obst_bbox, tmax)
      (this drives map_util.read_map AND auto-builds the planner), then
      solve_hgp(start, start_vel, goal, current_time). Dumps:
        - the full Parameters fields the C++ must mirror,
        - update_map inputs (wdx/wdy/wdz, center, cloud, obstacles, tmax),
        - mgr-derived res / drone_radius / weight / map dim,
        - SUCCESS, FINALG,
        - PROC (densified processed path, world + cells),
        - RAW  (raw A* path, world + cells).
      Also dumps a few free_start/free_goal carve probes: BEFORE solve we ask the
      mgr (a fresh re-built copy) whether a probe cell is free; AFTER free_start /
      free_goal carving we ask again — the C++ must reproduce the exact free flips.

  CMV cases    (KIND=1):  create_more_vertexes(polyline, d) on diverse + random +
      edge polylines (single pt, zero-length seg, exact-multiple seg, d<=0, ...).
      Dumps the input polyline + d + the exact output polyline (world abs 2e-4).

  CARVE cases  (KIND=2):  free_start / free_goal carving on a fresh map. Dumps the
      map build + the carve center + factor, and a list of probe cells with their
      is_free state BEFORE and AFTER the carve — C++ must reproduce every flip.

跑: python cpp/golden/gen_hgp_manager_golden.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.types import Parameters  # noqa: E402
from sando_py.hgp.hgp_manager import HGPManager  # noqa: E402
from sando_py.hgp.voxel_map import VAL_OCC  # noqa: E402

rng = np.random.default_rng(20260604)

lines = []


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


def fmt_i(a):
    return " ".join(str(int(x)) for x in np.asarray(a).reshape(-1))


# ---- Parameters fields the C++ HGPManager.set_parameters reads (in order) ----
# floats:
PAR_FLOAT = [
    "global_planner_heuristic_weight", "factor_hgp", "res", "inflation_hgp",
    "z_min", "z_max", "v_max", "a_max", "j_max", "max_dist_vertexes",
    "shrinked_box_size", "free_start_factor", "free_goal_factor",
    "w_unknown", "w_align", "decay_len_cells", "w_side", "min_len", "min_turn",
    "heat_weight", "obstacle_soft_cost",
    "heat_alpha0", "heat_alpha1", "heat_tau_ratio", "heat_gamma", "heat_Hmax",
    "dyn_base_inflation_m", "dyn_heat_tube_radius_m", "obst_max_vel",
    "static_heat_alpha", "static_heat_Hmax", "static_heat_rmax_m",
    "static_heat_default_radius_m",
]
# ints:
PAR_INT = [
    "hgp_timeout_duration_ms", "max_num_expansion", "los_cells",
    "heat_p", "heat_q", "heat_num_samples", "static_heat_p",
]
# bools:
PAR_BOOL = [
    "use_free_start", "use_free_goal", "use_shrinked_box",
    "use_heat_map", "dynamic_heat_enabled", "dynamic_as_occupied_current",
    "dynamic_as_occupied_future", "static_heat_enabled",
    "static_heat_boundary_only", "static_heat_apply_on_unknown",
    "static_heat_exclude_dynamic", "use_soft_cost_obstacles",
]


def box_cloud(center, half, step):
    """Surface point cloud of an axis-aligned box (static obstacle for read_map)."""
    c = np.asarray(center, dtype=float)
    h = np.asarray(half, dtype=float)
    pts = []
    nx = max(1, int(round(2 * h[0] / step)))
    ny = max(1, int(round(2 * h[1] / step)))
    nz = max(1, int(round(2 * h[2] / step)))
    xs = np.linspace(c[0] - h[0], c[0] + h[0], nx + 1)
    ys = np.linspace(c[1] - h[1], c[1] + h[1], ny + 1)
    zs = np.linspace(c[2] - h[2], c[2] + h[2], nz + 1)
    for x in xs:
        for y in ys:
            for z in zs:
                # keep only surface points (at least one coord on a face)
                on = (abs(x - (c[0] - h[0])) < 1e-9 or abs(x - (c[0] + h[0])) < 1e-9 or
                      abs(y - (c[1] - h[1])) < 1e-9 or abs(y - (c[1] + h[1])) < 1e-9 or
                      abs(z - (c[2] - h[2])) < 1e-9 or abs(z - (c[2] + h[2])) < 1e-9)
                if on:
                    pts.append([x, y, z])
    return pts


def emit_param_block(par):
    lines.append("PARFLOAT " + fmt([float(getattr(par, k)) for k in PAR_FLOAT]))
    lines.append("PARINT " + fmt_i([int(getattr(par, k)) for k in PAR_INT]))
    lines.append("PARBOOL " + " ".join(str(int(bool(getattr(par, k)))) for k in PAR_BOOL))
    lines.append("PARGP " + str(par.global_planner))
    lines.append("PARDRONEBBOX " + fmt(par.drone_bbox))
    lines.append("PARSFC " + fmt(par.sfc_size))


def emit_map_block(wdx, wdy, wdz, center, cloud_occ, obst_pos, obst_bbox, tmax):
    lines.append("WD " + fmt([wdx, wdy, wdz]))
    lines.append("CENTER " + fmt(center))
    lines.append("TMAX " + repr(float(tmax)))
    lines.append(f"NCLOUD {len(cloud_occ)}")
    lines.append("CLOUD " + (fmt([v for c in cloud_occ for v in c]) if len(cloud_occ) else ""))
    lines.append(f"NOBST {len(obst_pos)}")
    if obst_pos:
        lines.append("OBSTPOS " + fmt([v for c in obst_pos for v in c]))
        lines.append("OBSTBBOX " + fmt([v for b in obst_bbox for v in b]))
    else:
        lines.append("OBSTPOS")
        lines.append("OBSTBBOX")


def find_free_world(vm, prefer=None):
    dimX, dimY, dimZ = int(vm.dim[0]), int(vm.dim[1]), int(vm.dim[2])
    mid = [dimX // 2, dimY // 2, dimZ // 2]
    candidates = []
    if prefer is not None:
        # allow None entries -> substitute the corresponding mid coord
        candidates.append([prefer[i] if prefer[i] is not None else mid[i]
                           for i in range(3)])
    candidates.append(mid)
    for _ in range(3000):
        candidates.append([int(rng.integers(0, dimX)), int(rng.integers(0, dimY)),
                           int(rng.integers(0, dimZ))])
    for c in candidates:
        x, y, z = int(c[0]), int(c[1]), int(c[2])
        if not vm.in_bounds(x, y, z):
            continue
        if int(vm.cmap[vm.lin_index(x, y, z)]) == int(VAL_OCC):
            continue
        return vm.int_to_float(np.array([x, y, z]))
    return vm.int_to_float(np.array([dimX // 2, dimY // 2, dimZ // 2]))


def emit_solve_case(par, wdx, wdy, wdz, center, cloud_occ, obst_pos, obst_bbox,
                    tmax, start, start_vel, goal, current_time,
                    prefer_start=None, prefer_goal=None):
    # Build a "preview" manager to find sensible free start/goal if not given.
    mprev = HGPManager()
    mprev.set_parameters(par)
    mprev.update_map(wdx, wdy, wdz, np.asarray(center, dtype=float),
                     np.asarray(cloud_occ, dtype=float) if len(cloud_occ) else np.zeros((0, 3)),
                     None,
                     [np.asarray(p, dtype=float) for p in obst_pos],
                     [np.asarray(b, dtype=float) for b in obst_bbox], tmax)
    if start is None:
        start = find_free_world(mprev.map_util, prefer=prefer_start)
    if goal is None:
        goal = find_free_world(mprev.map_util, prefer=prefer_goal)
    start = np.asarray(start, dtype=float)
    start_vel = np.asarray(start_vel, dtype=float)
    goal = np.asarray(goal, dtype=float)

    # ---- free_start / free_goal carve probes (on a FRESH manager) ----
    mcarve = HGPManager()
    mcarve.set_parameters(par)
    mcarve.update_map(wdx, wdy, wdz, np.asarray(center, dtype=float),
                      np.asarray(cloud_occ, dtype=float) if len(cloud_occ) else np.zeros((0, 3)),
                      None,
                      [np.asarray(p, dtype=float) for p in obst_pos],
                      [np.asarray(b, dtype=float) for b in obst_bbox], tmax)
    vm = mcarve.map_util
    sx, sy, sz = (int(v) for v in vm.float_to_int(start))
    # probe cells = a 3x3x3 neighborhood around the start cell (clamped reported as state)
    probes = []
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            for dz in (-1, 0, 1):
                probes.append([sx + dx, sy + dy, sz + dz])
    before = [1 if vm.is_free(int(c[0]), int(c[1]), int(c[2])) else 0 for c in probes]
    if par.use_free_start:
        mcarve.free_start(start, par.free_start_factor)
    if par.use_free_goal:
        mcarve.free_goal(goal, par.free_goal_factor)
    after = [1 if vm.is_free(int(c[0]), int(c[1]), int(c[2])) else 0 for c in probes]

    # ---- the actual solve (fresh manager so carving inside solve is exercised) ----
    m = HGPManager()
    m.set_parameters(par)
    m.update_map(wdx, wdy, wdz, np.asarray(center, dtype=float),
                 np.asarray(cloud_occ, dtype=float) if len(cloud_occ) else np.zeros((0, 3)),
                 None,
                 [np.asarray(p, dtype=float) for p in obst_pos],
                 [np.asarray(b, dtype=float) for b in obst_bbox], tmax)
    vms = m.map_util
    dimX, dimY, dimZ = int(vms.dim[0]), int(vms.dim[1]), int(vms.dim[2])
    ok, final_g, path, raw = m.solve_hgp(start, start_vel, goal, current_time)

    def cells_of(pts):
        return [list(map(int, vms.float_to_int(np.asarray(p, dtype=float)))) for p in pts]

    proc_cells = cells_of(path)
    raw_cells = cells_of(raw)

    lines.append("CASE")
    lines.append("KIND 0")
    emit_param_block(par)
    emit_map_block(wdx, wdy, wdz, center, cloud_occ, obst_pos, obst_bbox, tmax)
    lines.append("START " + fmt(start))
    lines.append("STARTVEL " + fmt(start_vel))
    lines.append("GOAL " + fmt(goal))
    lines.append("CURTIME " + repr(float(current_time)))
    # mgr-derived scalars (C++ must reproduce these too)
    lines.append("MGRRES " + repr(float(m.res)))
    lines.append("MGRDRONER " + repr(float(m.drone_radius)))
    lines.append("MGRWEIGHT " + repr(float(m.weight)))
    lines.append("DIM " + fmt_i([dimX, dimY, dimZ]))
    lines.append("SUCCESS " + ("1" if ok else "0"))
    lines.append("FINALG " + repr(float(final_g)))
    lines.append(f"NPROC {len(path)}")
    lines.append("PROC " + (fmt([v for w in path for v in w]) if len(path) else ""))
    lines.append("PROCCELL " + (fmt_i([v for c in proc_cells for v in c]) if proc_cells else ""))
    lines.append(f"NRAW {len(raw)}")
    lines.append("RAW " + (fmt([v for w in raw for v in w]) if len(raw) else ""))
    lines.append("RAWCELL " + (fmt_i([v for c in raw_cells for v in c]) if raw_cells else ""))
    # carve probes
    lines.append(f"NPROBE {len(probes)}")
    lines.append("PROBE " + fmt_i([v for c in probes for v in c]))
    lines.append("CARVEBEFORE " + " ".join(str(v) for v in before))
    lines.append("CARVEAFTER " + " ".join(str(v) for v in after))
    lines.append("END")


def emit_cmv_case(poly, d):
    m = HGPManager()
    out = m.create_more_vertexes([np.asarray(p, dtype=float) for p in poly], float(d))
    lines.append("CASE")
    lines.append("KIND 1")
    lines.append(f"NIN {len(poly)}")
    lines.append("IN " + (fmt([v for p in poly for v in p]) if len(poly) else ""))
    lines.append("D " + repr(float(d)))
    lines.append(f"NOUT {len(out)}")
    lines.append("OUT " + (fmt([v for p in out for v in p]) if len(out) else ""))
    lines.append("END")


# =====================================================================
# SOLVE cases
# =====================================================================
# Case S1: defaults (factor_hgp=1, inflation_hgp=0.45, res=0.3), one dynamic box.
par1 = Parameters()
emit_solve_case(par1, wdx=8.0, wdy=8.0, wdz=3.0, center=[0.0, 0.0, 1.5],
                cloud_occ=[], obst_pos=[[0.5, 0.0, 1.5]], obst_bbox=[[0.3, 0.3, 0.4]],
                tmax=2.0, start=None, start_vel=[0.0, 0.0, 0.0], goal=None,
                current_time=0.0, prefer_start=[3, None, None],
                prefer_goal=[None, None, None])

# Case S2: coarser factor_hgp, static cloud box + dynamic box, free_goal ON,
#          sastar (no heat), smaller inflation.
par2 = Parameters()
par2.factor_hgp = 1.5
par2.inflation_hgp = 0.3
par2.res = 0.25
par2.global_planner = "sastar"
par2.use_free_goal = True
par2.free_start_factor = 1.0
par2.free_goal_factor = 1.5
par2.max_dist_vertexes = 0.8
par2.global_planner_heuristic_weight = 1.0
par2.min_len = 0.4
par2.min_turn = 12.0
par2.los_cells = 1
par2.drone_bbox = [0.3, 0.25, 0.2]
cloud2 = box_cloud([0.0, 0.6, 1.4], [0.3, 0.2, 0.5], 0.1)
emit_solve_case(par2, wdx=9.0, wdy=9.0, wdz=3.0, center=[0.0, 0.0, 1.4],
                cloud_occ=cloud2, obst_pos=[[-0.6, -0.4, 1.3]],
                obst_bbox=[[0.25, 0.25, 0.3]], tmax=2.5, start=None,
                start_vel=[0.1, 0.2, 0.0], goal=None, current_time=0.5,
                prefer_start=[2, 2, None], prefer_goal=[None, None, None])

# Case S3: heat ON, larger heat_weight, free_start OFF, eps>1, nonzero start vel.
par3 = Parameters()
par3.factor_hgp = 1.0
par3.inflation_hgp = 0.3
par3.res = 0.3
par3.use_free_start = False
par3.use_free_goal = False
par3.global_planner_heuristic_weight = 1.5
par3.heat_weight = 8.0
par3.heat_gamma = 0.1
par3.w_unknown = 0.5
par3.los_cells = 2
par3.min_len = 0.6
par3.min_turn = 20.0
par3.max_dist_vertexes = 1.2
emit_solve_case(par3, wdx=8.0, wdy=8.0, wdz=3.0, center=[0.0, 0.0, 1.4],
                cloud_occ=[], obst_pos=[[0.4, 0.4, 1.4], [-0.5, 0.5, 1.3]],
                obst_bbox=[[0.3, 0.3, 0.4], [0.25, 0.25, 0.3]], tmax=2.0,
                start=None, start_vel=[0.3, -0.2, 0.0], goal=None,
                current_time=1.0, prefer_start=[3, None, None],
                prefer_goal=[None, None, None])

# Case S4: hard obstacles (no soft cost), small max_dist_vertexes -> heavy densify.
par4 = Parameters()
par4.use_soft_cost_obstacles = False
par4.inflation_hgp = 0.3
par4.res = 0.3
par4.max_dist_vertexes = 0.5
par4.los_cells = 1
par4.min_len = 0.4
par4.min_turn = 10.0
par4.global_planner_heuristic_weight = 1.0
emit_solve_case(par4, wdx=7.0, wdy=7.0, wdz=2.5, center=[0.0, 0.0, 1.2],
                cloud_occ=[], obst_pos=[[0.3, 0.0, 1.2], [-0.4, 0.5, 1.1]],
                obst_bbox=[[0.3, 0.4, 0.5], [0.25, 0.25, 0.4]], tmax=2.0,
                start=None, start_vel=[0.0, 0.0, 0.0], goal=None,
                current_time=0.0, prefer_start=[3, 3, None],
                prefer_goal=[None, None, None])

# Case S5: start == goal (trivial), tiny map.
par5 = Parameters()
par5.inflation_hgp = 0.3
par5.res = 0.3
mp = HGPManager(); mp.set_parameters(par5)
mp.update_map(5.0, 5.0, 2.5, np.array([0.0, 0.0, 1.4]), np.zeros((0, 3)), None, [], [], 1.5)
sg = find_free_world(mp.map_util, prefer=[mp.map_util.dim[0] // 2,
                                          mp.map_util.dim[1] // 2,
                                          mp.map_util.dim[2] // 2])
emit_solve_case(par5, wdx=5.0, wdy=5.0, wdz=2.5, center=[0.0, 0.0, 1.4],
                cloud_occ=[], obst_pos=[], obst_bbox=[], tmax=1.5,
                start=sg, start_vel=[0.0, 0.0, 0.0], goal=sg, current_time=0.0)

# Case S6+: randomized maps + random Parameters knobs + random free start/goal.
for trial in range(8):
    par = Parameters()
    par.factor_hgp = float(rng.uniform(1.0, 1.6))
    par.res = float(rng.uniform(0.2, 0.32))
    par.inflation_hgp = float(rng.uniform(0.2, 0.4))
    par.global_planner = "astar_heat" if rng.integers(0, 2) else "sastar"
    par.global_planner_heuristic_weight = float(rng.uniform(1.0, 2.0))
    par.use_free_start = bool(rng.integers(0, 2))
    par.use_free_goal = bool(rng.integers(0, 2))
    par.free_start_factor = float(rng.uniform(1.0, 3.0))
    par.free_goal_factor = float(rng.uniform(1.0, 3.0))
    par.max_dist_vertexes = float(rng.uniform(0.4, 1.5))
    par.use_soft_cost_obstacles = bool(rng.integers(0, 2))
    par.heat_weight = float(rng.uniform(4.0, 12.0))
    par.obstacle_soft_cost = float(rng.uniform(3.0, 7.0))
    par.w_unknown = float(rng.uniform(0.0, 0.5))
    par.los_cells = int(rng.integers(0, 4))
    par.min_len = float(rng.uniform(0.2, 0.7))
    par.min_turn = float(rng.uniform(5.0, 25.0))
    par.heat_gamma = float(rng.uniform(0.0, 0.2))
    par.heat_alpha0 = float(rng.uniform(0.1, 0.5))
    par.heat_alpha1 = float(rng.uniform(0.6, 1.4))
    par.heat_p = int(rng.integers(1, 4))
    par.heat_q = int(rng.integers(1, 4))
    par.drone_bbox = [float(rng.uniform(0.15, 0.4)) for _ in range(3)]
    wdx = float(rng.uniform(6.0, 10.0))
    wdy = float(rng.uniform(6.0, 10.0))
    wdz = float(rng.uniform(2.5, 3.5))
    center = [float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)),
              float(rng.uniform(1.2, 1.7))]
    tmax = float(rng.uniform(1.5, 3.0))
    nobst = int(rng.integers(1, 4))
    op, ob = [], []
    for _ in range(nobst):
        op.append([float(rng.uniform(-1.2, 1.2)), float(rng.uniform(-1.2, 1.2)),
                   float(rng.uniform(1.1, 1.7))])
        ob.append([float(rng.uniform(0.15, 0.4)), float(rng.uniform(0.15, 0.4)),
                   float(rng.uniform(0.2, 0.5))])
    cloud = []
    if rng.integers(0, 2):
        cloud = box_cloud([float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.6, 0.6)),
                           float(rng.uniform(1.2, 1.6))],
                          [float(rng.uniform(0.2, 0.4)), float(rng.uniform(0.2, 0.4)),
                           float(rng.uniform(0.3, 0.5))], 0.12)
    sv = [float(rng.uniform(-0.3, 0.3)) for _ in range(3)]
    emit_solve_case(par, wdx, wdy, wdz, center, cloud, op, ob, tmax,
                    start=None, start_vel=sv, goal=None,
                    current_time=float(rng.uniform(0.0, 2.0)),
                    prefer_start=[2, 2, None], prefer_goal=[None, None, None])


# =====================================================================
# CMV (create_more_vertexes) cases — diverse + random + edge
# =====================================================================
# C1: straight line, d divides evenly
emit_cmv_case([[0, 0, 0], [4, 0, 0]], 1.0)
# C2: straight line, non-integer division
emit_cmv_case([[0, 0, 0], [3.5, 0, 0]], 1.0)
# C3: 3D polyline with multiple segments
emit_cmv_case([[0, 0, 0], [2, 2, 0], [2, 2, 3], [5, 2, 3]], 0.7)
# C4: segment shorter than d (no insertion, endpoint appended)
emit_cmv_case([[0, 0, 0], [0.4, 0.3, 0.0]], 1.0)
# C5: zero-length segment in the middle (skipped)
emit_cmv_case([[0, 0, 0], [0, 0, 0], [2, 0, 0]], 0.5)
# C6: single point (returned as-is, len<2)
emit_cmv_case([[1.0, 2.0, 3.0]], 0.5)
# C7: empty
emit_cmv_case([], 0.5)
# C8: d <= 0 (returns copy of input)
emit_cmv_case([[0, 0, 0], [2, 2, 2]], 0.0)
emit_cmv_case([[0, 0, 0], [2, 2, 2]], -1.0)
# C9: exact-multiple where last inserted equals endpoint (no duplicate tail)
emit_cmv_case([[0, 0, 0], [2.0, 0, 0]], 1.0)
# C10: d larger than every segment
emit_cmv_case([[0, 0, 0], [0.5, 0, 0], [1.0, 0.5, 0]], 5.0)
# random CMV
for _ in range(12):
    npts = int(rng.integers(1, 7))
    poly = [[float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)),
             float(rng.uniform(-3, 3))] for _ in range(npts)]
    # occasionally duplicate a point to make a zero-length seg
    if npts >= 2 and rng.integers(0, 3) == 0:
        j = int(rng.integers(0, npts - 1))
        poly[j + 1] = list(poly[j])
    d = float(rng.uniform(0.1, 3.0))
    emit_cmv_case(poly, d)


out = os.path.join(os.path.dirname(__file__), "hgp_manager_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
ncase = len([l for l in lines if l == "CASE"])
print(f"wrote {out}  ({ncase} cases)")
