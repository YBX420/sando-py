"""Golden-data generator for HGPPlanner (hgp_planner.py) C++ port verification.

不许欺骗、必须还原:这里用真 Python HGPPlanner.plan() 跑出「输入 -> 输出」金标准,
C++ 移植后用同样地图+同样 HGPPlanner 旋钮+同样起终点跑 plan(),得到的
  - raw waypoints(A* 原始路径,首尾替换成真实 start/goal 世界坐标)
  - processed waypoints(整条瘦身流水线后的路径)
必须逐点对上(world coords abs 2e-4,cell-index exact where possible);
对不上就不算还原。

Each case:
  - builds a real VoxelMapUtil via read_map(...) with box obstacles (so cmap
    occupancy + heat field are real, identical to what the C++ test rebuilds),
  - constructs HGPPlanner(...) with the relevant knobs
    (los_cells/min_len/min_turn/heat_weight/obstacle_soft_cost/...),
  - set_map_util(vm); set_max_expand(...); plan(start, start_vel, goal, t, eps),
  - dumps the HGPPlanner knobs, GraphSearch knobs, start/goal (WORLD coords) +
    start_vel + eps + max_expand, the success flag, final_g, AND:
      * RAW   : raw waypoints (world coords) + their float_to_int cell indices
      * PROC  : processed waypoints (world coords) + their float_to_int cell indices

The C++ test rebuilds the SAME map (reusing the golden voxel_map machinery), builds
the SAME HGPPlanner, runs plan(), and must reproduce BOTH sequences.

Timeout note: Python uses a wall-clock deadline inside GraphSearch + an HGPPlanner
timing dict; both are NON-reproducible and NOT ported. We pass a HUGE timeout so the
deadline never trips (deterministic, matches the C++ which drops the timeout).

dump 到 cpp/golden/hgp_planner_cases.txt(纯文本,C++ 无需 JSON 依赖即可解析)。
跑: python cpp/golden/gen_hgp_planner_golden.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from sando_py.hgp.voxel_map import VoxelMapUtil, VAL_OCC  # noqa: E402
from sando_py.hgp.hgp_planner import HGPPlanner  # noqa: E402

rng = np.random.default_rng(20260603)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


def fmt_i(a):
    return " ".join(str(int(x)) for x in np.asarray(a).reshape(-1))


lines = []

# ---- VoxelMapUtil knobs the C++ side must mirror (same list as voxel_map golden) ----
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

BIG_TIMEOUT = 10_000_000  # ms — effectively never trips (deterministic search)


def find_free_world(vm, prefer=None, avoid=None):
    """Return the WORLD coord (cell center) of an in-bounds non-occupied cell."""
    dimX, dimY, dimZ = int(vm.dim[0]), int(vm.dim[1]), int(vm.dim[2])
    candidates = []
    if prefer is not None:
        candidates.append(prefer)
    cx, cy, cz = dimX // 2, dimY // 2, dimZ // 2
    candidates.append([cx, cy, cz])
    for _ in range(2000):
        x = int(rng.integers(0, dimX))
        y = int(rng.integers(0, dimY))
        z = int(rng.integers(0, dimZ))
        candidates.append([x, y, z])
    for c in candidates:
        x, y, z = int(c[0]), int(c[1]), int(c[2])
        if not vm.in_bounds(x, y, z):
            continue
        if int(vm.cmap[vm.lin_index(x, y, z)]) == int(VAL_OCC):
            continue
        if avoid is not None and [x, y, z] == list(avoid):
            continue
        return vm.int_to_float(np.array([x, y, z]))
    return vm.int_to_float(np.array([cx, cy, cz]))


def emit_case(cfg, obst_pos, obst_bbox, cloud_occ, hgp_cfg,
              start, start_vel, goal, eps, max_expand):
    # Build the map.
    vm = VoxelMapUtil(res=cfg["res"])
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

    # Build HGPPlanner with the requested knobs.
    hgp = HGPPlanner(
        global_planner=hgp_cfg.get("global_planner", "astar_heat"),
        verbose=False,
        v_max=hgp_cfg.get("v_max", 3.0),
        a_max=hgp_cfg.get("a_max", 5.0),
        j_max=hgp_cfg.get("j_max", 20.0),
        timeout_duration_ms=hgp_cfg.get("timeout_ms", BIG_TIMEOUT),
        w_unknown=hgp_cfg.get("w_unknown", 0.0),
        w_align=hgp_cfg.get("w_align", 0.0),
        decay_len_cells=hgp_cfg.get("decay_len_cells", 20.0),
        w_side=hgp_cfg.get("w_side", 0.2),
        los_cells=hgp_cfg.get("los_cells", 0),
        min_len=hgp_cfg.get("min_len", 0.5),
        min_turn=hgp_cfg.get("min_turn", 10.0),
        heat_weight=hgp_cfg.get("heat_weight", 10.0),
        obstacle_soft_cost=hgp_cfg.get("obstacle_soft_cost", 5.0),
    )
    hgp.set_map_util(vm)
    hgp.set_max_expand(int(max_expand))

    start = np.asarray(start, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    ok, final_g = hgp.plan(start, start_vel, goal, 0.0, eps=float(eps))

    raw_path = hgp.get_raw_path()
    proc_path = hgp.get_path()

    def cells_of(pts):
        return [list(map(int, vm.float_to_int(np.asarray(p, dtype=np.float64)))) for p in pts]

    raw_cells = cells_of(raw_path)
    proc_cells = cells_of(proc_path)

    lines.append("CASE")
    # ---- VoxelMapUtil rebuild inputs (same schema as graph_search golden) ----
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
    lines.append("PREDTIMES 0")
    lines.append("PREDSAMP -1")
    # ---- HGPPlanner config ----
    lines.append("HGPPLANNER " + hgp_cfg.get("global_planner", "astar_heat"))
    # HGP_FLOAT order mirrored on C++ side:
    #   w_unknown, w_align, decay_len_cells, w_side, min_len, min_turn,
    #   heat_weight, obstacle_soft_cost
    lines.append("HGPFLOAT " + fmt([
        hgp.w_unknown, hgp.w_align, hgp.decay_len_cells, hgp.w_side,
        hgp.min_len, hgp.min_turn_deg, hgp.heat_weight, hgp.obstacle_soft_cost,
    ]))
    lines.append(f"LOSCELLS {int(hgp.los_cells)}")
    lines.append(f"MAXEXPAND {int(max_expand)}")
    lines.append(f"EPS {repr(float(eps))}")
    lines.append("START " + fmt(start))
    lines.append("STARTVEL " + fmt(start_vel))
    lines.append("GOAL " + fmt(goal))
    # ---- OUTPUTS ----
    lines.append(f"DIM {fmt_i([dimX, dimY, dimZ])}")
    lines.append(f"SUCCESS {1 if ok else 0}")
    lines.append(f"FINALG {repr(float(final_g))}")
    lines.append(f"NRAW {len(raw_path)}")
    lines.append("RAW " + (fmt([v for w in raw_path for v in w]) if len(raw_path) else ""))
    lines.append("RAWCELL " + (fmt_i([v for c in raw_cells for v in c]) if raw_cells else ""))
    lines.append(f"NPROC {len(proc_path)}")
    lines.append("PROC " + (fmt([v for w in proc_path for v in w]) if len(proc_path) else ""))
    lines.append("PROCCELL " + (fmt_i([v for c in proc_cells for v in c]) if proc_cells else ""))
    lines.append("END")


def base_cfg(**kw):
    d = dict(res=0.3, cells_x=24, cells_y=24, cells_z=8, center_map=[0.0, 0.0, 1.5],
             z_ground=0.0, z_max=3.0, inflation=0.3, traj_max_time=2.0, knobs={})
    d.update(kw)
    return d


# =====================================================================
# Case 1: one box between start & goal; LoS shortcut + corner removal active.
# =====================================================================
cfg1 = base_cfg(cells_x=24, cells_y=24, cells_z=8, inflation=0.3, traj_max_time=2.0)
vm1 = VoxelMapUtil(res=cfg1["res"])
vm1.read_map(cfg1["cells_x"], cfg1["cells_y"], cfg1["cells_z"],
             np.asarray(cfg1["center_map"]), np.zeros((0, 3)),
             cfg1["z_ground"], cfg1["z_max"], cfg1["inflation"],
             [np.asarray([0.5, 0.0, 1.5])], [np.asarray([0.3, 0.3, 0.4])],
             cfg1["traj_max_time"])
s1 = find_free_world(vm1, prefer=[3, vm1.dim[1] // 2, vm1.dim[2] // 2])
g1 = find_free_world(vm1, prefer=[vm1.dim[0] - 4, vm1.dim[1] // 2, vm1.dim[2] // 2])
emit_case(cfg1, obst_pos=[[0.5, 0.0, 1.5]], obst_bbox=[[0.3, 0.3, 0.4]], cloud_occ=[],
          hgp_cfg=dict(global_planner="astar_heat", heat_weight=10.0, los_cells=2,
                       min_len=0.5, min_turn=10.0),
          start=s1, start_vel=[0.0, 0.0, 0.0], goal=g1, eps=1.0, max_expand=200000)

# =====================================================================
# Case 2: several boxes, los_cells=0 (no inflation), tighter min_len/min_turn.
# =====================================================================
cfg2 = base_cfg(res=0.25, cells_x=30, cells_y=28, cells_z=10,
                center_map=[0.2, -0.1, 1.4], z_ground=0.1, z_max=2.9,
                inflation=0.25, traj_max_time=2.5, knobs={"heat_gamma": 0.1})
op2 = [[-0.6, 0.4, 1.3], [0.8, -0.5, 1.4], [0.0, 0.0, 1.2]]
ob2 = [[0.25, 0.25, 0.3], [0.3, 0.2, 0.4], [0.2, 0.3, 0.25]]
vm2 = VoxelMapUtil(res=cfg2["res"])
vm2.heat_gamma = 0.1
vm2.read_map(cfg2["cells_x"], cfg2["cells_y"], cfg2["cells_z"],
             np.asarray(cfg2["center_map"]), np.zeros((0, 3)),
             cfg2["z_ground"], cfg2["z_max"], cfg2["inflation"],
             [np.asarray(p) for p in op2], [np.asarray(b) for b in ob2],
             cfg2["traj_max_time"])
s2 = find_free_world(vm2, prefer=[2, 2, vm2.dim[2] // 2])
g2 = find_free_world(vm2, prefer=[vm2.dim[0] - 3, vm2.dim[1] - 3, vm2.dim[2] // 2])
emit_case(cfg2, obst_pos=op2, obst_bbox=ob2, cloud_occ=[],
          hgp_cfg=dict(global_planner="astar_heat", heat_weight=8.0,
                       obstacle_soft_cost=4.0, los_cells=0, min_len=0.3,
                       min_turn=15.0),
          start=s2, start_vel=[0.1, 0.2, 0.0], goal=g2, eps=1.0, max_expand=300000)

# =====================================================================
# Case 3: hard obstacles (use_soft_cost_obstacles False), plain A* (sastar).
#         Forces _repair_blocked_segments to potentially splice raw sub-paths.
# =====================================================================
cfg3 = base_cfg(res=0.3, cells_x=22, cells_y=22, cells_z=8, center_map=[0.0, 0.0, 1.2],
                z_ground=0.0, z_max=2.5, inflation=0.3, traj_max_time=2.0,
                knobs={"use_soft_cost_obstacles": False})
op3 = [[0.3, 0.0, 1.2], [-0.4, 0.5, 1.1]]
ob3 = [[0.3, 0.4, 0.5], [0.25, 0.25, 0.4]]
vm3 = VoxelMapUtil(res=cfg3["res"])
vm3.use_soft_cost_obstacles = False
vm3.read_map(cfg3["cells_x"], cfg3["cells_y"], cfg3["cells_z"],
             np.asarray(cfg3["center_map"]), np.zeros((0, 3)),
             cfg3["z_ground"], cfg3["z_max"], cfg3["inflation"],
             [np.asarray(p) for p in op3], [np.asarray(b) for b in ob3],
             cfg3["traj_max_time"])
s3 = find_free_world(vm3, prefer=[3, 3, vm3.dim[2] // 2])
g3 = find_free_world(vm3, prefer=[vm3.dim[0] - 4, vm3.dim[1] - 4, vm3.dim[2] // 2])
emit_case(cfg3, obst_pos=op3, obst_bbox=ob3, cloud_occ=[],
          hgp_cfg=dict(global_planner="sastar", los_cells=1, min_len=0.4,
                       min_turn=12.0),
          start=s3, start_vel=[0.0, 0.0, 0.0], goal=g3, eps=1.0, max_expand=300000)

# =====================================================================
# Case 4: start == goal (trivial path).
# =====================================================================
cfg4 = base_cfg(cells_x=16, cells_y=16, cells_z=6)
vm4 = VoxelMapUtil(res=cfg4["res"])
vm4.read_map(cfg4["cells_x"], cfg4["cells_y"], cfg4["cells_z"],
             np.asarray(cfg4["center_map"]), np.zeros((0, 3)),
             cfg4["z_ground"], cfg4["z_max"], cfg4["inflation"], [], [],
             cfg4["traj_max_time"])
s4 = find_free_world(vm4, prefer=[vm4.dim[0] // 2, vm4.dim[1] // 2, vm4.dim[2] // 2])
emit_case(cfg4, obst_pos=[], obst_bbox=[], cloud_occ=[],
          hgp_cfg=dict(global_planner="astar_heat", los_cells=2, min_len=0.5,
                       min_turn=10.0),
          start=s4, start_vel=[0.0, 0.0, 0.0], goal=s4, eps=1.0, max_expand=100000)

# =====================================================================
# Case 5: tiny max_expand -> best_node fallback (success likely False).
# =====================================================================
cfg5 = base_cfg(res=0.3, cells_x=26, cells_y=26, cells_z=8, center_map=[0.0, 0.0, 1.2],
                z_ground=0.0, z_max=2.5, inflation=0.3, traj_max_time=2.0)
vm5 = VoxelMapUtil(res=cfg5["res"])
vm5.read_map(cfg5["cells_x"], cfg5["cells_y"], cfg5["cells_z"],
             np.asarray(cfg5["center_map"]), np.zeros((0, 3)),
             cfg5["z_ground"], cfg5["z_max"], cfg5["inflation"],
             [np.asarray([0.0, 0.0, 1.2])], [np.asarray([0.4, 0.4, 0.5])],
             cfg5["traj_max_time"])
s5 = find_free_world(vm5, prefer=[2, 2, vm5.dim[2] // 2])
g5 = find_free_world(vm5, prefer=[vm5.dim[0] - 3, vm5.dim[1] - 3, vm5.dim[2] // 2])
emit_case(cfg5, obst_pos=[[0.0, 0.0, 1.2]], obst_bbox=[[0.4, 0.4, 0.5]], cloud_occ=[],
          hgp_cfg=dict(global_planner="astar_heat", heat_weight=10.0, los_cells=2,
                       min_len=0.5, min_turn=10.0),
          start=s5, start_vel=[0.0, 0.0, 0.0], goal=g5, eps=1.0, max_expand=40)

# =====================================================================
# Case 6: nonzero start velocity, eps>1 (weighted A*), w_unknown>0.
# =====================================================================
cfg6 = base_cfg(res=0.3, cells_x=24, cells_y=24, cells_z=8, center_map=[0.0, 0.0, 1.4],
                z_ground=0.0, z_max=2.8, inflation=0.3, traj_max_time=2.0)
vm6 = VoxelMapUtil(res=cfg6["res"])
vm6.read_map(cfg6["cells_x"], cfg6["cells_y"], cfg6["cells_z"],
             np.asarray(cfg6["center_map"]), np.zeros((0, 3)),
             cfg6["z_ground"], cfg6["z_max"], cfg6["inflation"],
             [np.asarray([0.4, 0.4, 1.4])], [np.asarray([0.3, 0.3, 0.4])],
             cfg6["traj_max_time"])
s6 = find_free_world(vm6, prefer=[3, vm6.dim[1] - 4, vm6.dim[2] // 2])
g6 = find_free_world(vm6, prefer=[vm6.dim[0] - 4, 3, vm6.dim[2] // 2])
emit_case(cfg6, obst_pos=[[0.4, 0.4, 1.4]], obst_bbox=[[0.3, 0.3, 0.4]], cloud_occ=[],
          hgp_cfg=dict(global_planner="astar_heat", heat_weight=6.0, w_unknown=0.5,
                       los_cells=3, min_len=0.6, min_turn=20.0),
          start=s6, start_vel=[0.3, -0.2, 0.0], goal=g6, eps=1.5, max_expand=300000)

# =====================================================================
# Case 7: start/goal at non-cell-center world coords (test endpoint override).
# =====================================================================
cfg7 = base_cfg(res=0.3, cells_x=24, cells_y=24, cells_z=8, center_map=[0.0, 0.0, 1.5])
vm7 = VoxelMapUtil(res=cfg7["res"])
vm7.read_map(cfg7["cells_x"], cfg7["cells_y"], cfg7["cells_z"],
             np.asarray(cfg7["center_map"]), np.zeros((0, 3)),
             cfg7["z_ground"], cfg7["z_max"], cfg7["inflation"],
             [np.asarray([0.2, 0.2, 1.5])], [np.asarray([0.25, 0.25, 0.4])],
             cfg7["traj_max_time"])
s7c = find_free_world(vm7, prefer=[4, 4, vm7.dim[2] // 2])
g7c = find_free_world(vm7, prefer=[vm7.dim[0] - 5, vm7.dim[1] - 5, vm7.dim[2] // 2])
# perturb off cell-center by a fraction of res (still in same free cell-ish)
s7 = s7c + np.array([0.07, -0.05, 0.03])
g7 = g7c + np.array([-0.06, 0.04, -0.02])
emit_case(cfg7, obst_pos=[[0.2, 0.2, 1.5]], obst_bbox=[[0.25, 0.25, 0.4]], cloud_occ=[],
          hgp_cfg=dict(global_planner="astar_heat", heat_weight=10.0, los_cells=2,
                       min_len=0.45, min_turn=10.0),
          start=s7, start_vel=[0.0, 0.0, 0.0], goal=g7, eps=1.0, max_expand=300000)

# =====================================================================
# Case 8+: randomized maps + random free start/goal + random HGPPlanner knobs.
# =====================================================================
for trial in range(10):
    res = float(rng.uniform(0.2, 0.35))
    cx = int(rng.integers(18, 30))
    cy = int(rng.integers(18, 30))
    cz = int(rng.integers(6, 12))
    center = [float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)),
              float(rng.uniform(1.0, 1.8))]
    infl = float(rng.uniform(0.15, 0.35))
    tmax = float(rng.uniform(1.5, 3.0))
    nobst = int(rng.integers(1, 4))
    op = []
    ob = []
    for _ in range(nobst):
        op.append([float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0)),
                   float(rng.uniform(1.0, 1.8))])
        ob.append([float(rng.uniform(0.15, 0.4)), float(rng.uniform(0.15, 0.4)),
                   float(rng.uniform(0.2, 0.5))])
    use_soft = bool(rng.integers(0, 2))
    use_heat = bool(rng.integers(0, 2))
    knobs = {
        "use_soft_cost_obstacles": use_soft,
        "heat_alpha0": float(rng.uniform(0.1, 0.5)),
        "heat_alpha1": float(rng.uniform(0.6, 1.4)),
        "heat_p": int(rng.integers(1, 4)),
        "heat_q": int(rng.integers(1, 4)),
        "heat_gamma": float(rng.uniform(0.0, 0.25)),
        "heat_Hmax": float(rng.uniform(1.5, 3.5)),
        "static_heat_rmax_m": float(rng.uniform(0.6, 1.2)),
        "static_heat_default_radius_m": float(rng.uniform(0.4, 0.9)),
        "static_heat_alpha": float(rng.uniform(0.6, 1.6)),
    }
    cfg = dict(res=res, cells_x=cx, cells_y=cy, cells_z=cz, center_map=center,
               z_ground=0.0, z_max=3.0, inflation=infl, traj_max_time=tmax,
               knobs=knobs)
    vm = VoxelMapUtil(res=res)
    for k, v in knobs.items():
        setattr(vm, k, v)
    vm.read_map(cx, cy, cz, np.asarray(center), np.zeros((0, 3)),
                0.0, 3.0, infl, [np.asarray(p) for p in op],
                [np.asarray(b) for b in ob], tmax)
    s = find_free_world(vm, prefer=[2, 2, vm.dim[2] // 2])
    gl = find_free_world(vm, prefer=[vm.dim[0] - 3, vm.dim[1] - 3, vm.dim[2] // 2])
    # off-center perturbation within +-0.3*res so start/goal cell is preserved-ish
    s = s + (rng.uniform(-0.3, 0.3, 3) * res)
    gl = gl + (rng.uniform(-0.3, 0.3, 3) * res)
    eps = float(rng.uniform(1.0, 2.0))
    planner = "astar_heat" if use_heat else "sastar"
    emit_case(cfg, obst_pos=op, obst_bbox=ob, cloud_occ=[],
              hgp_cfg=dict(global_planner=planner,
                           heat_weight=float(rng.uniform(4.0, 12.0)),
                           obstacle_soft_cost=float(rng.uniform(3.0, 7.0)),
                           w_unknown=float(rng.uniform(0.0, 0.5)),
                           los_cells=int(rng.integers(0, 4)),
                           min_len=float(rng.uniform(0.2, 0.7)),
                           min_turn=float(rng.uniform(5.0, 25.0))),
              start=s, start_vel=[float(rng.uniform(-0.3, 0.3)) for _ in range(3)],
              goal=gl, eps=eps, max_expand=300000)

out = os.path.join(os.path.dirname(__file__), "hgp_planner_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len([l for l in lines if l == 'CASE'])} cases)")
