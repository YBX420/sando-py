"""Golden-data generator for GraphSearch (graph_search.py) heat-A* C++ port verification.

不许欺骗、必须还原:这里用真 Python GraphSearch.plan() 跑出「输入 -> 输出」金标准,
C++ 移植后用同样地图+同样起终点跑搜索,得到的格路径(node sequence)必须逐点对上;
对不上就不算还原。

Each case:
  - builds a real VoxelMapUtil via read_map(...) with a few box obstacles (so cmap
    occupancy + heat field are real, identical to what the C++ test rebuilds),
  - runs GraphSearch(...).plan(start_int, goal_int, initial_g, start_vel, max_expand,
    timeout_ms) with a HUGE timeout so the wall-clock deadline never trips
    (deterministic, reproducible in C++),
  - dumps the GraphSearch knobs, start/goal/initial_g, max_expand, the success flag,
    AND the raw integer path (vm.lin_index-coordinate waypoints) returned in self._path.

The C++ test rebuilds the SAME map (reusing the golden voxel_map machinery) and must
reproduce the SAME node sequence exactly.

dump 到 cpp/golden/graph_search_cases.txt(纯文本,C++ 无需 JSON 依赖即可解析)。
跑: python cpp/golden/gen_graph_search_golden.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.hgp.voxel_map import VoxelMapUtil  # noqa: E402
from sando_py.hgp.graph_search import GraphSearch  # noqa: E402

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

# ---- GraphSearch knobs the C++ side must mirror ----
GS_FLOAT = ["eps", "w_unknown", "w_align", "decay_len_cells", "w_side",
            "heat_weight", "obstacle_soft_cost"]


def find_free_cell(vm, prefer=None, avoid=None):
    """Return an in-bounds non-occupied (cmap != VAL_OCC) cell as [x,y,z]."""
    from sando_py.hgp.voxel_map import VAL_OCC
    dimX, dimY, dimZ = int(vm.dim[0]), int(vm.dim[1]), int(vm.dim[2])
    candidates = []
    if prefer is not None:
        candidates.append(prefer)
    # deterministic scan from prefer/center outward, then random
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
        return [x, y, z]
    # fallback: center even if occupied
    return [cx, cy, cz]


def emit_case(vm, cfg, obst_pos, obst_bbox, cloud_occ,
              gs_cfg, start_int, goal_int, initial_g, start_vel,
              max_expand, timeout_ms):
    # apply VoxelMapUtil knob overrides BEFORE read_map
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

    # Build GraphSearch with the requested knobs.
    gs = GraphSearch(
        vm,
        eps=gs_cfg.get("eps", 1.0),
        global_planner=gs_cfg.get("global_planner", "astar_heat"),
        w_unknown=gs_cfg.get("w_unknown", 0.0),
        w_align=gs_cfg.get("w_align", 0.0),
        decay_len_cells=gs_cfg.get("decay_len_cells", 20.0),
        w_side=gs_cfg.get("w_side", 0.2),
        verbose=False,
        heat_weight=gs_cfg.get("heat_weight", 10.0),
        obstacle_soft_cost=gs_cfg.get("obstacle_soft_cost", 5.0),
    )

    start_int = np.asarray(start_int, dtype=np.int64)
    goal_int = np.asarray(goal_int, dtype=np.int64)
    start_vel = np.asarray(start_vel, dtype=np.float64)

    success = gs.plan(start_int, goal_int, float(initial_g), start_vel,
                      max_expand=int(max_expand), timeout_ms=int(timeout_ms))

    # raw integer path = the cell coordinates in self._path order (start..end)
    path_cells = [[int(s.x), int(s.y), int(s.z)] for s in gs._path]
    # world path for an extra cross-check
    path_world = gs.get_path_world()

    lines.append("CASE")
    # ---- VoxelMapUtil rebuild inputs (same schema as voxel_map golden) ----
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
    # predicted times / samples (graph_search cases keep these empty, but carry schema)
    lines.append("PREDTIMES 0")
    lines.append("PREDSAMP -1")
    # ---- GraphSearch config ----
    lines.append("GSPLANNER " + gs_cfg.get("global_planner", "astar_heat"))
    lines.append("GSFLOAT " + fmt([float(getattr(gs, k)) for k in GS_FLOAT]))
    lines.append("START " + fmt_i(start_int))
    lines.append("GOAL " + fmt_i(goal_int))
    lines.append(f"INITG {repr(float(initial_g))}")
    lines.append("STARTVEL " + fmt(start_vel))
    lines.append(f"MAXEXPAND {int(max_expand)}")
    # ---- OUTPUTS ----
    lines.append(f"DIM {fmt_i([dimX, dimY, dimZ])}")
    lines.append(f"SUCCESS {1 if success else 0}")
    lines.append(f"NPATH {len(path_cells)}")
    lines.append("PATH " + (fmt_i([v for c in path_cells for v in c]) if path_cells else ""))
    lines.append("PATHWORLD " + (fmt([v for w in path_world for v in w]) if len(path_world) else ""))
    lines.append("END")


def base_cfg(**kw):
    d = dict(cells_x=20, cells_y=20, cells_z=8, center_map=[0.0, 0.0, 1.5],
             z_ground=0.0, z_max=3.0, inflation=0.3, traj_max_time=2.0, knobs={})
    d.update(kw)
    return d


BIG_TIMEOUT = 10_000_000  # ms — effectively never trips (deterministic search)

# =====================================================================
# Case 1: simple heat-A*, one box obstacle between start and goal
# =====================================================================
vm1 = VoxelMapUtil(res=0.3)
cfg1 = base_cfg(cells_x=24, cells_y=24, cells_z=8, inflation=0.3, traj_max_time=2.0)
# build map first to choose free start/goal
vm1.read_map(cfg1["cells_x"], cfg1["cells_y"], cfg1["cells_z"],
             np.asarray(cfg1["center_map"]), np.zeros((0, 3)),
             cfg1["z_ground"], cfg1["z_max"], cfg1["inflation"],
             [np.asarray([0.5, 0.0, 1.5])], [np.asarray([0.3, 0.3, 0.4])],
             cfg1["traj_max_time"])
s1 = find_free_cell(vm1, prefer=[3, vm1.dim[1] // 2, vm1.dim[2] // 2])
g1 = find_free_cell(vm1, prefer=[vm1.dim[0] - 4, vm1.dim[1] // 2, vm1.dim[2] // 2], avoid=s1)
emit_case(VoxelMapUtil(res=0.3), cfg1,
          obst_pos=[[0.5, 0.0, 1.5]], obst_bbox=[[0.3, 0.3, 0.4]], cloud_occ=[],
          gs_cfg=dict(global_planner="astar_heat", eps=1.0, heat_weight=10.0),
          start_int=s1, goal_int=g1, initial_g=0.0, start_vel=[0.0, 0.0, 0.0],
          max_expand=200000, timeout_ms=BIG_TIMEOUT)

# =====================================================================
# Case 2: weighted A* (eps>1), several boxes, heat on
# =====================================================================
vm2 = VoxelMapUtil(res=0.25)
cfg2 = base_cfg(cells_x=30, cells_y=28, cells_z=10, center_map=[0.2, -0.1, 1.4],
                z_ground=0.1, z_max=2.9, inflation=0.25, traj_max_time=2.5,
                knobs={"heat_gamma": 0.1})
op2 = [[-0.6, 0.4, 1.3], [0.8, -0.5, 1.4], [0.0, 0.0, 1.2]]
ob2 = [[0.25, 0.25, 0.3], [0.3, 0.2, 0.4], [0.2, 0.3, 0.25]]
vm2.heat_gamma = 0.1
vm2.read_map(cfg2["cells_x"], cfg2["cells_y"], cfg2["cells_z"],
             np.asarray(cfg2["center_map"]), np.zeros((0, 3)),
             cfg2["z_ground"], cfg2["z_max"], cfg2["inflation"],
             [np.asarray(p) for p in op2], [np.asarray(b) for b in ob2],
             cfg2["traj_max_time"])
s2 = find_free_cell(vm2, prefer=[2, 2, vm2.dim[2] // 2])
g2 = find_free_cell(vm2, prefer=[vm2.dim[0] - 3, vm2.dim[1] - 3, vm2.dim[2] // 2], avoid=s2)
emit_case(VoxelMapUtil(res=0.25), cfg2,
          obst_pos=op2, obst_bbox=ob2, cloud_occ=[],
          gs_cfg=dict(global_planner="astar_heat", eps=1.5, heat_weight=8.0,
                      obstacle_soft_cost=4.0),
          start_int=s2, goal_int=g2, initial_g=0.0, start_vel=[0.1, 0.2, 0.0],
          max_expand=300000, timeout_ms=BIG_TIMEOUT)

# =====================================================================
# Case 3: plain A* (use_heat False via global_planner="sastar"), hard obstacles
#         (use_soft_cost_obstacles False -> corner-cut + hard block active)
# =====================================================================
vm3 = VoxelMapUtil(res=0.3)
cfg3 = base_cfg(cells_x=22, cells_y=22, cells_z=8, center_map=[0.0, 0.0, 1.2],
                z_ground=0.0, z_max=2.5, inflation=0.3, traj_max_time=2.0,
                knobs={"use_soft_cost_obstacles": False})
op3 = [[0.3, 0.0, 1.2], [-0.4, 0.5, 1.1]]
ob3 = [[0.3, 0.4, 0.5], [0.25, 0.25, 0.4]]
vm3.use_soft_cost_obstacles = False
vm3.read_map(cfg3["cells_x"], cfg3["cells_y"], cfg3["cells_z"],
             np.asarray(cfg3["center_map"]), np.zeros((0, 3)),
             cfg3["z_ground"], cfg3["z_max"], cfg3["inflation"],
             [np.asarray(p) for p in op3], [np.asarray(b) for b in ob3],
             cfg3["traj_max_time"])
s3 = find_free_cell(vm3, prefer=[3, 3, vm3.dim[2] // 2])
g3 = find_free_cell(vm3, prefer=[vm3.dim[0] - 4, vm3.dim[1] - 4, vm3.dim[2] // 2], avoid=s3)
emit_case(VoxelMapUtil(res=0.3), cfg3,
          obst_pos=op3, obst_bbox=ob3, cloud_occ=[],
          gs_cfg=dict(global_planner="sastar", eps=1.0),  # use_heat=False
          start_int=s3, goal_int=g3, initial_g=0.0, start_vel=[0.0, 0.0, 0.0],
          max_expand=300000, timeout_ms=BIG_TIMEOUT)

# =====================================================================
# Case 4: start == goal (trivial path, success)
# =====================================================================
vm4 = VoxelMapUtil(res=0.3)
cfg4 = base_cfg(cells_x=16, cells_y=16, cells_z=6)
vm4.read_map(cfg4["cells_x"], cfg4["cells_y"], cfg4["cells_z"],
             np.asarray(cfg4["center_map"]), np.zeros((0, 3)),
             cfg4["z_ground"], cfg4["z_max"], cfg4["inflation"],
             [], [], cfg4["traj_max_time"])
s4 = find_free_cell(vm4, prefer=[vm4.dim[0] // 2, vm4.dim[1] // 2, vm4.dim[2] // 2])
emit_case(VoxelMapUtil(res=0.3), cfg4,
          obst_pos=[], obst_bbox=[], cloud_occ=[],
          gs_cfg=dict(global_planner="astar_heat", eps=1.0),
          start_int=s4, goal_int=s4, initial_g=0.0, start_vel=[0.0, 0.0, 0.0],
          max_expand=100000, timeout_ms=BIG_TIMEOUT)

# =====================================================================
# Case 5: goal unreachable-ish: max_expand small -> best_node fallback path
# =====================================================================
vm5 = VoxelMapUtil(res=0.3)
cfg5 = base_cfg(cells_x=26, cells_y=26, cells_z=8, center_map=[0.0, 0.0, 1.2],
                z_ground=0.0, z_max=2.5, inflation=0.3, traj_max_time=2.0)
vm5.read_map(cfg5["cells_x"], cfg5["cells_y"], cfg5["cells_z"],
             np.asarray(cfg5["center_map"]), np.zeros((0, 3)),
             cfg5["z_ground"], cfg5["z_max"], cfg5["inflation"],
             [np.asarray([0.0, 0.0, 1.2])], [np.asarray([0.4, 0.4, 0.5])],
             cfg5["traj_max_time"])
s5 = find_free_cell(vm5, prefer=[2, 2, vm5.dim[2] // 2])
g5 = find_free_cell(vm5, prefer=[vm5.dim[0] - 3, vm5.dim[1] - 3, vm5.dim[2] // 2], avoid=s5)
emit_case(VoxelMapUtil(res=0.3), cfg5,
          obst_pos=[[0.0, 0.0, 1.2]], obst_bbox=[[0.4, 0.4, 0.5]], cloud_occ=[],
          gs_cfg=dict(global_planner="astar_heat", eps=1.0, heat_weight=10.0),
          start_int=s5, goal_int=g5, initial_g=0.0, start_vel=[0.0, 0.0, 0.0],
          max_expand=40, timeout_ms=BIG_TIMEOUT)  # tiny -> best_node fallback

# =====================================================================
# Case 6: w_unknown > 0 (penalize unknown cells), nonzero initial_g
# =====================================================================
vm6 = VoxelMapUtil(res=0.3)
cfg6 = base_cfg(cells_x=24, cells_y=24, cells_z=8, center_map=[0.0, 0.0, 1.4],
                z_ground=0.0, z_max=2.8, inflation=0.3, traj_max_time=2.0)
vm6.read_map(cfg6["cells_x"], cfg6["cells_y"], cfg6["cells_z"],
             np.asarray(cfg6["center_map"]), np.zeros((0, 3)),
             cfg6["z_ground"], cfg6["z_max"], cfg6["inflation"],
             [np.asarray([0.4, 0.4, 1.4])], [np.asarray([0.3, 0.3, 0.4])],
             cfg6["traj_max_time"])
s6 = find_free_cell(vm6, prefer=[3, vm6.dim[1] - 4, vm6.dim[2] // 2])
g6 = find_free_cell(vm6, prefer=[vm6.dim[0] - 4, 3, vm6.dim[2] // 2], avoid=s6)
emit_case(VoxelMapUtil(res=0.3), cfg6,
          obst_pos=[[0.4, 0.4, 1.4]], obst_bbox=[[0.3, 0.3, 0.4]], cloud_occ=[],
          gs_cfg=dict(global_planner="astar_heat", eps=1.0, heat_weight=6.0,
                      w_unknown=0.5),
          start_int=s6, goal_int=g6, initial_g=2.5, start_vel=[0.0, 0.0, 0.0],
          max_expand=300000, timeout_ms=BIG_TIMEOUT)

# =====================================================================
# Case 7+: randomized maps + random free start/goal + random GraphSearch knobs
# =====================================================================
for trial in range(8):
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
    cfg = dict(cells_x=cx, cells_y=cy, cells_z=cz, center_map=center,
               z_ground=0.0, z_max=3.0, inflation=infl, traj_max_time=tmax,
               knobs=knobs)
    vm = VoxelMapUtil(res=res)
    for k, v in knobs.items():
        setattr(vm, k, v)
    vm.read_map(cx, cy, cz, np.asarray(center), np.zeros((0, 3)),
                0.0, 3.0, infl, [np.asarray(p) for p in op],
                [np.asarray(b) for b in ob], tmax)
    s = find_free_cell(vm, prefer=[2, 2, vm.dim[2] // 2])
    gl = find_free_cell(vm, prefer=[vm.dim[0] - 3, vm.dim[1] - 3, vm.dim[2] // 2], avoid=s)
    eps = float(rng.uniform(1.0, 2.0))
    planner = "astar_heat" if use_heat else "sastar"
    emit_case(VoxelMapUtil(res=res), cfg,
              obst_pos=op, obst_bbox=ob, cloud_occ=[],
              gs_cfg=dict(global_planner=planner, eps=eps,
                          heat_weight=float(rng.uniform(4.0, 12.0)),
                          obstacle_soft_cost=float(rng.uniform(3.0, 7.0)),
                          w_unknown=float(rng.uniform(0.0, 0.5))),
              start_int=s, goal_int=gl, initial_g=float(rng.uniform(0.0, 1.0)),
              start_vel=[float(rng.uniform(-0.3, 0.3)) for _ in range(3)],
              max_expand=300000, timeout_ms=BIG_TIMEOUT)

out = os.path.join(os.path.dirname(__file__), "graph_search_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len([l for l in lines if l == 'CASE'])} cases)")
