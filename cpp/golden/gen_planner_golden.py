"""Golden-data generator for the SANDO orchestrator C++ port (planner.hpp),
MINCO PATH ONLY.

不许欺骗、必须还原: drives the REAL Python sando_py.planner.SANDO over diverse +
random + edge scenarios. Two kinds of golden are written:

  (A) DETERMINISTIC methods — golden-tested EXACTLY (no optimiser involved):
        COMPUTE_G    : compute_G(A, G_term, horizon) -> projected G pos + yaw
        FINDA        : find_A_and_Atime(k_value logic) -> (ok, A.pos, A_time, k_value)
        SAFESUB      : find_safe_sub_goal(unknown-cloud truncation) -> path points
        OBSTTIME     : _compute_obst_pos_and_traj_max_time -> Th + per-class snapshot
        OBSTSNAP     : _obstacles_from_snapshot -> per-obstacle (kind, centre, r/lo/hi, vel, accel, class)
        MJ2PWP       : _minjerk_to_pwp eval-equivalence (pos/vel/accel/jerk vs MinjerkTraj)
        APPEND       : append_to_plan + get_next_goal queue splice -> final deque + popped goal
        NEEDREPLAN   : need_replan state-machine transitions -> (bool, status_before, status_after)
        GLOBALPATH   : generate_global_path -> deterministic processed path (via solve_hgp)

  (B) REPLAN END-TO-END — NOT bit-exact (the local trajectory inherits plan_minco's
        optimiser non-determinism: LBFGSpp != scipy on the nonconvex ALM). We dump the
        full scenario + Python's committed plan tail (sampled positions) + flags and
        the C++ test REPORTS the true sampled-position max error honestly. We DO assert
        the WIRING exactly (planned_ok / attempted flags, plan length bookkeeping).

Run: python cpp/golden/gen_planner_golden.py
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sando_py.planner import SANDO, _minjerk_to_pwp
from sando_py.types import Parameters, RobotState, DynTraj, DroneStatus
from sando_py.local.minco import MinjerkTraj

rng = np.random.default_rng(20260603)

lines = []


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


def mk_par(**kw):
    p = Parameters()
    # default to the rviz_only / minco path so generate_global_path is deterministic
    p.sim_env = "rviz_only"
    p.local_solver = "minco"
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def mk_state(pos=(0, 0, 0), vel=(0, 0, 0), acc=(0, 0, 0), jerk=(0, 0, 0), yaw=0.0):
    s = RobotState()
    s.pos = np.array(pos, dtype=float)
    s.vel = np.array(vel, dtype=float)
    s.accel = np.array(acc, dtype=float)
    s.jerk = np.array(jerk, dtype=float)
    s.yaw = float(yaw)
    return s


def emit_par(p):
    # only the parameters the C++ test re-creates explicitly. The C++ test reads the
    # named subset; defaults for everything else come from Parameters() == types.hpp.
    pass


# ===========================================================================
# (A1) COMPUTE_G — compute_G(A, G_term, horizon)
# ===========================================================================
def compute_g_case(name, a_pos, gterm_pos, horizon):
    p = mk_par(horizon=horizon)
    s = SANDO(p)
    A = mk_state(pos=a_pos)
    Gt = mk_state(pos=gterm_pos)
    s.compute_G(A, Gt, horizon)
    G = s.get_G()
    lines.append("CASE")
    lines.append("KIND COMPUTE_G")
    lines.append(f"NAME {name}")
    lines.append(f"HORIZON {repr(float(horizon))}")
    lines.append(f"A_POS {fmt(a_pos)}")
    lines.append(f"GTERM_POS {fmt(gterm_pos)}")
    lines.append(f"G_POS {fmt(G.pos)}")
    lines.append(f"G_YAW {repr(float(G.yaw))}")
    lines.append("END")


compute_g_case("g_inside", [0, 0, 0], [3, 0, 0], 15.0)       # goal within horizon
compute_g_case("g_far_x", [0, 0, 0], [30, 0, 0], 15.0)       # far -> project to sphere
compute_g_case("g_far_diag", [1, 2, 3], [20, -15, 8], 10.0)
compute_g_case("g_same", [5, 5, 5], [5, 5, 5], 15.0)         # degenerate (n<1e-9 -> yaw stays 0)
compute_g_case("g_neg", [-2, -3, 1], [-25, 4, -7], 12.0)
for ridx in range(10):
    a = rng.uniform(-10, 10, 3)
    g = rng.uniform(-40, 40, 3)
    h = float(rng.uniform(5, 20))
    compute_g_case(f"g_fuzz{ridx}", a, g, h)


# ===========================================================================
# (A2) FINDA — find_A_and_Atime (k_value logic)
# ===========================================================================
def finda_case(name, plan_pts, current_time, last_comp, *, use_adapt=False,
               num_replanning=0, est_comp=0.0, par_kw=None):
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    # build the plan deque
    from collections import deque
    s.plan = deque(mk_state(pos=pt) for pt in plan_pts)
    s.use_adapt_k_value = use_adapt
    s.num_replanning = num_replanning
    s.est_comp_time = est_comp
    ok, A, A_time = s.find_A_and_Atime(current_time, last_comp)
    lines.append("CASE")
    lines.append("KIND FINDA")
    lines.append(f"NAME {name}")
    lines.append(f"USE_STATE_UPDATE {1 if p.use_state_update else 0}")
    lines.append(f"USE_ADAPT {1 if use_adapt else 0}")
    lines.append(f"NUM_REPLANNING {int(num_replanning)}")
    lines.append(f"EST_COMP {repr(float(est_comp))}")
    lines.append(f"DEFAULT_K {int(p.default_k_value)}")
    lines.append(f"ALPHA_K {repr(float(p.alpha_k_value_filtering))}")
    lines.append(f"K_VALUE_FACTOR {repr(float(p.k_value_factor))}")
    lines.append(f"DC {repr(float(p.dc))}")
    lines.append(f"ZMIN {repr(float(p.z_min))}")
    lines.append(f"ZMAX {repr(float(p.z_max))}")
    lines.append(f"XMIN {repr(float(p.x_min))}")
    lines.append(f"XMAX {repr(float(p.x_max))}")
    lines.append(f"YMIN {repr(float(p.y_min))}")
    lines.append(f"YMAX {repr(float(p.y_max))}")
    lines.append(f"NPLAN {len(plan_pts)}")
    lines.append(f"PLAN {fmt(np.array(plan_pts))}" if plan_pts else "PLAN")
    lines.append(f"CURRENT_TIME {repr(float(current_time))}")
    lines.append(f"LAST_COMP {repr(float(last_comp))}")
    lines.append(f"OK {1 if ok else 0}")
    lines.append(f"A_POS {fmt(A.pos)}")
    lines.append(f"A_TIME {repr(float(A_time))}")
    lines.append(f"K_VALUE {int(s.k_value)}")
    lines.append("END")


# small plan -> k_value = max(plan_size - default_k_value, 0) = 0; index out of range guard
finda_case("finda_small", [[i * 0.5, 0, 2.0] for i in range(5)], 1.0, 0.0)
# default_k smaller than plan
finda_case("finda_default_k", [[i * 0.1, 0, 2.0] for i in range(80)], 2.0, 0.0)
# adaptive k_value
finda_case("finda_adapt", [[i * 0.1, 0, 2.0] for i in range(120)], 0.5, 0.02,
           use_adapt=True, est_comp=0.01)
# A out of map (z too high)
finda_case("finda_oob_z", [[0, 0, 10.0]], 0.0, 0.0)
# empty plan
finda_case("finda_empty", [], 0.0, 0.0)
# num_replanning == 1 (skip store)
finda_case("finda_nr1", [[i * 0.2, 0, 2.0] for i in range(30)], 1.0, 0.05,
           num_replanning=1)
for ridx in range(10):
    n = int(rng.integers(1, 150))
    z = float(rng.uniform(1.0, 5.0))
    pts = [[i * 0.1, float(rng.uniform(-2, 2)), z] for i in range(n)]
    ct = float(rng.uniform(0, 5))
    lc = float(rng.uniform(0, 0.1))
    adapt = bool(rng.random() < 0.5)
    finda_case(f"finda_fuzz{ridx}", pts, ct, lc, use_adapt=adapt,
               est_comp=float(rng.uniform(0, 0.05)),
               num_replanning=int(rng.integers(0, 3)))


# ===========================================================================
# (A3) SAFESUB — find_safe_sub_goal (unknown-cloud truncation)
# ===========================================================================
def safesub_case(name, path, unk_cloud, traj_max_time, *, par_kw=None):
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    s.traj_max_time = float(traj_max_time)
    unk = np.asarray(unk_cloud, dtype=float).reshape(-1, 3) if len(unk_cloud) else np.zeros((0, 3))
    s.pclptr_unk = unk
    s._build_kdtree_unk(unk)
    out = s.find_safe_sub_goal([np.asarray(pt, dtype=float) for pt in path])
    lines.append("CASE")
    lines.append("KIND SAFESUB")
    lines.append(f"NAME {name}")
    lines.append(f"DRONE_RADIUS {repr(float(p.drone_radius))}")
    lines.append(f"OBST_MAX_VEL {repr(float(p.obst_max_vel))}")
    lines.append(f"TRAJ_MAX_TIME {repr(float(traj_max_time))}")
    lines.append(f"NPATH {len(path)}")
    lines.append(f"PATH {fmt(np.array(path))}" if path else "PATH")
    lines.append(f"NUNK {unk.shape[0]}")
    lines.append(f"UNK {fmt(unk)}" if unk.shape[0] else "UNK")
    lines.append(f"NOUT {len(out)}")
    lines.append(f"OUT {fmt(np.array(out))}" if out else "OUT")
    lines.append("END")


# no unknown cloud -> whole path passes through
straight = [[0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2], [4, 0, 2]]
safesub_case("safesub_no_unk", straight, [], 0.5)
# unknown blob ahead -> truncates
blob = [[2.5, 0.05, 2.0]]
safesub_case("safesub_blob", straight, blob, 0.5)
# unknown right at start (out[0] kept, returns immediately)
safesub_case("safesub_unk_start", straight, [[0.05, 0.0, 2.0]], 0.5)
# inflate radius large (traj_max_time big) -> backtrack further
safesub_case("safesub_big_inflate", straight, [[3.05, 0.0, 2.0]], 3.0)
# single point path
safesub_case("safesub_single", [[1, 1, 2]], [[5, 5, 5]], 0.5)
# empty path
safesub_case("safesub_empty", [], [[0, 0, 0]], 0.5)
# 3D diagonal with mid blob
diag = [[0, 0, 1], [1, 1, 1.5], [2, 2, 2], [3, 3, 2.5], [4, 4, 3]]
safesub_case("safesub_diag", diag, [[2.05, 2.05, 2.0]], 0.5)
for ridx in range(10):
    n = int(rng.integers(3, 8))
    t = np.linspace(0, float(rng.uniform(3, 8)), n)
    path = np.stack([t, 0.3 * np.sin(t), 2.0 + 0.0 * t], axis=1)
    nunk = int(rng.integers(0, 4))
    unk = []
    for _ in range(nunk):
        idx = int(rng.integers(1, n))
        unk.append(path[idx] + rng.uniform(-0.15, 0.15, 3))
    safesub_case(f"safesub_fuzz{ridx}", path.tolist(), unk,
                 float(rng.uniform(0.2, 2.0)),
                 par_kw={"drone_radius": float(rng.uniform(0.1, 0.4)),
                         "obst_max_vel": float(rng.uniform(0.2, 0.8))})


# ===========================================================================
# (A4) OBSTTIME — _compute_obst_pos_and_traj_max_time
# ===========================================================================
def make_traj(tid, x_expr, y_expr, z_expr, bbox=(0.5, 0.5, 0.5), vx=None, vy=None, vz=None):
    t = DynTraj()
    t.mode = "Analytic"
    t.traj_x = x_expr
    t.traj_y = y_expr
    t.traj_z = z_expr
    if vx is not None:
        t.traj_vx, t.traj_vy, t.traj_vz = vx, vy, vz
    t.bbox = np.array(bbox, dtype=float)
    t.id = int(tid)
    return t


def emit_traj(prefix, t):
    lines.append(f"{prefix}_ID {int(t.id)}")
    lines.append(f"{prefix}_MODE {t.mode}")
    lines.append(f"{prefix}_TX {t.traj_x}")
    lines.append(f"{prefix}_TY {t.traj_y}")
    lines.append(f"{prefix}_TZ {t.traj_z}")
    lines.append(f"{prefix}_TVX {t.traj_vx if t.traj_vx else '_'}")
    lines.append(f"{prefix}_TVY {t.traj_vy if t.traj_vy else '_'}")
    lines.append(f"{prefix}_TVZ {t.traj_vz if t.traj_vz else '_'}")
    lines.append(f"{prefix}_BBOX {fmt(t.bbox)}")
    lines.append(f"{prefix}_TIME_RECEIVED {repr(float(t.time_received))}")


def obsttime_case(name, trajs, state_pos, current_time, worst_traj_time,
                  *, map_init=True, par_kw=None):
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    s.state = mk_state(pos=state_pos)
    s.worst_traj_time = float(worst_traj_time)
    # mark map as seen so check_point_within_map applies the membership test
    if map_init:
        s.map_size_initialized = True
        s._map_seen = True
        s.map_center = np.array(state_pos, dtype=float)
        s.wdx = 40.0; s.wdy = 40.0; s.wdz = 20.0
    s.trajs = list(trajs)
    opos, obbox, preds, ptimes = [], [], [], []
    Th = s._compute_obst_pos_and_traj_max_time(opos, obbox, preds, ptimes, current_time)
    lines.append("CASE")
    lines.append("KIND OBSTTIME")
    lines.append(f"NAME {name}")
    lines.append(f"STATE_POS {fmt(state_pos)}")
    lines.append(f"CURRENT_TIME {repr(float(current_time))}")
    lines.append(f"WORST_TRAJ_TIME {repr(float(worst_traj_time))}")
    lines.append(f"HORIZON {repr(float(p.horizon))}")
    lines.append(f"MAP_INIT {1 if map_init else 0}")
    lines.append(f"MAP_CENTER {fmt(s.map_center)}")
    lines.append(f"WDX {repr(float(s.wdx))}")
    lines.append(f"WDY {repr(float(s.wdy))}")
    lines.append(f"WDZ {repr(float(s.wdz))}")
    # the last factor (drives Th = worst_traj_time * factors[-1])
    lines.append(f"FACTOR_LAST {repr(float(s.factors[-1] if s.factors else 1.0))}")
    lines.append(f"NTRAJ {len(trajs)}")
    for i, t in enumerate(trajs):
        emit_traj(f"TRAJ{i}", t)
    lines.append(f"TH {repr(float(Th))}")
    lines.append(f"NSEL {len(opos)}")
    lines.append(f"OPOS {fmt(np.array(opos))}" if opos else "OPOS")
    lines.append(f"OBBOX {fmt(np.array(obbox))}" if obbox else "OBBOX")
    lines.append(f"OCLASS {' '.join(s.obst_class)}" if s.obst_class else "OCLASS")
    lines.append(f"OVEL {fmt(np.array(s.obst_vel))}" if s.obst_vel else "OVEL")
    lines.append(f"OACCEL {fmt(np.array(s.obst_accel))}" if s.obst_accel else "OACCEL")
    lines.append(f"NPTIMES {len(ptimes)}")
    lines.append(f"PTIMES {fmt(np.array(ptimes))}" if ptimes else "PTIMES")
    # predicted samples per selected obstacle
    lines.append(f"NPRED {len(preds)}")
    for i, pk in enumerate(preds):
        lines.append(f"PRED{i} {fmt(np.array(pk))}")
    lines.append("END")


# human (id<200), moving linearly; worst_traj_time>0 -> predicted samples produced
h1 = make_traj(5, "1.0 + 0.5*t", "0.0", "2.0")
obsttime_case("obsttime_human_lin", [h1], [0, 0, 2], 0.0, 1.5)
# wall (200<=id<300), static
w1 = make_traj(210, "3.0", "1.0", "2.0", bbox=(1.0, 0.5, 2.0))
obsttime_case("obsttime_wall_static", [w1], [0, 0, 2], 0.0, 1.5)
# mixed, one far (filtered out by horizon)
h2 = make_traj(7, "2.0", "0.0 + 0.3*t", "2.0")
far = make_traj(8, "100.0", "0.0", "2.0")
obsttime_case("obsttime_mixed", [h1, w1, h2, far], [0, 0, 2], 0.5, 1.0)
# worst_traj_time == 0 -> Th==0 -> no predicted samples
obsttime_case("obsttime_no_pred", [h1], [0, 0, 2], 0.0, 0.0)
# nonlinear motion (sin) -> velocity/accel via central difference
hsin = make_traj(9, "2.0 + sin(t)", "cos(t)", "2.0")
obsttime_case("obsttime_sin", [hsin], [0, 0, 2], 0.3, 2.0)
# analytic with explicit velocity expressions
hv = make_traj(11, "1.0 + t", "0.5*t", "2.0", vx="1.0", vy="0.5", vz="0.0")
obsttime_case("obsttime_expl_vel", [hv], [0, 0, 2], 0.7, 1.2)


# ===========================================================================
# (A5) OBSTSNAP — _obstacles_from_snapshot
# ===========================================================================
def obstsnap_case(name, opos, obbox, oclass, ovel, oaccel):
    p = mk_par()
    s = SANDO(p)
    obstacles, cfg = s._obstacles_from_snapshot(
        [np.asarray(x, float) for x in opos],
        [np.asarray(x, float) for x in obbox],
        list(oclass),
        [np.asarray(x, float) for x in ovel],
        [np.asarray(x, float) for x in oaccel])
    lines.append("CASE")
    lines.append("KIND OBSTSNAP")
    lines.append(f"NAME {name}")
    lines.append(f"N {len(opos)}")
    for i in range(len(opos)):
        lines.append(f"IN{i}_POS {fmt(opos[i])}")
        lines.append(f"IN{i}_BBOX {fmt(obbox[i])}")
        lines.append(f"IN{i}_CLASS {oclass[i] if i < len(oclass) else '_'}")
        lines.append(f"IN{i}_VEL {fmt(ovel[i]) if i < len(ovel) else '0 0 0'}")
        lines.append(f"IN{i}_ACCEL {fmt(oaccel[i]) if i < len(oaccel) else '0 0 0'}")
    lines.append(f"NOUT {len(obstacles)}")
    for i, o in enumerate(obstacles):
        if hasattr(o, "centre0"):
            lines.append(f"OUT{i}_KIND sphere")
            lines.append(f"OUT{i}_C0 {fmt(o.centre0)}")
            lines.append(f"OUT{i}_R {repr(float(o.radius))}")
            lines.append(f"OUT{i}_VEL {fmt(o.vel)}")
            lines.append(f"OUT{i}_ACC {fmt(o.accel)}")
            lines.append(f"OUT{i}_CLASS {o.class_name}")
        else:
            lines.append(f"OUT{i}_KIND aabb")
            lines.append(f"OUT{i}_LO {fmt(o.lo)}")
            lines.append(f"OUT{i}_HI {fmt(o.hi)}")
            lines.append(f"OUT{i}_CLASS {o.class_name}")
    lines.append("END")


obstsnap_case("snap_human",
              [[2.0, 0.0, 2.0]], [[0.6, 0.6, 1.8]], ["human"],
              [[0.3, 0.0, 0.0]], [[0.0, 0.1, 0.0]])
obstsnap_case("snap_wall",
              [[3.0, 1.0, 2.0]], [[1.0, 0.4, 2.0]], ["wall"],
              [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
obstsnap_case("snap_missing_class",   # missing tag -> human (fail-safe)
              [[1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5]], [],
              [], [])
obstsnap_case("snap_mixed",
              [[2.0, 0.0, 2.0], [4.0, 1.0, 2.0], [0.0, 3.0, 1.0]],
              [[0.6, 0.6, 1.8], [1.0, 0.4, 2.0], [0.5, 0.5, 0.5]],
              ["human", "wall", "human"],
              [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.4, 0.0]],
              [[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
for ridx in range(8):
    n = int(rng.integers(1, 5))
    opos = [rng.uniform(-4, 4, 3) for _ in range(n)]
    obbox = [rng.uniform(0.3, 2.0, 3) for _ in range(n)]
    oclass = ["wall" if rng.random() < 0.5 else "human" for _ in range(n)]
    ovel = [rng.normal(0, 0.3, 3) for _ in range(n)]
    oaccel = [rng.normal(0, 0.15, 3) for _ in range(n)]
    obstsnap_case(f"snap_fuzz{ridx}", opos, obbox, oclass, ovel, oaccel)


# ===========================================================================
# (A6) MJ2PWP — _minjerk_to_pwp eval-equivalence
# ===========================================================================
def mj2pwp_case(name, start, goal, q, T, A_time, v0=None, a0=None):
    tr = MinjerkTraj.from_endpoints(np.array(start, float), np.array(goal, float),
                                    np.array(q, float).reshape(-1, 3) if len(q) else np.zeros((0, 3)),
                                    np.array(T, float),
                                    v0=np.array(v0, float) if v0 is not None else None,
                                    a0=np.array(a0, float) if a0 is not None else None)
    pwp = _minjerk_to_pwp(tr, A_time)
    # sample at NSAMP local times and dump pwp pos/vel/accel + mj pos/vel/accel
    NSAMP = 25
    ts = np.linspace(0.0, float(tr.t_end), NSAMP)
    lines.append("CASE")
    lines.append("KIND MJ2PWP")
    lines.append(f"NAME {name}")
    lines.append(f"M {tr.M}")
    lines.append(f"A_TIME {repr(float(A_time))}")
    lines.append(f"WP {fmt(tr.waypoints)}")
    lines.append(f"T {fmt(tr.T)}")
    lines.append(f"V0 {fmt(tr.v0)}")
    lines.append(f"A0 {fmt(tr.a0)}")
    lines.append(f"VF {fmt(tr.vf)}")
    lines.append(f"AF {fmt(tr.af)}")
    # pwp segment boundary times (absolute) and per-segment ascending coeffs
    lines.append(f"NTIMES {len(pwp.times)}")
    lines.append(f"TIMES {fmt(np.array(pwp.times))}")
    cx = np.array([np.asarray(c).reshape(-1) for c in pwp.coeff_x]).reshape(-1)
    cy = np.array([np.asarray(c).reshape(-1) for c in pwp.coeff_y]).reshape(-1)
    cz = np.array([np.asarray(c).reshape(-1) for c in pwp.coeff_z]).reshape(-1)
    lines.append(f"COEFF_X {fmt(cx)}")
    lines.append(f"COEFF_Y {fmt(cy)}")
    lines.append(f"COEFF_Z {fmt(cz)}")
    lines.append(f"NSAMP {NSAMP}")
    # absolute sample times (= A_time + local)
    abst = [A_time + float(tt) for tt in ts]
    lines.append(f"SAMP_T {fmt(np.array(abst))}")
    pos = np.array([pwp.eval(t) for t in abst])
    vel = np.array([pwp.velocity(t) for t in abst])
    acc = np.array([pwp.acceleration(t) for t in abst])
    lines.append(f"PWP_POS {fmt(pos)}")
    lines.append(f"PWP_VEL {fmt(vel)}")
    lines.append(f"PWP_ACC {fmt(acc)}")
    lines.append("END")


mj2pwp_case("mj_2seg", [0, 0, 0], [3, 0, 0], [[1.5, 0, 0]], [1.0, 1.0], 0.0)
mj2pwp_case("mj_3seg_atime", [0, 0, 1], [4, 1, 2], [[1, 0.5, 1.2], [2.5, 0.8, 1.6]],
            [0.8, 0.9, 1.1], 7.35)
mj2pwp_case("mj_1seg", [0, 0, 0], [2, 1, 0], [], [1.3], 2.0)
mj2pwp_case("mj_v0a0", [0, 0, 2], [5, 0, 2], [[2.5, 0.3, 2]], [1.0, 1.0], 3.0,
            v0=[1.0, 0.2, 0.0], a0=[0.5, 0.0, 0.0])
for ridx in range(8):
    M = int(rng.integers(1, 6))
    start = rng.uniform(-3, 3, 3); goal = rng.uniform(-3, 3, 3)
    q = rng.uniform(-3, 3, (M - 1, 3)) if M > 1 else np.zeros((0, 3))
    T = rng.uniform(0.5, 1.5, M)
    at = float(rng.uniform(0, 10))
    mj2pwp_case(f"mj_fuzz{ridx}", start, goal, q.tolist(), T.tolist(), at,
                v0=rng.normal(0, 0.3, 3).tolist(), a0=rng.normal(0, 0.2, 3).tolist())


# ===========================================================================
# (A7) APPEND — append_to_plan + get_next_goal queue splice
# ===========================================================================
def append_case(name, plan_pts, setpoints_pts, k_value, *, par_kw=None):
    from collections import deque
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    s.plan = deque(mk_state(pos=pt) for pt in plan_pts)
    s.goal_setpoints = [mk_state(pos=pt) for pt in setpoints_pts]
    s.k_value = int(k_value)
    s.got_enough_replanning = True  # skip the adapt bookkeeping branch
    ok = s.append_to_plan()
    final_plan = [st.pos.copy() for st in s.plan]
    lines.append("CASE")
    lines.append("KIND APPEND")
    lines.append(f"NAME {name}")
    lines.append(f"NPLAN {len(plan_pts)}")
    lines.append(f"PLAN {fmt(np.array(plan_pts))}" if plan_pts else "PLAN")
    lines.append(f"NSET {len(setpoints_pts)}")
    lines.append(f"SET {fmt(np.array(setpoints_pts))}" if setpoints_pts else "SET")
    lines.append(f"K_VALUE_IN {int(k_value)}")
    lines.append(f"OK {1 if ok else 0}")
    lines.append(f"K_VALUE_OUT {int(s.k_value)}")
    lines.append(f"NFINAL {len(final_plan)}")
    lines.append(f"FINAL {fmt(np.array(final_plan))}" if final_plan else "FINAL")
    lines.append("END")


append_case("append_normal",
            [[i * 0.1, 0, 2] for i in range(20)],
            [[2.0 + i * 0.1, 0, 2] for i in range(10)], 5)
append_case("append_k_zero",
            [[i * 0.1, 0, 2] for i in range(10)],
            [[1.0 + i * 0.1, 0, 2] for i in range(8)], 0)
append_case("append_k_exceeds",   # plan_size < k_value -> k_value = max(1, plan_size-1)
            [[i * 0.1, 0, 2] for i in range(3)],
            [[0.3 + i * 0.1, 0, 2] for i in range(5)], 10)
append_case("append_k_all",       # k_value == plan_size
            [[i * 0.1, 0, 2] for i in range(6)],
            [[i * 0.1, 0, 2] for i in range(6)], 6)
for ridx in range(8):
    np_ = int(rng.integers(2, 30))
    ns = int(rng.integers(1, 15))
    k = int(rng.integers(0, np_ + 3))
    plan_pts = [[i * 0.1, float(rng.uniform(-1, 1)), 2.0] for i in range(np_)]
    set_pts = [[float(rng.uniform(0, 3)), float(rng.uniform(-1, 1)), 2.0] for _ in range(ns)]
    append_case(f"append_fuzz{ridx}", plan_pts, set_pts, k)


# ===========================================================================
# (A8) NEEDREPLAN — need_replan state-machine transitions
# ===========================================================================
def needreplan_case(name, status_before, state_pos, state_vel, gterm_pos,
                    last_plan_pos, *, par_kw=None):
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    s._drone_status = status_before
    ls = mk_state(pos=state_pos, vel=state_vel)
    gt = mk_state(pos=gterm_pos)
    lps = mk_state(pos=last_plan_pos)
    res = s.need_replan(ls, gt, lps)
    lines.append("CASE")
    lines.append("KIND NEEDREPLAN")
    lines.append(f"NAME {name}")
    lines.append(f"STATUS_BEFORE {int(status_before)}")
    lines.append(f"HOVER_ENABLED {1 if p.hover_avoidance_enabled else 0}")
    lines.append(f"GOAL_RADIUS {repr(float(p.goal_radius))}")
    lines.append(f"GOAL_SEEN_RADIUS {repr(float(p.goal_seen_radius))}")
    lines.append(f"STATE_POS {fmt(state_pos)}")
    lines.append(f"STATE_VEL {fmt(state_vel)}")
    lines.append(f"GTERM_POS {fmt(gterm_pos)}")
    lines.append(f"LAST_PLAN_POS {fmt(last_plan_pos)}")
    lines.append(f"RESULT {1 if res else 0}")
    lines.append(f"STATUS_AFTER {int(s._drone_status)}")
    lines.append("END")


# at goal, low velocity -> GOAL_REACHED, returns False
needreplan_case("nr_reached", DroneStatus.TRAVELING, [0.1, 0, 2], [0.01, 0, 0],
                [0.0, 0, 2], [0.0, 0, 2])
# traveling, far from goal -> True
needreplan_case("nr_traveling", DroneStatus.TRAVELING, [0, 0, 2], [1.0, 0, 0],
                [10, 0, 2], [5, 0, 2])
# within goal_seen_radius -> GOAL_SEEN
needreplan_case("nr_seen", DroneStatus.TRAVELING, [0, 0, 2], [1.0, 0, 0],
                [1.5, 0, 2], [0.5, 0, 2])
# GOAL_SEEN + last plan near goal -> False
needreplan_case("nr_seen_done", DroneStatus.GOAL_SEEN, [0, 0, 2], [1.0, 0, 0],
                [1.5, 0, 2], [1.4, 0, 2])
# YAWING -> False (not traveling)
needreplan_case("nr_yawing", DroneStatus.YAWING, [0, 0, 2], [0, 0, 0],
                [10, 0, 2], [0, 0, 2])
# GOAL_REACHED far again -> False (not traveling)
needreplan_case("nr_reached_far", DroneStatus.GOAL_REACHED, [0, 0, 2], [0, 0, 0],
                [10, 0, 2], [0, 0, 2])
# at goal but high velocity -> not reached, continue
needreplan_case("nr_at_goal_fast", DroneStatus.TRAVELING, [0.1, 0, 2], [2.0, 0, 0],
                [0.0, 0, 2], [0.0, 0, 2])
# hover enabled + GOAL_REACHED -> True
needreplan_case("nr_hover", DroneStatus.GOAL_REACHED, [0, 0, 2], [0, 0, 0],
                [0, 0, 2], [0, 0, 2], par_kw={"hover_avoidance_enabled": True})
for ridx in range(8):
    sb = [DroneStatus.TRAVELING, DroneStatus.GOAL_SEEN, DroneStatus.YAWING,
          DroneStatus.GOAL_REACHED][int(rng.integers(0, 4))]
    sp = rng.uniform(-5, 5, 3); sp[2] = 2.0
    gp = rng.uniform(-5, 5, 3); gp[2] = 2.0
    lp = rng.uniform(-5, 5, 3); lp[2] = 2.0
    sv = rng.normal(0, 1.0, 3)
    needreplan_case(f"nr_fuzz{ridx}", sb, sp.tolist(), sv.tolist(),
                    gp.tolist(), lp.tolist())


# ===========================================================================
# (A9) GLOBALPATH — generate_global_path deterministic processed path
# ===========================================================================
def globalpath_case(name, state_pos, gterm_pos, occ_cloud, current_time,
                    *, par_kw=None):
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    # initialize state + goal so the planner is ready
    s.update_state(mk_state(pos=state_pos))
    occ = np.asarray(occ_cloud, dtype=float).reshape(-1, 3) if len(occ_cloud) else np.zeros((0, 3))
    s.update_occupancy_map_ptr(occ)
    s.set_terminal_goal(mk_state(pos=gterm_pos))
    # drive to TRAVELING so generate_global_path proceeds
    s.change_drone_status(DroneStatus.TRAVELING)
    ok, gp = s.generate_global_path(current_time, 0.0)
    lines.append("CASE")
    lines.append("KIND GLOBALPATH")
    lines.append(f"NAME {name}")
    lines.append(f"WORST_TRAJ_TIME {repr(float(s.worst_traj_time))}")
    lines.append(f"STATE_POS {fmt(state_pos)}")
    lines.append(f"GTERM_POS {fmt(gterm_pos)}")
    lines.append(f"CURRENT_TIME {repr(float(current_time))}")
    lines.append(f"NOCC {occ.shape[0]}")
    lines.append(f"OCC {fmt(occ)}" if occ.shape[0] else "OCC")
    lines.append(f"OK {1 if ok else 0}")
    lines.append(f"NGP {len(gp)}")
    lines.append(f"GP {fmt(np.array(gp))}" if gp else "GP")
    # also dump A / G / A_time set as side effects
    lines.append(f"A_POS {fmt(s.get_A().pos)}")
    lines.append(f"A_TIME {repr(float(s.get_A_time()))}")
    lines.append(f"G_POS {fmt(s.get_G().pos)}")
    lines.append(f"FINAL_G {repr(float(s.final_g))}")
    lines.append("END")


# empty map -> straight global path
globalpath_case("gp_empty", [0, 0, 2], [6, 0, 2], [], 0.0)
# a small occupied wall between start and goal
wall_pts = []
for yy in np.linspace(-1.0, 1.0, 7):
    for zz in np.linspace(1.5, 2.5, 4):
        wall_pts.append([3.0, float(yy), float(zz)])
globalpath_case("gp_wall", [0, 0, 2], [6, 0, 2], wall_pts, 0.0)
# diagonal goal, empty
globalpath_case("gp_diag", [0, 0, 2], [5, 4, 3], [], 0.0)
for ridx in range(6):
    sp = [0.0, 0.0, 2.0]
    gp = [float(rng.uniform(4, 8)), float(rng.uniform(-3, 3)), 2.0]
    occ = []
    if rng.random() < 0.6:
        cx = float(rng.uniform(2, 4))
        for yy in np.linspace(-1.0, 1.0, 5):
            occ.append([cx, float(yy), 2.0])
    globalpath_case(f"gp_fuzz{ridx}", sp, gp, occ, 0.0)


# ===========================================================================
# (B) REPLAN — end-to-end (NOT bit-exact local; WIRING asserted exactly)
# ===========================================================================
NSAMP_REPLAN = 40


def replan_case(name, state_pos, gterm_pos, occ_cloud, trajs, *, par_kw=None):
    p = mk_par(**(par_kw or {}))
    s = SANDO(p)
    s.update_state(mk_state(pos=state_pos))
    occ = np.asarray(occ_cloud, dtype=float).reshape(-1, 3) if len(occ_cloud) else np.zeros((0, 3))
    s.update_occupancy_map_ptr(occ)
    s.set_terminal_goal(mk_state(pos=gterm_pos))
    s.change_drone_status(DroneStatus.TRAVELING)
    for t in trajs:
        s.add_traj(t, 0.0)
    planned_ok, attempted = s.replan(0.0, 0.0)
    # committed plan tail = the goal_setpoints (positions), if any
    setpts = s.retrieve_goal_setpoints()
    lines.append("CASE")
    lines.append("KIND REPLAN")
    lines.append(f"NAME {name}")
    lines.append(f"WORST_TRAJ_TIME {repr(float(s.worst_traj_time))}")
    lines.append(f"STATE_POS {fmt(state_pos)}")
    lines.append(f"GTERM_POS {fmt(gterm_pos)}")
    lines.append(f"NOCC {occ.shape[0]}")
    lines.append(f"OCC {fmt(occ)}" if occ.shape[0] else "OCC")
    lines.append(f"NTRAJ {len(trajs)}")
    for i, t in enumerate(trajs):
        emit_traj(f"TRAJ{i}", t)
    lines.append(f"PLANNED_OK {1 if planned_ok else 0}")
    lines.append(f"ATTEMPTED {1 if attempted else 0}")
    lines.append(f"NPLAN_AFTER {len(s.plan)}")
    lines.append(f"NSET {len(setpts)}")
    # sample setpoint positions (committed plan tail) at NSAMP indices
    if setpts:
        idxs = np.linspace(0, len(setpts) - 1, min(NSAMP_REPLAN, len(setpts))).astype(int)
        samp = np.array([setpts[i].pos for i in idxs])
        lines.append(f"NSAMP {len(idxs)}")
        lines.append(f"SET_POS {fmt(samp)}")
    else:
        lines.append("NSAMP 0")
        lines.append("SET_POS")
    lines.append("END")


# empty scene -> straight smooth flight
replan_case("replan_empty", [0, 0, 2], [6, 0, 2], [], [])
# soft wall
replan_case("replan_wall", [0, 0, 2], [6, 0, 2], wall_pts, [])
# hard human in the way (id<200 -> human/hard)
human_traj = make_traj(3, "3.0", "0.2", "2.0", bbox=(0.5, 0.5, 1.8))
replan_case("replan_human", [0, 0, 2], [6, 0, 2], [], [human_traj])
# wall-tagged dyn obstacle (200<=id<300 -> wall/soft)
wall_traj = make_traj(205, "3.0", "0.0", "2.0", bbox=(1.0, 1.0, 2.0))
replan_case("replan_dynwall", [0, 0, 2], [6, 0, 2], [], [wall_traj])
# moving human (space-time)
mover_traj = make_traj(4, "3.0", "0.8 - 0.3*t", "2.0", bbox=(0.5, 0.5, 1.8))
replan_case("replan_moving_human", [0, 0, 2], [6, 0, 2], [], [mover_traj])

out = os.path.join(os.path.dirname(__file__), "planner_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
ncase = len([l for l in lines if l == "CASE"])
print(f"wrote {out}  ({ncase} cases)")
