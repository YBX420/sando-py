"""Golden-data generator for the MINCO optimiser DRIVER C++ port (plan_minco.hpp).

不许欺骗、必须还原: drives the REAL Python sando_py.local.local_opt over diverse +
random + edge scenarios. Two kinds of golden are written:

  (A) DETERMINISTIC helpers — golden-tested EXACTLY (no optimiser involved):
        ORTHO   : _orthonormal_pair(axis)              -> (u, v)
        PART    : _partition_obstacles                 -> per-obstacle soft/hard tag
        SEEDS   : _generate_detour_seeds               -> exact seed names + path geometry
                  (covers _make_detour_path / _obstacle_geom / _resample_polyline /
                   _min_clearance_per_obstacle / _orthonormal_pair end to end)
        FEAS    : check_feasibility(GIVEN MinjerkTraj) -> all feasibility fields

  (B) plan_minco END-TO-END — NOT bit-exact (LBFGSpp vs scipy L-BFGS-B differ on
        iterates). We dump the FULL scenario + Python's final SAMPLED trajectory
        positions, trajectory_valid, final_cost. The C++ test runs its own
        plan_minco and REPORTS the true sampled-position max error + whether
        trajectory_valid matches. We do NOT assert bit-equality on these.

Run: python cpp/golden/gen_plan_minco_golden.py
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sando_py.local.minco import MinjerkTraj
from sando_py.local.obstacles import SphereObstacle, AABBObstacle
from sando_py.local.avoid_config import default_config, AvoidParams
from sando_py.local import local_opt as LO
from sando_py.local.local_opt import OptParams, DetourConfig, plan_minco, check_feasibility

rng = np.random.default_rng(20260603)

lines = []


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


def emit_obstacle(prefix, obs):
    if hasattr(obs, "centre0"):
        lines.append(f"{prefix}_KIND sphere")
        lines.append(f"{prefix}_C0 {fmt(obs.centre0)}")
        lines.append(f"{prefix}_R {repr(float(obs.radius))}")
        lines.append(f"{prefix}_VEL {fmt(obs.vel)}")
        lines.append(f"{prefix}_ACC {fmt(obs.accel)}")
        lines.append(f"{prefix}_CLASS {obs.class_name}")
    else:
        lines.append(f"{prefix}_KIND aabb")
        lines.append(f"{prefix}_LO {fmt(obs.lo)}")
        lines.append(f"{prefix}_HI {fmt(obs.hi)}")
        lines.append(f"{prefix}_CLASS {obs.class_name}")


def emit_cfg(cfg, classes):
    lines.append(f"NCLASS {len(classes)}")
    for ci, cn in enumerate(classes):
        p = cfg.get(cn)
        if p is None:
            lines.append(f"CLASS{ci} {cn} 0.8 1.0 hard")
        else:
            lines.append(f"CLASS{ci} {cn} {repr(float(p.d_safe))} "
                         f"{repr(float(p.weight))} {p.mode}")


def emit_opt(opt):
    lines.append(f"W_SMOOTH {repr(float(opt.w_smooth))}")
    lines.append(f"W_VEL {repr(float(opt.w_vel))}")
    lines.append(f"W_ACCEL {repr(float(opt.w_accel))}")
    lines.append(f"W_TIME {repr(float(opt.w_time))}")
    lines.append(f"W_OBS {repr(float(opt.w_obs))}")
    lines.append(f"VMAX {repr(float(opt.vmax))}")
    lines.append(f"AMAX {repr(float(opt.amax))}")
    lines.append(f"DT_MIN {repr(float(opt.dt_min))}")
    lines.append(f"K {int(opt.K)}")
    lines.append(f"MAXITER {int(opt.maxiter)}")
    lines.append(f"KAPPA {int(opt.kappa)}")
    lines.append(f"M_CAP {int(opt.M_cap)}")
    lines.append(f"CLEAR_TOL {repr(float(opt.clearance_tol))}")
    lines.append(f"VEL_TOL {repr(float(opt.vel_tol))}")
    lines.append(f"ACCEL_TOL {repr(float(opt.accel_tol))}")
    lines.append(f"AVOID_OVERRIDE {opt.avoid_override if opt.avoid_override else '_'}")
    lines.append(f"ALM_RHO0 {repr(float(opt.alm_rho0))}")
    lines.append(f"ALM_RHO_MAX {repr(float(opt.alm_rho_max))}")
    lines.append(f"ALM_RHO_GROW {repr(float(opt.alm_rho_grow))}")
    lines.append(f"ALM_OUTER {int(opt.alm_outer_iters)}")
    lines.append(f"ALM_VIOL_TOL {repr(float(opt.alm_viol_tol))}")
    lines.append(f"ALM_INNER_MAXITER {int(opt.alm_inner_maxiter)}")
    lines.append(f"TAU_TRUST {repr(float(opt.tau_trust))}")
    lines.append(f"SPACETIME_HARD {1 if opt.spacetime_hard else 0}")
    lines.append(f"Q_CONFORMAL {repr(float(opt.q_conformal))}")
    lines.append(f"EPSILON_TRACK {repr(float(opt.epsilon_track))}")
    lines.append(f"SAFETY_MODE {opt.safety_mode}")
    lines.append(f"REACH_MODEL {opt.reach_model}")
    lines.append(f"A_MAX_HUMAN {repr(float(opt.a_max_human))}")
    lines.append(f"V_MAX_HUMAN {repr(float(opt.v_max_human))}")
    lines.append(f"EST_POS_ERR {repr(float(opt.est_pos_err))}")
    lines.append(f"EST_VEL_ERR {repr(float(opt.est_vel_err))}")


def emit_detour(dc):
    lines.append(f"D_ENABLED {1 if dc.enabled else 0}")
    lines.append(f"D_DIRECTIONS {int(dc.directions)}")
    lines.append(f"D_TRIGGER_MARGIN {repr(float(dc.trigger_margin))}")
    lines.append(f"D_DETOUR_MARGIN {repr(float(dc.detour_margin))}")
    lines.append(f"D_TOPK {int(dc.top_k_obstacles)}")
    lines.append(f"D_MAX_SEEDS {int(dc.max_seeds)}")
    lines.append(f"D_KEVAL {int(dc.K_eval)}")
    lines.append(f"D_PWINDOW {int(dc.perturb_window)}")


def mk_opt(**kw):
    o = OptParams()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def mk_detour(**kw):
    d = DetourConfig()
    for k, v in kw.items():
        setattr(d, k, v)
    return d


# ===========================================================================
# (A1) ORTHO — _orthonormal_pair
# ===========================================================================
ortho_axes = [
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0], [3.0, -2.0, 0.5], [0.0, 0.0, 0.0],   # degenerate
    [1e-12, 0.0, 0.0],                                     # near-degenerate
    [-1.0, 2.0, -3.0], [0.7, 0.7, 0.0], [0.0, 4.0, 0.001],
]
for _ in range(20):
    ortho_axes.append(list(rng.normal(0, 3, 3)))
for idx, ax in enumerate(ortho_axes):
    ax = np.asarray(ax, dtype=float)
    u, v = LO._orthonormal_pair(ax)
    lines.append("CASE")
    lines.append("KIND ORTHO")
    lines.append(f"NAME ortho{idx}")
    lines.append(f"AXIS {fmt(ax)}")
    lines.append(f"U {fmt(u)}")
    lines.append(f"V {fmt(v)}")
    lines.append("END")


# ===========================================================================
# (A2) PART — _partition_obstacles  (per-obstacle soft/hard tag, in order)
# ===========================================================================
def part_case(name, obstacles, override, cfg_extra=None):
    cfg = default_config()
    if cfg_extra:
        cfg.update(cfg_extra)
    soft, hard = LO._partition_obstacles(obstacles, cfg, override)
    # tag per obstacle in original order: 1=hard, 0=soft
    tags = []
    for o in obstacles:
        tags.append(1.0 if LO.resolve_mode(o, cfg, override) == "hard" else 0.0)
    lines.append("CASE")
    lines.append("KIND PART")
    lines.append(f"NAME {name}")
    classes = sorted(set(o.class_name for o in obstacles))
    emit_cfg(cfg, classes)
    lines.append(f"OVERRIDE {override if override else '_'}")
    lines.append(f"N {len(obstacles)}")
    for i, o in enumerate(obstacles):
        emit_obstacle(f"OBS{i}", o)
    lines.append(f"NSOFT {len(soft)}")
    lines.append(f"NHARD {len(hard)}")
    lines.append(f"TAGS {fmt(tags)}")
    lines.append("END")


base = np.array([0.0, 0.0, 0.0])
part_case("part_mixed", [
    SphereObstacle(base, 0.4, class_name="human"),
    AABBObstacle(base + 1, base + 2, class_name="wall"),
    SphereObstacle(base + 3, 0.3, vel=np.array([0.2, 0.0, 0.0]), class_name="human"),
], None)
part_case("part_all_soft", [
    SphereObstacle(base, 0.4, class_name="human"),
    AABBObstacle(base + 1, base + 2, class_name="wall"),
], "soft")
part_case("part_all_hard", [
    AABBObstacle(base + 1, base + 2, class_name="wall"),
    SphereObstacle(base, 0.4, class_name="human"),
], "hard")
part_case("part_unknown_class", [
    SphereObstacle(base, 0.4, class_name="mystery"),   # unknown -> fail-safe hard
    AABBObstacle(base + 1, base + 2, class_name="wall"),
], None)
part_case("part_custom_cfg", [
    SphereObstacle(base, 0.4, class_name="robot"),
    AABBObstacle(base + 1, base + 2, class_name="door"),
], None, cfg_extra={
    "robot": AvoidParams("robot", "hard", 0.6, 5e3),
    "door":  AvoidParams("door",  "soft", 0.3, 5.0),
})
for ridx in range(8):
    nobs = int(rng.integers(1, 5))
    obs = []
    for _ in range(nobs):
        if rng.random() < 0.5:
            obs.append(SphereObstacle(rng.uniform(-3, 3, 3), float(rng.uniform(0.3, 0.6)),
                                      vel=rng.normal(0, 0.3, 3), class_name="human"))
        else:
            lo = rng.uniform(-3, 2, 3)
            obs.append(AABBObstacle(lo, lo + rng.uniform(0.4, 1.0, 3), class_name="wall"))
    ov = [None, "soft", "hard"][int(rng.integers(0, 3))]
    part_case(f"part_fuzz{ridx}", obs, ov)


# ===========================================================================
# (A3) SEEDS — _generate_detour_seeds (exact names + full path geometry)
# ===========================================================================
def seeds_case(name, astar_path, obstacles, opt, dc, cfg_extra=None):
    cfg = default_config()
    if cfg_extra:
        cfg.update(cfg_extra)
    seeds = LO._generate_detour_seeds(np.asarray(astar_path, float),
                                      list(obstacles), cfg, opt, dc)
    lines.append("CASE")
    lines.append("KIND SEEDS")
    lines.append(f"NAME {name}")
    classes = sorted(set(o.class_name for o in obstacles)) if obstacles else []
    emit_cfg(cfg, classes)
    emit_opt(opt)
    emit_detour(dc)
    ap = np.asarray(astar_path, float)
    lines.append(f"NP {ap.shape[0]}")
    lines.append(f"PATH {fmt(ap)}")
    lines.append(f"NOBS {len(obstacles)}")
    for i, o in enumerate(obstacles):
        emit_obstacle(f"OBS{i}", o)
    lines.append(f"NSEEDS {len(seeds)}")
    for si, (sname, spath) in enumerate(seeds):
        lines.append(f"SEED{si}_NAME {sname}")
        sp = np.asarray(spath, float)
        lines.append(f"SEED{si}_NP {sp.shape[0]}")
        lines.append(f"SEED{si}_PATH {fmt(sp)}")
    lines.append("END")


# straight path that grazes a wall -> triggers a detour
sp = np.stack([np.linspace(0, 6, 9), np.zeros(9), np.zeros(9)], axis=1)
wall = AABBObstacle(np.array([2.7, -0.6, -0.6]), np.array([3.3, 0.6, 0.6]), class_name="wall")
seeds_case("seeds_wall_graze", sp, [wall], mk_opt(), mk_detour())

# straight path through a human -> triggers
human = SphereObstacle(np.array([3.0, 0.0, 0.0]), 0.5, class_name="human")
seeds_case("seeds_human", sp, [human], mk_opt(), mk_detour())

# no obstacles -> only 'original'
seeds_case("seeds_none", sp, [], mk_opt(), mk_detour())

# detour disabled
seeds_case("seeds_disabled", sp, [wall], mk_opt(), mk_detour(enabled=False))

# far obstacle -> no violation -> only 'original'
farwall = AABBObstacle(np.array([2.7, 5.0, 5.0]), np.array([3.3, 6.0, 6.0]), class_name="wall")
seeds_case("seeds_far", sp, [farwall], mk_opt(), mk_detour())

# multiple obstacles, top_k=2, more directions
sp2 = np.stack([np.linspace(0, 8, 12),
                0.3 * np.sin(np.linspace(0, 3, 12)),
                np.zeros(12)], axis=1)
obs2 = [AABBObstacle(np.array([2.5, -0.5, -0.5]), np.array([3.1, 0.5, 0.5]), class_name="wall"),
        SphereObstacle(np.array([5.5, 0.2, 0.0]), 0.5, class_name="human")]
seeds_case("seeds_multi_topk2", sp2, obs2,
           mk_opt(), mk_detour(top_k_obstacles=2, directions=4, max_seeds=8))

# path point ON the obstacle centre (axis degenerates -> start-goal fallback)
sp3 = np.stack([np.linspace(0, 4, 7), np.zeros(7), np.zeros(7)], axis=1)
on_centre = SphereObstacle(np.array([2.0, 0.0, 0.0]), 0.5, class_name="human")
seeds_case("seeds_on_centre", sp3, [on_centre], mk_opt(),
           mk_detour(perturb_window=2, detour_margin=0.4))

# diagonal 3D path with a sphere
sp4 = np.stack([np.linspace(0, 5, 10), np.linspace(0, 3, 10), np.linspace(0, 2, 10)], axis=1)
sph4 = SphereObstacle(np.array([2.5, 1.5, 1.0]), 0.6, class_name="human")
seeds_case("seeds_diag3d", sp4, [sph4], mk_opt(), mk_detour(directions=2))

# fuzz
for ridx in range(8):
    npn = int(rng.integers(5, 14))
    t = np.linspace(0, float(rng.uniform(4, 9)), npn)
    path = np.stack([t,
                     rng.uniform(-0.4, 0.4) * np.sin(t),
                     rng.uniform(-0.3, 0.3) * np.cos(t)], axis=1)
    obs = []
    for _ in range(int(rng.integers(1, 3))):
        mid = path[int(rng.integers(npn // 4, 3 * npn // 4))]
        if rng.random() < 0.5:
            obs.append(SphereObstacle(mid + rng.uniform(-0.2, 0.2, 3),
                                      float(rng.uniform(0.4, 0.7)),
                                      vel=rng.normal(0, 0.2, 3), class_name="human"))
        else:
            obs.append(AABBObstacle(mid - 0.4, mid + 0.4, class_name="wall"))
    dc = mk_detour(top_k_obstacles=int(rng.integers(1, 3)),
                   directions=int(rng.integers(1, 5)),
                   max_seeds=int(rng.integers(3, 8)),
                   detour_margin=float(rng.uniform(0.1, 0.5)),
                   trigger_margin=float(rng.uniform(0.05, 0.3)),
                   perturb_window=int(rng.integers(2, 4)),
                   K_eval=int(rng.integers(50, 150)))
    seeds_case(f"seeds_fuzz{ridx}", path, obs, mk_opt(vmax=float(rng.uniform(1, 3))), dc)


# ===========================================================================
# (A4) FEAS — check_feasibility on a GIVEN MinjerkTraj
# ===========================================================================
def feas_case(name, tr, obstacles, opt, cfg_extra=None):
    cfg = default_config()
    if cfg_extra:
        cfg.update(cfg_extra)
    feas = check_feasibility(tr, list(obstacles), cfg, opt, K_eval=opt.K)
    lines.append("CASE")
    lines.append("KIND FEAS")
    lines.append(f"NAME {name}")
    classes = sorted(set(o.class_name for o in obstacles)) if obstacles else []
    emit_cfg(cfg, classes)
    emit_opt(opt)
    # trajectory: M, waypoints (M+1,3), T (M), v0/a0/vf/af
    M = tr.M
    lines.append(f"M {M}")
    lines.append(f"WP {fmt(tr.waypoints)}")
    lines.append(f"T {fmt(tr.T)}")
    lines.append(f"V0 {fmt(tr.v0)}")
    lines.append(f"A0 {fmt(tr.a0)}")
    lines.append(f"VF {fmt(tr.vf)}")
    lines.append(f"AF {fmt(tr.af)}")
    lines.append(f"NOBS {len(obstacles)}")
    for i, o in enumerate(obstacles):
        emit_obstacle(f"OBS{i}", o)
    lines.append(f"F_MIN_CLR {repr(float(feas['min_clearance']))}")
    lines.append(f"F_MAX_CLR_VIOL {repr(float(feas['max_clearance_violation']))}")
    lines.append(f"F_MAX_V {repr(float(feas['max_v']))}")
    lines.append(f"F_MAX_A {repr(float(feas['max_a']))}")
    lines.append(f"F_VEL_VIOL {repr(float(feas['vel_violation']))}")
    lines.append(f"F_ACCEL_VIOL {repr(float(feas['accel_violation']))}")
    lines.append(f"F_VALID {1 if feas['trajectory_valid'] else 0}")
    lines.append(f"F_REASON {feas['failure_reason'] if feas['failure_reason'] else '_'}")
    lines.append("END")


def rand_traj(M, scale=3.0):
    start = rng.uniform(-scale, scale, 3)
    goal = rng.uniform(-scale, scale, 3)
    q = rng.uniform(-scale, scale, (M - 1, 3)) if M > 1 else np.zeros((0, 3))
    T = rng.uniform(0.4, 1.4, M)
    v0 = rng.normal(0, 0.3, 3); a0 = rng.normal(0, 0.2, 3)
    vf = rng.normal(0, 0.3, 3); af = rng.normal(0, 0.2, 3)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T, v0=v0, a0=a0, vf=vf, af=af)
    return tr


# clear trajectory, no obstacles, slow -> valid
tr = MinjerkTraj.from_endpoints(np.array([0, 0, 0.]), np.array([3, 0, 0.]),
                                np.array([[1.5, 0, 0.]]), np.array([1.0, 1.0]))
feas_case("feas_clear", tr, [], mk_opt(vmax=10, amax=20))

# with a wall it grazes
wall = AABBObstacle(np.array([1.3, -0.3, -0.3]), np.array([1.7, 0.3, 0.3]), class_name="wall")
feas_case("feas_wall_graze", tr, [wall], mk_opt(vmax=10, amax=20))

# tight velocity limit -> velocity_violation
feas_case("feas_vel_viol", tr, [], mk_opt(vmax=0.3, amax=20))

# tight accel limit -> acceleration_violation (vmax loose)
feas_case("feas_accel_viol", tr, [], mk_opt(vmax=10, amax=0.3))

# moving human, dynamic obstacle (time-dependent signed dist)
trm = MinjerkTraj.from_endpoints(np.array([0, 0, 0.]), np.array([4, 0, 0.]),
                                 np.array([[2.0, 0.5, 0.]]), np.array([1.2, 1.2]),
                                 v0=np.array([1.0, 0, 0.]))
mover = SphereObstacle(np.array([2.0, 0.0, 0.0]), 0.5,
                       vel=np.array([0.0, 0.3, 0.0]), class_name="human")
feas_case("feas_moving_human", trm, [mover], mk_opt(vmax=10, amax=20))

# fuzz feasibility
for ridx in range(12):
    M = int(rng.integers(1, 6))
    tr = rand_traj(M)
    nobs = int(rng.integers(0, 3))
    obs = []
    base = 0.5 * (tr.eval(tr.t_start) + tr.eval(tr.t_end))
    for _ in range(nobs):
        if rng.random() < 0.5:
            obs.append(SphereObstacle(base + rng.uniform(-1, 1, 3),
                                      float(rng.uniform(0.3, 0.7)),
                                      vel=rng.normal(0, 0.3, 3), class_name="human"))
        else:
            lo = base + rng.uniform(-0.6, 0.2, 3)
            obs.append(AABBObstacle(lo, lo + rng.uniform(0.4, 1.0, 3), class_name="wall"))
    opt = mk_opt(vmax=float(rng.uniform(0.5, 4)), amax=float(rng.uniform(0.5, 6)),
                 K=int(rng.integers(50, 200)),
                 clearance_tol=float(rng.uniform(0.01, 0.1)),
                 vel_tol=float(rng.uniform(0.05, 0.2)),
                 accel_tol=float(rng.uniform(0.05, 0.2)))
    feas_case(f"feas_fuzz{ridx}", tr, obs, opt)


# ===========================================================================
# (B) PLAN — plan_minco END-TO-END (NOT bit-exact; report measured agreement)
# ===========================================================================
NSAMP = 80   # trajectory sample count for the end-to-end comparison


def plan_case(name, astar_path, obstacles, opt, dc, v0=None, a0=None, cfg_extra=None):
    cfg = default_config()
    if cfg_extra:
        cfg.update(cfg_extra)
    ap = np.asarray(astar_path, float)
    tr, info = plan_minco(ap, list(obstacles), cfg, opt_params=opt,
                          detour_cfg=dc, v0=v0, a0=a0)
    ts = np.linspace(tr.t_start, tr.t_end, NSAMP)
    pts = tr.eval(ts)
    lines.append("CASE")
    lines.append("KIND PLAN")
    lines.append(f"NAME {name}")
    classes = sorted(set(o.class_name for o in obstacles)) if obstacles else []
    emit_cfg(cfg, classes)
    emit_opt(opt)
    emit_detour(dc)
    lines.append(f"NP {ap.shape[0]}")
    lines.append(f"PATH {fmt(ap)}")
    lines.append(f"NOBS {len(obstacles)}")
    for i, o in enumerate(obstacles):
        emit_obstacle(f"OBS{i}", o)
    lines.append(f"V0 {fmt(v0 if v0 is not None else np.zeros(3))}")
    lines.append(f"A0 {fmt(a0 if a0 is not None else np.zeros(3))}")
    # Python outputs: final sampled positions + flags
    lines.append(f"NSAMP {NSAMP}")
    lines.append(f"TSTART {repr(float(tr.t_start))}")
    lines.append(f"TEND {repr(float(tr.t_end))}")
    lines.append(f"POS {fmt(pts)}")
    lines.append(f"VALID {1 if info['trajectory_valid'] else 0}")
    lines.append(f"FINAL_COST {repr(float(info['final_cost']))}")
    lines.append(f"SEED_USED {info['seed_used']}")
    lines.append("END")


# Soft-only: straight path past a wall (convex-ish; should agree well).
spA = np.stack([np.linspace(0, 6, 9), np.zeros(9), np.zeros(9)], axis=1)
wallA = AABBObstacle(np.array([2.6, 0.3, -0.5]), np.array([3.4, 1.2, 0.5]), class_name="wall")
plan_case("plan_soft_wall", spA, [wallA], mk_opt(), mk_detour())

# Soft-only, no obstacle (pure smooth+time): the cleanest convex case.
plan_case("plan_no_obs", spA, [], mk_opt(), mk_detour(enabled=False))

# Hard static human off to the side (ALM active).
humanB = SphereObstacle(np.array([3.0, 0.7, 0.0]), 0.5, class_name="human")
plan_case("plan_hard_static", spA, [humanB], mk_opt(), mk_detour())

# Mixed soft wall + hard static human.
plan_case("plan_mixed", spA, [wallA, humanB], mk_opt(), mk_detour())

# Hard moving human (space-time).
moverC = SphereObstacle(np.array([3.0, 0.8, 0.0]), 0.45,
                        vel=np.array([0.0, -0.3, 0.0]), class_name="human")
plan_case("plan_hard_moving", spA, [moverC], mk_opt(tau_trust=3.0), mk_detour())

# Diagonal 3D, soft wall, with nonzero start velocity (receding-horizon).
spD = np.stack([np.linspace(0, 5, 9), np.linspace(0, 3, 9), np.linspace(0, 1.5, 9)], axis=1)
wallD = AABBObstacle(np.array([2.0, 1.0, 0.3]), np.array([2.8, 1.8, 1.0]), class_name="wall")
plan_case("plan_diag3d", spD, [wallD], mk_opt(), mk_detour(),
          v0=np.array([0.5, 0.3, 0.1]))

# All-hard override with a wall.
plan_case("plan_wall_allhard", spA, [wallA], mk_opt(avoid_override="hard"), mk_detour())

out = os.path.join(os.path.dirname(__file__), "plan_minco_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
ncase = len([l for l in lines if l == "CASE"])
print(f"wrote {out}  ({ncase} cases)")
