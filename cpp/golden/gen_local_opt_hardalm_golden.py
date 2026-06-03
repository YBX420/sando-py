"""Golden-data generator for the HARD per-class convex-hull ALM constraint
cost+gradient C++ port (local_opt_hardalm.hpp).

不许欺骗、必须还原: drive the REAL Python local_opt over diverse + random + edge
cases (Stage-3 static human, Stage-4 moving human, CA accel, walls-all-hard,
conformal Q_alpha, epsilon_track, deterministic accel/speed reach, multi-obstacle,
trust-window demotion, frozen-vs-recomputed normals/mask), dump one inner ALM
evaluation per case, and verify C++ reproduces it.

For each case we dump:
  - the trajectory build inputs (start, goal, q, T, v0/a0/vf/af)
  - the obstacle set (sphere centre0/r/vel/accel/class OR aabb lo/hi/class)
  - the OptParams ALM fields
  - the FIXED lambda / rho
  - whether normals + trust mask are FROZEN (pre-frozen from a perturbed iterate)
    or recomputed-from-tr (normals=None, trust_mask=None)
  - the Python _alm_term outputs: cost, dcost_dc (6M,3), dT_exp (M)
  - the per-point g / dist / R / dgdt / trust from _alm_constraints

Run: python cpp/golden/gen_local_opt_hardalm_golden.py
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sando_py.local.minco import MinjerkTraj
from sando_py.local.obstacles import SphereObstacle, AABBObstacle
from sando_py.local.avoid_config import default_config, AvoidParams
from sando_py.local import local_opt as LO

rng = np.random.default_rng(20260603)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []


def emit_obstacle(prefix, obs):
    """Dump one obstacle so C++ can rebuild it. SPHERE/AABB tag + params."""
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


def make_case(name, start, goal, q, T, v0, a0, vf, af,
              obstacles, opt, lam_scale, rho, freeze, q_extra_cfg=None):
    """Build one case: real Python _alm_term over the given inputs, dump I/O."""
    tr = MinjerkTraj.from_endpoints(start, goal, q, T, v0=v0, a0=a0, vf=vf, af=af)
    M = tr.M
    cfg = default_config()
    if q_extra_cfg:
        cfg.update(q_extra_cfg)

    # partition: which obstacles are HARD under this override (so the case mirrors
    # exactly what _minco_cost_grad feeds _alm_term)
    soft_obs, hard_obs = LO._partition_obstacles(obstacles, cfg, opt.avoid_override)
    if not hard_obs:
        return  # nothing to evaluate

    # number of constraints (to size lambda)
    cons0, _ = LO._alm_constraints(tr, hard_obs, cfg, opt=opt)
    Nc = cons0["g"].size
    # diverse fixed multipliers: random nonneg lambda (PHR keeps lambda >= 0)
    lam = np.abs(rng.normal(0.0, lam_scale, Nc))

    if freeze:
        # FROZEN normals + trust mask from a PERTURBED iterate (mimics the outer-loop
        # freeze where normals come from a different (q,T) than the one being evaluated)
        qf = q + rng.normal(0.0, 0.15, q.shape) if q.size else q
        Tf = np.maximum(T + rng.normal(0.0, 0.05, T.shape), 0.1)
        trf = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0, vf=vf, af=af)
        normals = LO._seg_normals(trf, hard_obs, opt=opt)
        tmask = LO._trust_mask(trf, opt)
        cost, gc, gT, cons = LO._alm_term(tr, hard_obs, cfg, lam, rho,
                                          normals=normals, opt=opt, trust_mask=tmask)
    else:
        normals = None
        tmask = None
        cost, gc, gT, cons = LO._alm_term(tr, hard_obs, cfg, lam, rho,
                                          normals=None, opt=opt, trust_mask=None)

    H = len(hard_obs)
    lines.append("CASE")
    lines.append(f"NAME {name}")
    lines.append(f"M {M}")
    lines.append(f"T {fmt(T)}")
    lines.append(f"START {fmt(start)}")
    lines.append(f"GOAL {fmt(goal)}")
    lines.append(f"Q {fmt(q) if q.size else ''}")
    lines.append(f"V0 {fmt(v0)}")
    lines.append(f"A0 {fmt(a0)}")
    lines.append(f"VF {fmt(vf)}")
    lines.append(f"AF {fmt(af)}")
    # OptParams ALM fields
    lines.append(f"AVOID_OVERRIDE {opt.avoid_override if opt.avoid_override else '_'}")
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
    # config: dump d_safe for each hard obstacle's class so C++ uses the same cfg
    classes = sorted(set(o.class_name for o in hard_obs))
    lines.append(f"NCLASS {len(classes)}")
    for ci, cn in enumerate(classes):
        params = cfg.get(cn)
        ds = params.d_safe if params is not None else 0.8
        lines.append(f"CLASS{ci} {cn} {repr(float(ds))}")
    # hard obstacles
    lines.append(f"H {H}")
    for hi, obs in enumerate(hard_obs):
        emit_obstacle(f"OBS{hi}", obs)
    # fixed multipliers + penalty
    lines.append(f"RHO {repr(float(rho))}")
    lines.append(f"NC {Nc}")
    lines.append(f"LAM {fmt(lam)}")
    # freeze info
    lines.append(f"FREEZE {1 if freeze else 0}")
    if freeze:
        lines.append(f"NORMALS {fmt(normals)}")     # (M,6,H,3)
        lines.append(f"TMASK {fmt(tmask.astype(float))}")  # (M,6)
    # outputs
    lines.append(f"COST {repr(float(cost))}")
    lines.append(f"DCC {fmt(gc)}")     # (6M,3)
    lines.append(f"DTEXP {fmt(gT)}")   # (M)
    # per-point constraint records
    lines.append(f"G {fmt(cons['g'])}")
    lines.append(f"NRM {fmt(cons['n'])}")       # (Nc,3)
    lines.append(f"DIST {fmt(cons['dist'])}")
    lines.append(f"RARR {fmt(cons['R'])}")
    lines.append(f"DGDT {fmt(cons['dgdt'])}")
    lines.append(f"TRUST {fmt(cons['trust'].astype(float))}")
    lines.append("END")


class Opt:
    """Lightweight OptParams stand-in carrying only the ALM fields _alm_term reads.
    Mirrors local_opt.OptParams defaults exactly for the fields used."""
    def __init__(self, **kw):
        self.avoid_override = kw.get("avoid_override", None)
        self.tau_trust = kw.get("tau_trust", 0.75)
        self.spacetime_hard = kw.get("spacetime_hard", True)
        self.q_conformal = kw.get("q_conformal", 0.0)
        self.epsilon_track = kw.get("epsilon_track", 0.0)
        self.safety_mode = kw.get("safety_mode", "conformal")
        self.reach_model = kw.get("reach_model", "accel")
        self.a_max_human = kw.get("a_max_human", 0.0)
        self.v_max_human = kw.get("v_max_human", 0.0)
        self.est_pos_err = kw.get("est_pos_err", 0.0)
        self.est_vel_err = kw.get("est_vel_err", 0.0)


def rand_traj(M):
    start = rng.uniform(-4, 4, 3)
    goal = rng.uniform(-4, 4, 3)
    q = rng.uniform(-4, 4, (M - 1, 3)) if M > 1 else np.zeros((0, 3))
    T = rng.uniform(0.4, 1.4, M)
    v0 = rng.normal(0, 0.4, 3); a0 = rng.normal(0, 0.25, 3)
    vf = rng.normal(0, 0.4, 3); af = rng.normal(0, 0.25, 3)
    return start, goal, q, T, v0, a0, vf, af


# ---------------------------------------------------------------------------
# Case 1: Stage-3 STATIC human (single sphere, vel=accel=0), spacetime on but
# static gate keeps it on the Stage-3 segment-END centroid path. recomputed normals.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid + np.array([0.3, 0.2, 0.0]), 0.5, class_name="human")]
make_case("stage3_static", *st, obs, Opt(), lam_scale=2.0, rho=10.0, freeze=False)

# Case 2: Stage-4 MOVING human (CV velocity), spacetime on, recomputed normals + mask.
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.6, -0.3, 0.1]), class_name="human")]
make_case("stage4_moving_cv", *st, obs, Opt(tau_trust=2.0),
          lam_scale=3.0, rho=20.0, freeze=False)

# Case 3: Stage-4 MOVING human with CA accel (nonzero accel path).
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.45, vel=np.array([0.3, 0.4, -0.2]),
                      class_name="human", accel=np.array([0.2, -0.1, 0.15]))]
make_case("stage4_moving_ca", *st, obs, Opt(tau_trust=3.0),
          lam_scale=2.5, rho=15.0, freeze=False)

# Case 4: TRUST-WINDOW demotion — small tau_trust so some control points are
# beyond the window (g -> G_NEG, trust=False, dgdt=0). moving human.
st = rand_traj(5)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.5, 0.2, 0.0]), class_name="human")]
make_case("stage4_trust_demote", *st, obs, Opt(tau_trust=0.6),
          lam_scale=3.0, rho=25.0, freeze=False)

# Case 5: FROZEN normals + trust mask from a perturbed iterate (the real outer-loop
# discipline). moving CA human.
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.5, vel=np.array([0.4, 0.3, 0.1]),
                      class_name="human", accel=np.array([-0.1, 0.2, 0.0]))]
make_case("frozen_moving", *st, obs, Opt(tau_trust=1.5),
          lam_scale=4.0, rho=30.0, freeze=True)

# Case 6: conformal Q_alpha margin (inflates sphere R, gradients unchanged but g shifts).
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.3, -0.2, 0.1]), class_name="human")]
make_case("conformal_q", *st, obs, Opt(q_conformal=0.93, tau_trust=2.0),
          lam_scale=2.0, rho=18.0, freeze=False)

# Case 7: epsilon_track tube (every hard obstacle, sphere + wall).
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.2, 0.2, 0.0]), class_name="human"),
       AABBObstacle(mid + np.array([0.5, 0.0, -0.3]),
                    mid + np.array([1.2, 0.8, 0.4]), class_name="wall")]
make_case("eps_track_mixed", *st, obs,
          Opt(epsilon_track=0.25, avoid_override="hard", tau_trust=2.0),
          lam_scale=2.0, rho=12.0, freeze=False)

# Case 8: deterministic ACCEL reach.
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.5, 0.2, -0.1]), class_name="human")]
make_case("det_accel", *st, obs,
          Opt(safety_mode="deterministic", reach_model="accel",
              a_max_human=2.0, est_pos_err=0.1, est_vel_err=0.3, tau_trust=0.75),
          lam_scale=3.0, rho=20.0, freeze=False)

# Case 9: deterministic SPEED reach. NOTE: _alm_term itself does NOT apply the
# _safety_surrogate (that swap happens in _alm_solve / _minco path before calling
# _alm_term). So here we pass the moving obstacle as-is; pred_margin uses the speed
# formula. This matches _alm_term's behaviour exactly.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.4, 0.3, 0.0]), class_name="human")]
make_case("det_speed", *st, obs,
          Opt(safety_mode="deterministic", reach_model="speed",
              v_max_human=2.5, est_pos_err=0.15, tau_trust=0.75),
          lam_scale=2.5, rho=16.0, freeze=False)

# Case 10: WALL all-hard (AABB constraint path, dg/dt=0, no conformal margin on wall).
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [AABBObstacle(mid + np.array([0.3, -0.2, -0.2]),
                    mid + np.array([1.0, 0.5, 0.3]), class_name="wall")]
make_case("wall_hard", *st, obs, Opt(avoid_override="hard"),
          lam_scale=1.5, rho=10.0, freeze=False)

# Case 11: MULTI-obstacle (2 humans + 1 wall), all-hard, moving + frozen.
st = rand_traj(5)
m0 = 0.5 * (st[0] + st[1]); m1 = 0.5 * (st[1] + st[1])
obs = [SphereObstacle(m0, 0.4, vel=np.array([0.5, 0.1, 0.0]), class_name="human"),
       SphereObstacle(m0 + np.array([0.6, 0.6, 0.2]), 0.35,
                      vel=np.array([-0.2, 0.3, 0.1]), class_name="human",
                      accel=np.array([0.1, 0.0, -0.1])),
       AABBObstacle(m0 + np.array([1.0, -0.5, -0.4]),
                    m0 + np.array([1.6, 0.4, 0.5]), class_name="wall")]
make_case("multi_frozen", *st, obs,
          Opt(avoid_override="hard", tau_trust=2.0, q_conformal=0.5,
              epsilon_track=0.1),
          lam_scale=3.0, rho=22.0, freeze=True)

# Case 12: spacetime_hard OFF with a moving human -> Stage-3 segment-END path
# (byte-identical to static treatment of the moving obstacle).
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.45, vel=np.array([0.7, 0.2, -0.3]), class_name="human")]
make_case("spacetime_off", *st, obs, Opt(spacetime_hard=False),
          lam_scale=2.0, rho=14.0, freeze=False)

# Case 13: M=1 single segment, moving human.
st = rand_traj(1)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.3, 0.3, 0.0]), class_name="human")]
make_case("single_seg", *st, obs, Opt(tau_trust=2.0),
          lam_scale=2.0, rho=12.0, freeze=False)

# Case 14: large lambda + small rho (PHR active/inactive boundary stress).
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid + np.array([2.0, 1.5, 0.5]), 0.4,
                      vel=np.array([0.2, 0.1, 0.0]), class_name="human")]
make_case("big_lam_small_rho", *st, obs, Opt(tau_trust=2.0),
          lam_scale=8.0, rho=2.0, freeze=False)

# Random fuzz cases: many M / obstacle / opt combinations.
for ridx in range(12):
    M = int(rng.integers(1, 7))
    st = rand_traj(M)
    nobs = int(rng.integers(1, 4))
    obs = []
    base = 0.5 * (st[0] + st[1])
    for _ in range(nobs):
        if rng.random() < 0.7:
            o = SphereObstacle(base + rng.uniform(-1.0, 1.0, 3),
                               float(rng.uniform(0.3, 0.6)),
                               vel=rng.normal(0, 0.4, 3),
                               class_name="human",
                               accel=(rng.normal(0, 0.2, 3)
                                      if rng.random() < 0.4 else None))
        else:
            lo = base + rng.uniform(-0.5, 0.2, 3)
            o = AABBObstacle(lo, lo + rng.uniform(0.4, 1.0, 3), class_name="wall")
        obs.append(o)
    opt = Opt(
        avoid_override=("hard" if rng.random() < 0.5 else None),
        tau_trust=float(rng.uniform(0.4, 2.5)),
        spacetime_hard=bool(rng.random() < 0.85),
        q_conformal=float(rng.uniform(0.0, 0.8)) if rng.random() < 0.5 else 0.0,
        epsilon_track=float(rng.uniform(0.0, 0.3)) if rng.random() < 0.5 else 0.0,
        safety_mode=("deterministic" if rng.random() < 0.3 else "conformal"),
        reach_model=("speed" if rng.random() < 0.5 else "accel"),
        a_max_human=float(rng.uniform(0.5, 2.5)),
        v_max_human=float(rng.uniform(0.5, 3.0)),
        est_pos_err=float(rng.uniform(0.0, 0.2)),
        est_vel_err=float(rng.uniform(0.0, 0.4)),
    )
    freeze = bool(rng.random() < 0.5)
    lam_scale = float(rng.uniform(0.5, 6.0))
    rho = float(rng.uniform(2.0, 35.0))
    make_case(f"fuzz{ridx}_M{M}", *st, obs, opt, lam_scale=lam_scale,
              rho=rho, freeze=freeze)

out = os.path.join(os.path.dirname(__file__), "local_opt_hardalm_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
ncase = len([l for l in lines if l == "CASE"])
print(f"wrote {out}  ({ncase} cases)")
