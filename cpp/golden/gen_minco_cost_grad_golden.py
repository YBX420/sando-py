"""Golden-data generator for the FULL MINCO objective+gradient (_minco_cost_grad)
C++ port (minco_cost_grad.hpp) verification.

不许欺骗、必须还原: drive the REAL Python local_opt._minco_cost_grad over diverse +
random + edge scenarios (soft-only, hard static/moving CV/CA, frozen vs recomputed
normals+trust mask, conformal Q_alpha, epsilon_track, deterministic accel/speed reach,
walls-all-hard, multi-obstacle mixed soft+hard, single segment, large M, every weight
combination), dump the inputs + (cost, grad_x), and verify C++ reproduces them.

The function L-BFGS-B calls. x = [q.ravel() ((M-1)*3), T (M)]; returns (f, grad).
We mirror the real solver discipline exactly: the SOFT set is passed as `obstacles`,
the HARD set + FIXED (lambda, rho) + FROZEN normals / trust mask are passed via the
hard_obs/alm_* kwargs (frozen from a possibly-perturbed iterate, like the outer loop).

For each case we dump:
  - x, start, goal, M, T_target, v0/a0/vf/af
  - the SOFT obstacle set + the HARD obstacle set (sphere or aabb params + class)
  - the per-class d_safe / weight / mode the cfg resolves (so C++ uses the same cfg)
  - all OptParams weights/limits + ALM/space-time fields + avoid_override
  - FIXED lambda / rho ; whether normals + trust mask are FROZEN (and their values)
  - the Python outputs: cost f, grad (nq + M,)

Run: python cpp/golden/gen_minco_cost_grad_golden.py
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)

from sando_py.local.minco import MinjerkTraj
from sando_py.local.obstacles import SphereObstacle, AABBObstacle
from sando_py.local.avoid_config import default_config, AvoidParams
from sando_py.local import local_opt as LO
from sando_py.local.local_opt import OptParams, _minco_cost_grad

rng = np.random.default_rng(20260603)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []


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


def make_case(name, start, goal, q, T, v0, a0, vf, af, obstacles, opt,
              lam_scale, rho, freeze, cfg_extra=None):
    """One case: real Python _minco_cost_grad over the given inputs, dump I/O.

    Mirrors the solver: partition obstacles into (soft, hard) under avoid_override,
    feed SOFT as `obstacles`, freeze the hard-ALM state (lambda, rho, normals, mask).
    """
    M = len(T)
    nq = 3 * (M - 1)
    x = np.concatenate([q.ravel(), T]) if M > 1 else T.copy()
    T_target = float(np.sum(T))

    cfg = default_config()
    if cfg_extra:
        cfg.update(cfg_extra)

    soft_obs, hard_obs = LO._partition_obstacles(obstacles, cfg, opt.avoid_override)

    # ALM frozen state (only if there is a hard set; else rho stays 0 -> ALM skipped)
    lam = None
    rho_used = 0.0
    normals = None
    tmask = None
    if hard_obs:
        tr0 = MinjerkTraj.from_endpoints(start, goal, q, T, v0=v0, a0=a0, vf=vf, af=af)
        cons0, _ = LO._alm_constraints(tr0, hard_obs, cfg, opt=opt)
        Nc = cons0["g"].size
        lam = np.abs(rng.normal(0.0, lam_scale, Nc))
        rho_used = rho
        if freeze:
            # FROZEN normals + mask from a PERTURBED iterate (outer-loop freeze discipline)
            qf = q + rng.normal(0.0, 0.15, q.shape) if q.size else q
            Tf = np.maximum(T + rng.normal(0.0, 0.05, T.shape), 0.1)
            trf = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0, vf=vf, af=af)
            normals = LO._seg_normals(trf, hard_obs, opt=opt)
            tmask = LO._trust_mask(trf, opt)
        else:
            normals = LO._seg_normals(tr0, hard_obs, opt=opt)
            tmask = LO._trust_mask(tr0, opt)

    # THE call (real Python). hard_obs=None / rho=0 disables the ALM term.
    f, grad = _minco_cost_grad(
        x, start, goal, M, list(soft_obs), cfg, opt, T_target,
        v0=v0, a0=a0, vf=vf, af=af,
        hard_obs=(hard_obs if hard_obs else None),
        alm_lambda=lam, alm_rho=rho_used,
        alm_normals=normals, alm_trust_mask=tmask)

    H = len(hard_obs)
    S = len(soft_obs)
    lines.append("CASE")
    lines.append(f"NAME {name}")
    lines.append(f"M {M}")
    lines.append(f"X {fmt(x)}")
    lines.append(f"START {fmt(start)}")
    lines.append(f"GOAL {fmt(goal)}")
    lines.append(f"TTARGET {repr(float(T_target))}")
    lines.append(f"V0 {fmt(v0)}")
    lines.append(f"A0 {fmt(a0)}")
    lines.append(f"VF {fmt(vf)}")
    lines.append(f"AF {fmt(af)}")
    # OptParams weights + limits
    lines.append(f"W_SMOOTH {repr(float(opt.w_smooth))}")
    lines.append(f"W_OBS {repr(float(opt.w_obs))}")
    lines.append(f"W_VEL {repr(float(opt.w_vel))}")
    lines.append(f"W_ACCEL {repr(float(opt.w_accel))}")
    lines.append(f"W_TIME {repr(float(opt.w_time))}")
    lines.append(f"VMAX {repr(float(opt.vmax))}")
    lines.append(f"AMAX {repr(float(opt.amax))}")
    lines.append(f"KAPPA {int(opt.kappa)}")
    # OptParams ALM / space-time fields
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
    # SFC corridor — only dumped when active so existing cases stay byte-identical
    if opt.w_corridor > 0.0 and opt.corridor_radius > 0.0 and opt.corridor_centers is not None:
        lines.append(f"W_CORRIDOR {repr(float(opt.w_corridor))}")
        lines.append(f"CORRIDOR_RADIUS {repr(float(opt.corridor_radius))}")
        lines.append(f"CORRIDOR_CENTERS {fmt(opt.corridor_centers)}")   # (M-1,3) flat
    # per-class cfg (dump every class referenced by ANY obstacle so the lookup matches)
    classes = sorted(set(o.class_name for o in obstacles))
    lines.append(f"NCLASS {len(classes)}")
    for ci, cn in enumerate(classes):
        p = cfg.get(cn)
        if p is None:
            lines.append(f"CLASS{ci} {cn} 0.8 1.0 hard")
        else:
            lines.append(f"CLASS{ci} {cn} {repr(float(p.d_safe))} "
                         f"{repr(float(p.weight))} {p.mode}")
    # soft obstacles
    lines.append(f"S {S}")
    for si, obs in enumerate(soft_obs):
        emit_obstacle(f"SOFT{si}", obs)
    # hard obstacles
    lines.append(f"H {H}")
    for hi, obs in enumerate(hard_obs):
        emit_obstacle(f"HARD{hi}", obs)
    # ALM frozen state
    lines.append(f"RHO {repr(float(rho_used))}")
    if hard_obs:
        lines.append(f"NC {lam.size}")
        lines.append(f"LAM {fmt(lam)}")
        lines.append(f"FREEZE {1 if freeze else 0}")
        lines.append(f"NORMALS {fmt(normals)}")     # (M,6,H,3) flat
        lines.append(f"TMASK {fmt(tmask.astype(float))}")  # (M,6)
    else:
        lines.append("NC 0")
        lines.append("FREEZE 0")
    # outputs
    lines.append(f"COST {repr(float(f))}")
    lines.append(f"GRAD {fmt(grad)}")    # (nq + M,)
    lines.append("END")


def mk_opt(**kw):
    o = OptParams()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def rand_traj(M):
    start = rng.uniform(-4, 4, 3)
    goal = rng.uniform(-4, 4, 3)
    q = rng.uniform(-4, 4, (M - 1, 3)) if M > 1 else np.zeros((0, 3))
    T = rng.uniform(0.4, 1.4, M)
    v0 = rng.normal(0, 0.4, 3); a0 = rng.normal(0, 0.25, 3)
    vf = rng.normal(0, 0.4, 3); af = rng.normal(0, 0.25, 3)
    return start, goal, q, T, v0, a0, vf, af


# ---------------------------------------------------------------------------
# Case 1: SOFT-ONLY (wall, ALM off) — pure EGO + dynamic + time + smooth.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [AABBObstacle(mid + np.array([0.2, -0.2, -0.2]),
                    mid + np.array([0.9, 0.5, 0.4]), class_name="wall")]
make_case("soft_only_wall", *st, obs, mk_opt(), lam_scale=0.0, rho=0.0, freeze=False)

# Case 2: NO obstacles at all (smooth + vel/accel hinge + time anchor only).
st = rand_traj(4)
make_case("no_obstacles", *st, [], mk_opt(vmax=1.0, amax=1.0), 0.0, 0.0, False)

# Case 3: hard STATIC human (single sphere vel=accel=0), recomputed normals.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid + np.array([0.3, 0.2, 0.0]), 0.5, class_name="human")]
make_case("hard_static", *st, obs, mk_opt(), lam_scale=2.0, rho=10.0, freeze=False)

# Case 4: hard MOVING human (CV velocity), spacetime on, recomputed.
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.6, -0.3, 0.1]), class_name="human")]
make_case("hard_moving_cv", *st, obs, mk_opt(tau_trust=2.0),
          lam_scale=3.0, rho=20.0, freeze=False)

# Case 5: hard MOVING human with CA accel.
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.45, vel=np.array([0.3, 0.4, -0.2]),
                      class_name="human", accel=np.array([0.2, -0.1, 0.15]))]
make_case("hard_moving_ca", *st, obs, mk_opt(tau_trust=3.0),
          lam_scale=2.5, rho=15.0, freeze=False)

# Case 6: FROZEN normals + trust mask from a perturbed iterate (real outer-loop).
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.5, vel=np.array([0.4, 0.3, 0.1]),
                      class_name="human", accel=np.array([-0.1, 0.2, 0.0]))]
make_case("frozen_moving", *st, obs, mk_opt(tau_trust=1.5),
          lam_scale=4.0, rho=30.0, freeze=True)

# Case 7: trust-window demotion (small tau_trust, some points beyond window).
st = rand_traj(5)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.5, 0.2, 0.0]), class_name="human")]
make_case("trust_demote", *st, obs, mk_opt(tau_trust=0.6),
          lam_scale=3.0, rho=25.0, freeze=False)

# Case 8: MIXED soft wall + hard human (per-class split, both terms active).
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.3, 0.2, 0.0]), class_name="human"),
       AABBObstacle(mid + np.array([0.6, 0.1, -0.3]),
                    mid + np.array([1.3, 0.9, 0.5]), class_name="wall")]
make_case("mixed_soft_hard", *st, obs, mk_opt(tau_trust=2.0),
          lam_scale=2.5, rho=18.0, freeze=False)

# Case 9: conformal Q_alpha margin.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.3, -0.2, 0.1]), class_name="human")]
make_case("conformal_q", *st, obs, mk_opt(q_conformal=0.93, tau_trust=2.0),
          lam_scale=2.0, rho=18.0, freeze=False)

# Case 10: epsilon_track tube, mixed sphere+wall, all-hard override.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.2, 0.2, 0.0]), class_name="human"),
       AABBObstacle(mid + np.array([0.5, 0.0, -0.3]),
                    mid + np.array([1.2, 0.8, 0.4]), class_name="wall")]
make_case("eps_track_allhard", *st, obs,
          mk_opt(epsilon_track=0.25, avoid_override="hard", tau_trust=2.0),
          lam_scale=2.0, rho=12.0, freeze=False)

# Case 11: deterministic ACCEL reach.
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.5, 0.2, -0.1]), class_name="human")]
make_case("det_accel", *st, obs,
          mk_opt(safety_mode="deterministic", reach_model="accel",
                 a_max_human=2.0, est_pos_err=0.1, est_vel_err=0.3, tau_trust=0.75),
          lam_scale=3.0, rho=20.0, freeze=False)

# Case 12: deterministic SPEED reach.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.4, 0.3, 0.0]), class_name="human")]
make_case("det_speed", *st, obs,
          mk_opt(safety_mode="deterministic", reach_model="speed",
                 v_max_human=2.5, est_pos_err=0.15, tau_trust=0.75),
          lam_scale=2.5, rho=16.0, freeze=False)

# Case 13: WALL forced all-hard (AABB in the ALM path).
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [AABBObstacle(mid + np.array([0.3, -0.2, -0.2]),
                    mid + np.array([1.0, 0.5, 0.3]), class_name="wall")]
make_case("wall_all_hard", *st, obs, mk_opt(avoid_override="hard"),
          lam_scale=1.5, rho=10.0, freeze=False)

# Case 14: multi-obstacle (2 humans + 1 wall) all-hard, moving + frozen.
st = rand_traj(5)
m0 = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(m0, 0.4, vel=np.array([0.5, 0.1, 0.0]), class_name="human"),
       SphereObstacle(m0 + np.array([0.6, 0.6, 0.2]), 0.35,
                      vel=np.array([-0.2, 0.3, 0.1]), class_name="human",
                      accel=np.array([0.1, 0.0, -0.1])),
       AABBObstacle(m0 + np.array([1.0, -0.5, -0.4]),
                    m0 + np.array([1.6, 0.4, 0.5]), class_name="wall")]
make_case("multi_frozen", *st, obs,
          mk_opt(avoid_override="hard", tau_trust=2.0, q_conformal=0.5,
                 epsilon_track=0.1),
          lam_scale=3.0, rho=22.0, freeze=True)

# Case 15: spacetime_hard OFF with a moving human (Stage-3 path).
st = rand_traj(4)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.45, vel=np.array([0.7, 0.2, -0.3]), class_name="human")]
make_case("spacetime_off", *st, obs, mk_opt(spacetime_hard=False),
          lam_scale=2.0, rho=14.0, freeze=False)

# Case 16: M=1 single segment, moving human.
st = rand_traj(1)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.3, 0.3, 0.0]), class_name="human")]
make_case("single_seg", *st, obs, mk_opt(tau_trust=2.0),
          lam_scale=2.0, rho=12.0, freeze=False)

# Case 17: M=1 soft-only.
st = rand_traj(1)
mid = 0.5 * (st[0] + st[1])
obs = [AABBObstacle(mid + np.array([0.1, -0.3, -0.2]),
                    mid + np.array([0.8, 0.4, 0.4]), class_name="wall")]
make_case("single_seg_soft", *st, obs, mk_opt(), 0.0, 0.0, False)

# Case 18: large M=6, mixed, frozen, conformal + eps.
st = rand_traj(6)
m0 = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(m0 + np.array([0.2, 0.1, 0.0]), 0.4,
                      vel=np.array([0.4, -0.2, 0.1]), class_name="human",
                      accel=np.array([0.1, 0.1, 0.0])),
       AABBObstacle(m0 + np.array([0.8, -0.4, -0.3]),
                    m0 + np.array([1.5, 0.5, 0.5]), class_name="wall")]
make_case("large_m6_mixed", *st, obs,
          mk_opt(tau_trust=2.5, q_conformal=0.4, epsilon_track=0.15,
                 w_smooth=2.0, w_time=5.0, w_vel=50.0, w_accel=80.0,
                 vmax=2.0, amax=2.0, kappa=24),
          lam_scale=2.5, rho=20.0, freeze=True)

# Case 19: high vel/accel so the dynamic hinges are strongly active (small limits).
st = rand_traj(4)
st = (st[0], st[1], st[2], st[3] * 0.4, st[4], st[5], st[6], st[7])  # short T -> fast
obs = []
make_case("active_hinges", *st, obs, mk_opt(vmax=0.5, amax=0.5, w_vel=200.0,
                                            w_accel=200.0), 0.0, 0.0, False)

# Case 20: zero weights edge (w_obs=w_vel=w_accel=w_time=0, only smooth) + hard human.
st = rand_traj(3)
mid = 0.5 * (st[0] + st[1])
obs = [SphereObstacle(mid, 0.4, vel=np.array([0.3, 0.2, 0.0]), class_name="human")]
make_case("zero_weights", *st, obs,
          mk_opt(w_obs=0.0, w_vel=0.0, w_accel=0.0, w_time=0.0, w_smooth=1.0,
                 tau_trust=2.0),
          lam_scale=2.0, rho=10.0, freeze=False)

# ---------------------------------------------------------------------------
# Random fuzz: many M / obstacle mixes / weights / ALM-on-off combinations.
for ridx in range(16):
    M = int(rng.integers(1, 7))
    st = rand_traj(M)
    nobs = int(rng.integers(0, 4))
    obs = []
    base = 0.5 * (st[0] + st[1])
    for _ in range(nobs):
        if rng.random() < 0.6:
            obs.append(SphereObstacle(
                base + rng.uniform(-1.0, 1.0, 3), float(rng.uniform(0.3, 0.6)),
                vel=rng.normal(0, 0.4, 3), class_name="human",
                accel=(rng.normal(0, 0.2, 3) if rng.random() < 0.4 else None)))
        else:
            lo = base + rng.uniform(-0.5, 0.2, 3)
            obs.append(AABBObstacle(lo, lo + rng.uniform(0.4, 1.0, 3),
                                    class_name="wall"))
    opt = mk_opt(
        w_smooth=float(rng.uniform(0.5, 3.0)),
        w_obs=float(rng.uniform(0.5, 2.0)),
        w_vel=float(rng.uniform(10.0, 200.0)),
        w_accel=float(rng.uniform(10.0, 200.0)),
        w_time=float(rng.uniform(0.0, 20.0)),
        vmax=float(rng.uniform(0.5, 3.0)),
        amax=float(rng.uniform(0.5, 3.0)),
        kappa=int(rng.integers(8, 28)),
        avoid_override=(["soft", "hard", None][int(rng.integers(0, 3))]),
        tau_trust=float(rng.uniform(0.4, 2.5)),
        spacetime_hard=bool(rng.random() < 0.85),
        q_conformal=(float(rng.uniform(0.0, 0.8)) if rng.random() < 0.5 else 0.0),
        epsilon_track=(float(rng.uniform(0.0, 0.3)) if rng.random() < 0.5 else 0.0),
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

# SFC soft flight-corridor active (placed AFTER the fuzz loop so it does not perturb the fuzz
# RNG stream -> all prior cases stay byte-identical). No obstacles -> the corridor term is isolated
# (smooth + vel/accel/time + corridor). Tube centres shifted off the waypoints so some q_i leave
# the radius -> non-trivial pull-back cost+grad. Exercises the (previously Python-dormant) term.
st = rand_traj(4)
_centers = st[2] - np.array([0.0, 1.1, 0.0])   # (M-1,3) shifted so several wpts leave the tube
make_case("sfc_corridor", *st, [],
          mk_opt(w_corridor=80.0, corridor_radius=0.4, corridor_centers=_centers),
          lam_scale=0.0, rho=0.0, freeze=False)


out = os.path.join(os.path.dirname(__file__), "minco_cost_grad_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
ncase = len([l for l in lines if l == "CASE"])
print(f"wrote {out}  ({ncase} cases)")
