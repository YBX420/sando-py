"""Golden-data generator for the two MINCO analytic cost-term functions.

不许欺骗、必须还原:用真 Python _minco_obstacle_term / _minco_dynamic_term 跑
「输入 -> (cost, dcost_dc, dcost_dT_explicit)」金标准, C++ 移植后逐项对上(容差内)。

Covers DIVERSE + RANDOM + edge cases:
  - obstacle term: walls (AABB) + moving sphere humans + static + AABB-inside +
    far obstacle (no active samples) + multiple obstacles + various kappa.
  - dynamic term: order=1 (vel) and order=2 (accel), limits below/above the
    trajectory's actual magnitudes (active / inactive), various weights / kappa.

dump 到 cpp/golden/local_opt_terms_cases.txt (纯文本)。
跑: python cpp/golden/gen_local_opt_terms_golden.py
"""
import os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.local.minco import MinjerkTraj
from sando_py.local.obstacles import SphereObstacle, AABBObstacle
from sando_py.local.avoid_config import AvoidParams
from sando_py.local.local_opt import (
    OptParams, _minco_obstacle_term, _minco_dynamic_term,
)

rng = np.random.default_rng(20260603)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []


def build_traj(M, scale=4.0, vscale=1.5):
    wp = rng.uniform(-scale, scale, (M + 1, 3))
    T = rng.uniform(0.3, 1.6, M)
    v0 = rng.normal(0, 0.4 * vscale, 3); a0 = rng.normal(0, 0.3, 3)
    vf = rng.normal(0, 0.4 * vscale, 3); af = rng.normal(0, 0.3, 3)
    return MinjerkTraj(wp, T, v0=v0, a0=a0, vf=vf, af=af), wp, T, v0, a0, vf, af


def obs_block(prefix, obstacles):
    """Serialize a list of obstacles in a parseable per-type encoding.
    Each obstacle: 'OBSi TYPE <fields...> CLASS <name> DSAFE <d> W <w>'.
    Sphere fields: centre0(3) radius vel(3) accel(3). AABB fields: lo(3) hi(3).
    The d_safe/weight come from the avoid_cfg lookup so C++ reads them inline."""
    out = []
    for k, (o, dsafe, w) in enumerate(obstacles):
        if isinstance(o, SphereObstacle):
            out.append(
                f"{prefix}{k} SPHERE {fmt(o.centre0)} {repr(float(o.radius))} "
                f"{fmt(o.vel)} {fmt(o.accel)} CLASS {o.class_name} "
                f"DSAFE {repr(float(dsafe))} W {repr(float(w))}")
        else:
            out.append(
                f"{prefix}{k} AABB {fmt(o.lo)} {fmt(o.hi)} CLASS {o.class_name} "
                f"DSAFE {repr(float(dsafe))} W {repr(float(w))}")
    return out


def emit_obstacle_case(name, M, obstacles, kappa, traj_scale=4.0):
    """obstacles: list of (obs, d_safe, weight). avoid_cfg built from them."""
    tr, wp, T, v0, a0, vf, af = build_traj(M, scale=traj_scale)
    avoid_cfg = {}
    obs_list = []
    for o, dsafe, w in obstacles:
        avoid_cfg[o.class_name] = AvoidParams(o.class_name, "soft", float(dsafe), float(w))
        obs_list.append(o)
    opt = OptParams()
    opt.kappa = kappa
    cost, dcc, dT = _minco_obstacle_term(tr, obs_list, avoid_cfg, opt)

    lines.append("CASE")
    lines.append("KIND OBS")
    lines.append(f"NAME {name}")
    lines.append(f"M {M}")
    lines.append(f"KAPPA {kappa}")
    lines.append(f"T {fmt(T)}")
    lines.append(f"WP {fmt(wp)}")
    lines.append(f"V0 {fmt(v0)}")
    lines.append(f"A0 {fmt(a0)}")
    lines.append(f"VF {fmt(vf)}")
    lines.append(f"AF {fmt(af)}")
    lines.append(f"NOBS {len(obstacles)}")
    lines.extend(obs_block("OBS", obstacles))
    lines.append(f"COST {repr(float(cost))}")
    lines.append(f"DCC {fmt(dcc)}")     # (6M,3)
    lines.append(f"DT {fmt(dT)}")       # (M,)
    lines.append("END")


def emit_dynamic_case(name, M, order, limit, weight, kappa, traj_scale=4.0, vscale=1.5):
    tr, wp, T, v0, a0, vf, af = build_traj(M, scale=traj_scale, vscale=vscale)
    opt = OptParams()
    opt.kappa = kappa
    cost, dcc, dT = _minco_dynamic_term(tr, order, float(limit), float(weight), opt)

    lines.append("CASE")
    lines.append("KIND DYN")
    lines.append(f"NAME {name}")
    lines.append(f"M {M}")
    lines.append(f"KAPPA {kappa}")
    lines.append(f"ORDER {order}")
    lines.append(f"LIMIT {repr(float(limit))}")
    lines.append(f"WEIGHT {repr(float(weight))}")
    lines.append(f"T {fmt(T)}")
    lines.append(f"WP {fmt(wp)}")
    lines.append(f"V0 {fmt(v0)}")
    lines.append(f"A0 {fmt(a0)}")
    lines.append(f"VF {fmt(vf)}")
    lines.append(f"AF {fmt(af)}")
    lines.append(f"COST {repr(float(cost))}")
    lines.append(f"DCC {fmt(dcc)}")
    lines.append(f"DT {fmt(dT)}")
    lines.append("END")


# ===========================================================================
# OBSTACLE-TERM cases
# ===========================================================================

# 1) single AABB wall close to a small trajectory (active samples)
emit_obstacle_case(
    "wall_close", M=2,
    obstacles=[(AABBObstacle(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]),
                             class_name="wall"), 0.6, 10.0)],
    kappa=16, traj_scale=2.0)

# 2) single moving sphere human (CV), close
emit_obstacle_case(
    "human_cv", M=3,
    obstacles=[(SphereObstacle(np.array([0.5, 0.2, -0.3]), 0.5,
                               vel=np.array([0.4, -0.2, 0.1]), class_name="human"),
                0.8, 1.0e2)],
    kappa=20, traj_scale=2.5)

# 3) moving sphere human with ACCELERATION (CA) -> per-sample dddt path
emit_obstacle_case(
    "human_ca", M=3,
    obstacles=[(SphereObstacle(np.array([0.3, -0.4, 0.2]), 0.6,
                               vel=np.array([0.3, 0.2, -0.1]), class_name="human",
                               accel=np.array([0.2, -0.15, 0.05])),
                0.9, 50.0)],
    kappa=18, traj_scale=2.5)

# 4) static sphere (vel=accel=0) -> dddt all zero
emit_obstacle_case(
    "sphere_static", M=2,
    obstacles=[(SphereObstacle(np.array([0.0, 0.0, 0.0]), 0.7, class_name="human"),
                0.9, 30.0)],
    kappa=16, traj_scale=1.8)

# 5) multiple mixed obstacles (wall + two humans), various d_safe/weight
emit_obstacle_case(
    "mixed_multi", M=4,
    obstacles=[
        (AABBObstacle(np.array([-2.0, 0.5, -2.0]), np.array([0.0, 2.0, 2.0]),
                      class_name="wall"), 0.5, 8.0),
        (SphereObstacle(np.array([1.0, -0.5, 0.4]), 0.4,
                        vel=np.array([-0.3, 0.1, 0.0]), class_name="human"),
         0.8, 100.0),
        (SphereObstacle(np.array([-0.5, 1.0, -0.6]), 0.55,
                        vel=np.array([0.1, -0.25, 0.2]), class_name="human2",
                        accel=np.array([-0.1, 0.05, 0.1])),
         0.7, 40.0),
    ],
    kappa=24, traj_scale=2.0)

# 6) far obstacle -> NO active samples (cost should be 0, grads 0)
emit_obstacle_case(
    "far_inactive", M=2,
    obstacles=[(SphereObstacle(np.array([50.0, 50.0, 50.0]), 0.3, class_name="human"),
                0.8, 100.0)],
    kappa=16, traj_scale=2.0)

# 7) AABB where some samples land INSIDE the box (penetration subgradient path)
emit_obstacle_case(
    "wall_inside", M=2,
    obstacles=[(AABBObstacle(np.array([-3.0, -3.0, -3.0]), np.array([3.0, 3.0, 3.0]),
                             class_name="wall"), 0.5, 12.0)],
    kappa=16, traj_scale=2.0)

# 8) kappa=1 edge (single midpoint sample), moving human
emit_obstacle_case(
    "kappa1_human", M=2,
    obstacles=[(SphereObstacle(np.array([0.4, 0.3, 0.1]), 0.5,
                               vel=np.array([0.5, 0.0, -0.2]), class_name="human"),
                0.9, 80.0)],
    kappa=1, traj_scale=2.0)

# 9) M=1 single segment, moving human (cross-segment c2 path absent)
emit_obstacle_case(
    "single_seg_human", M=1,
    obstacles=[(SphereObstacle(np.array([0.6, -0.2, 0.3]), 0.6,
                               vel=np.array([0.3, 0.3, 0.0]), class_name="human"),
                0.95, 60.0)],
    kappa=16, traj_scale=2.0)

# 10) several RANDOM cases for breadth
for r in range(6):
    M = int(rng.integers(1, 6))
    nobs = int(rng.integers(1, 4))
    obstacles = []
    for o in range(nobs):
        if rng.random() < 0.5:
            obstacles.append((
                SphereObstacle(rng.uniform(-2, 2, 3), float(rng.uniform(0.3, 0.8)),
                               vel=rng.normal(0, 0.4, 3), class_name=f"human{o}",
                               accel=(rng.normal(0, 0.2, 3) if rng.random() < 0.5
                                      else None)),
                float(rng.uniform(0.5, 1.0)), float(rng.uniform(10, 200))))
        else:
            lo = rng.uniform(-3, 0, 3)
            hi = lo + rng.uniform(0.5, 3.0, 3)
            obstacles.append((
                AABBObstacle(lo, hi, class_name=f"wall{o}"),
                float(rng.uniform(0.3, 0.8)), float(rng.uniform(5, 20))))
    kappa = int(rng.integers(2, 30))
    emit_obstacle_case(f"rand_obs{r}", M, obstacles, kappa, traj_scale=2.5)


# ===========================================================================
# DYNAMIC-TERM cases
# ===========================================================================

# vel hinge, limit below typical speeds -> active
emit_dynamic_case("vel_active", M=3, order=1, limit=0.5, weight=100.0, kappa=16,
                  traj_scale=3.0, vscale=2.5)
# vel hinge, very high limit -> inactive (cost 0)
emit_dynamic_case("vel_inactive", M=3, order=1, limit=1000.0, weight=100.0, kappa=16,
                  traj_scale=3.0, vscale=2.5)
# accel hinge active
emit_dynamic_case("accel_active", M=4, order=2, limit=0.3, weight=80.0, kappa=20,
                  traj_scale=3.5, vscale=3.0)
# accel hinge inactive
emit_dynamic_case("accel_inactive", M=2, order=2, limit=5000.0, weight=80.0, kappa=16,
                  traj_scale=2.0, vscale=1.0)
# kappa=1 edge vel
emit_dynamic_case("vel_kappa1", M=2, order=1, limit=0.4, weight=120.0, kappa=1,
                  traj_scale=3.0, vscale=2.5)
# M=1 single segment accel
emit_dynamic_case("accel_single", M=1, order=2, limit=0.4, weight=60.0, kappa=16,
                  traj_scale=2.5, vscale=2.0)
# random dynamic cases
for r in range(6):
    M = int(rng.integers(1, 6))
    order = int(rng.integers(1, 3))
    limit = float(rng.uniform(0.2, 1.2))
    weight = float(rng.uniform(20, 200))
    kappa = int(rng.integers(2, 28))
    emit_dynamic_case(f"rand_dyn{r}", M, order, limit, weight, kappa,
                      traj_scale=3.0, vscale=2.5)


out = os.path.join(os.path.dirname(__file__), "local_opt_terms_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({sum(1 for l in lines if l == 'CASE')} cases)")
