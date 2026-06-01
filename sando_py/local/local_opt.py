"""Local trajectory optimisation: B-spline + L-BFGS-B + detour multi-start.

Pipeline (per replan):
  1. take A* post-processed polyline + obstacles + per-class avoid config
  2. detour seed generation:
        original LSQ seed
        + (for top_k violating obstacles) push waypoints by ±u/±v
          where u,v are two orthogonal directions perpendicular to the
          obstacle-to-path axis. Each detour pushes by (r + d_safe + margin).
  3. for each seed, LSQ-fit a clamped quintic B-spline → L-BFGS-B over
        (interior ctrl, dt) → check_feasibility on the result
  4. select best:
        if any candidate is feasible (clearance / vel / accel within tol):
            pick min final_cost
        else:
            pick "least bad" by (max_clearance_violation, vel_violation, cost)
  5. return (spline, info) where info splits:
        trajectory_valid   ← business feasibility
        optimizer_success  ← scipy convergence flag (independent!)

Cost (per seed):
    obstacle_cost (EGO cubic, per class)
  + w_smooth · mean(||snap||²)
  + w_vel    · mean(max(0, |v| - vmax))²
  + w_accel  · mean(max(0, |a| - amax))²
  + w_time   · ((T - T_target) / T_target)²
        (without this anchor, dt blows up because smooth/vel/accel cost ~ 1/dt²)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.optimize

from .avoid_config import AvoidParams, resolve_mode
from .bspline import UniformBSpline
from .cost import obstacle_cost
from .minco import MinjerkTraj, C2B


@dataclass
class OptParams:
    w_smooth: float = 1.0
    w_vel: float = 100.0
    w_accel: float = 100.0
    w_time: float = 10.0
    vmax: float = 3.0
    amax: float = 3.0
    dt_min: float = 0.05
    K: int = 200
    maxiter: int = 200
    degree: int = 5
    # feasibility tolerances (Q4 of Step 1 spec)
    clearance_tol: float = 0.05    # accept if min_clearance ≥ d_safe - tol
    vel_tol: float = 0.10          # accept if max_v ≤ vmax · (1 + tol)
    accel_tol: float = 0.10        # accept if max_a ≤ amax · (1 + tol)
    # --- MINCO path additions (plan_minco only; unused by the B-spline plan) ---
    w_obs: float = 1.0             # obstacle term weight in the MINCO cost
    kappa: int = 16                # per-segment time-integral quadrature samples
    M_cap: int = 12                # max interior-segment count for the MINCO seed
    # --- Stage 3: per-class HARD / SOFT + augmented-Lagrangian (ALM) ---
    # avoid_override: None -> per-class (mode from avoid_cfg[class].mode, fail-safe HARD);
    #                 'soft' -> force ALL obstacles soft (EGO field, ALM off) — the M2 baseline arm;
    #                 'hard' -> force ALL obstacles into the ALM convex-hull constraint (incl. walls).
    avoid_override: object = None
    alm_rho0: float = 10.0         # initial penalty parameter rho
    alm_rho_max: float = 1.0e8     # cap rho to avoid ill-conditioning the inner solve
    alm_rho_grow: float = 2.0      # multiply rho when the outer step stalls progress
    alm_outer_iters: int = 12      # outer-loop cap; STOP if still violating after this
    alm_viol_tol: float = 1.0e-3   # max constraint violation below which ALM has converged
    alm_inner_maxiter: int = 40    # L-BFGS-B iterations per inner solve (PHR warm-starts well)
    # --- Stage 4: per-control-point SPACE-TIME ALM (S4a) + trust window (S4b) ---
    # tau_trust: seconds of prediction trust AHEAD of the replan instant (offline
    #   t_now=0). A control point with wall-clock t_{i,k} <= tau_trust is HARD
    #   (ALM); beyond it the prediction is stale -> the point is DEMOTED from the
    #   hard set and covered by the SOFT EGO field instead (no double-count).
    # spacetime_hard: master switch. When False (or the human is static, vel==0)
    #   the EXACT Stage-3 segment-END single-time path runs (byte-identical).
    #   The _is_moving gate makes static humans Stage-3-identical even when True,
    #   so existing static tests are unaffected by the default-on switch.
    tau_trust: float = 0.75
    spacetime_hard: bool = True


@dataclass
class DetourConfig:
    enabled: bool = True
    directions: int = 4               # ±u, ±v
    trigger_margin: float = 0.10      # m — trigger detour if min_clearance < d_safe + margin
    detour_margin: float = 0.30       # m — push to (r + d_safe + margin) outside obstacle
    top_k_obstacles: int = 1          # only generate detours for the top-k violators
    max_seeds: int = 5                # hard cap on candidates
    K_eval: int = 100                 # samples along polyline for clearance/trigger check
    perturb_window: int = 3           # half-window of path points to nudge


# ---------------------------------------------------------------------------
# obstacle geometry helpers (sphere + AABB)
# ---------------------------------------------------------------------------

def _obstacle_geom(obs) -> Tuple[np.ndarray, float]:
    """Return (centre, radius_equiv). For AABB we fall back to its bounding sphere
    (first-version simplification — good enough to choose a detour direction)."""
    if hasattr(obs, "centre0"):  # SphereObstacle
        return obs.centre0.copy(), float(obs.radius)
    if hasattr(obs, "lo"):       # AABBObstacle
        centre = 0.5 * (obs.lo + obs.hi)
        radius = 0.5 * float(np.linalg.norm(obs.hi - obs.lo))
        return centre, radius
    raise ValueError(f"unknown obstacle type {type(obs).__name__}")


def _orthonormal_pair(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to `axis` and to each other."""
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    a = axis / n
    world = np.eye(3)
    cosines = np.abs(world @ a)
    pick = int(np.argmin(cosines))   # world axis least aligned with `axis`
    u = world[pick] - float(np.dot(world[pick], a)) * a
    u /= max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(a, u)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    return u, v


# ---------------------------------------------------------------------------
# detour seed generation
# ---------------------------------------------------------------------------

def _resample_polyline(path: np.ndarray, K: int) -> np.ndarray:
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total == 0:
        return np.tile(path[0], (K, 1))
    target = np.linspace(0, total, K)
    return np.stack([np.interp(target, cum, path[:, d]) for d in range(path.shape[1])], axis=1)


def _min_clearance_per_obstacle(path: np.ndarray, obstacles, K_eval: int):
    """Return list of (min_clearance, closest_path_index) per obstacle."""
    samples = _resample_polyline(path, K_eval)
    out = []
    for obs in obstacles:
        ds = np.array([obs.signed_dist(samples[i], 0.0) for i in range(K_eval)])
        i_min = int(np.argmin(ds))
        # map sample index back to path index (proportional)
        i_path = int(round(i_min * (len(path) - 1) / max(K_eval - 1, 1)))
        out.append((float(ds[i_min]), i_path))
    return out


def _make_detour_path(path: np.ndarray, obstacle, direction: np.ndarray,
                      d_safe: float, detour_cfg: DetourConfig) -> np.ndarray:
    """Bulge the polyline around obstacle in the given direction.

    Find the closest path index to the obstacle centre, compute the detour
    target at (centre + direction · (r + d_safe + margin)), then shift a
    window of nearby path points toward that target with a bell-shape weight.
    Endpoints stay pinned.
    """
    centre, r_eq = _obstacle_geom(obstacle)
    dists = np.linalg.norm(path - centre, axis=1)
    i_close = int(np.argmin(dists))
    target = centre + direction * (r_eq + d_safe + detour_cfg.detour_margin)
    shift = target - path[i_close]
    w = detour_cfg.perturb_window
    new_path = path.copy()
    M = len(path)
    for j in range(max(0, i_close - w), min(M, i_close + w + 1)):
        if j == 0 or j == M - 1:
            continue  # endpoints stay pinned
        weight = max(0.0, 1.0 - abs(j - i_close) / (w + 1))
        new_path[j] += shift * weight
    return new_path


def _generate_detour_seeds(astar_path: np.ndarray, obstacles, avoid_cfg, opt: OptParams,
                           detour_cfg: DetourConfig) -> List[Tuple[str, np.ndarray]]:
    """Build the list of candidate seed paths. Always includes 'original'."""
    seeds: List[Tuple[str, np.ndarray]] = [("original", astar_path)]
    if not detour_cfg.enabled or not obstacles:
        return seeds
    clearances = _min_clearance_per_obstacle(astar_path, obstacles, detour_cfg.K_eval)
    violators = []
    for i, obs in enumerate(obstacles):
        params = avoid_cfg.get(obs.class_name)
        d_safe = params.d_safe if params is not None else 0.8  # fail-safe default
        min_clr, i_path = clearances[i]
        violation = (d_safe + detour_cfg.trigger_margin) - min_clr
        if violation > 0:
            violators.append((violation, i, obs, d_safe, i_path))
    if not violators:
        return seeds
    violators.sort(key=lambda v: -v[0])  # most-violated first
    for violation, i, obs, d_safe, i_path in violators[: detour_cfg.top_k_obstacles]:
        centre, _r = _obstacle_geom(obs)
        path_pt = astar_path[min(i_path, len(astar_path) - 1)]
        axis = centre - path_pt
        if float(np.linalg.norm(axis)) < 1e-6:
            # path point coincides with obstacle centre — fall back to start-goal axis
            axis = astar_path[-1] - astar_path[0]
        u, v = _orthonormal_pair(axis)
        dirs = [u, -u, v, -v][: detour_cfg.directions]
        for j, d in enumerate(dirs):
            new_path = _make_detour_path(astar_path, obs, d, d_safe, detour_cfg)
            seeds.append((f"detour_obs{i}_dir{j}", new_path))
            if len(seeds) >= detour_cfg.max_seeds:
                return seeds
    return seeds


# ---------------------------------------------------------------------------
# feasibility check (Step 3: independent of optimizer flag)
# ---------------------------------------------------------------------------

def check_feasibility(spline, obstacles, avoid_cfg,
                      opt: OptParams, K_eval: int = 200) -> dict:
    """Compute trajectory-level feasibility, independent of optimiser convergence.

    `spline` may be a UniformBSpline OR a MinjerkTraj — only .t_start/.t_end/
    .eval/.eval_deriv are used, which both expose identically.

    Returns dict with min_clearance / max_clearance_violation / max_v / max_a /
    vel_violation / accel_violation / trajectory_valid / failure_reason.
    """
    ts = np.linspace(spline.t_start, spline.t_end, K_eval)
    pts = spline.eval(ts)
    min_clr = float("inf")
    max_violation = 0.0
    for obs in obstacles:
        params = avoid_cfg.get(obs.class_name)
        d_safe = params.d_safe if params is not None else 0.8  # fail-safe default
        for i in range(K_eval):
            d = obs.signed_dist(pts[i], float(ts[i]))
            if d < min_clr:
                min_clr = d
            violation = d_safe - d
            if violation > max_violation:
                max_violation = violation
    if min_clr == float("inf"):
        min_clr = 0.0  # no obstacles to measure against
    v_mag = np.linalg.norm(spline.eval_deriv(ts, 1), axis=1)
    a_mag = np.linalg.norm(spline.eval_deriv(ts, 2), axis=1)
    max_v = float(v_mag.max())
    max_a = float(a_mag.max())
    vel_violation = max(0.0, max_v - opt.vmax)
    accel_violation = max(0.0, max_a - opt.amax)

    failure_reason = None
    if obstacles and max_violation > opt.clearance_tol:
        failure_reason = "clearance_violation"
    elif max_v > opt.vmax * (1.0 + opt.vel_tol):
        failure_reason = "velocity_violation"
    elif max_a > opt.amax * (1.0 + opt.accel_tol):
        failure_reason = "acceleration_violation"

    return {
        "min_clearance": float(min_clr),
        "max_clearance_violation": float(max_violation),
        "max_v": max_v,
        "max_a": max_a,
        "vel_violation": float(vel_violation),
        "accel_violation": float(accel_violation),
        "trajectory_valid": failure_reason is None,
        "failure_reason": failure_reason,
    }


# ---------------------------------------------------------------------------
# inner optimisation primitives (shared)
# ---------------------------------------------------------------------------

def _build_spline(x: np.ndarray, num_ctrl: int, p: int,
                  start_pt: np.ndarray, end_pt: np.ndarray) -> UniformBSpline:
    dt = float(x[-1])
    interior = x[:-1].reshape(-1, 3)
    ctrl = np.empty((num_ctrl, 3), dtype=np.float64)
    ctrl[: p + 1] = start_pt
    ctrl[p + 1 : -(p + 1)] = interior
    ctrl[-(p + 1) :] = end_pt
    return UniformBSpline(ctrl, degree=p, dt=dt)


def _total_cost(x: np.ndarray, num_ctrl: int, p: int,
                start_pt: np.ndarray, end_pt: np.ndarray,
                obstacles, avoid_cfg, opt: OptParams, T_target: float) -> float:
    bs = _build_spline(x, num_ctrl, p, start_pt, end_pt)
    K = opt.K
    ts = np.linspace(bs.t_start, bs.t_end, K)
    c_obs = obstacle_cost(bs, obstacles, avoid_cfg, K=K)
    snap = bs.eval_deriv(ts, order=4)
    c_smooth = opt.w_smooth * float(np.mean(np.sum(snap * snap, axis=1)))
    vel = bs.eval_deriv(ts, order=1)
    v_mag = np.linalg.norm(vel, axis=1)
    v_over = np.maximum(v_mag - opt.vmax, 0.0)
    c_vel = opt.w_vel * float(np.mean(v_over * v_over))
    accel = bs.eval_deriv(ts, order=2)
    a_mag = np.linalg.norm(accel, axis=1)
    a_over = np.maximum(a_mag - opt.amax, 0.0)
    c_accel = opt.w_accel * float(np.mean(a_over * a_over))
    T_actual = (num_ctrl - p) * float(x[-1])
    rel = (T_actual - T_target) / max(T_target, 1e-6)
    c_time = opt.w_time * rel * rel
    return c_obs + c_smooth + c_vel + c_accel + c_time


def _optimise_one(seed_path: np.ndarray, obstacles, avoid_cfg, opt: OptParams,
                  num_ctrl: int) -> dict:
    """Run one LBFGS-B from a given seed polyline. Return candidate record."""
    p = opt.degree
    seg_lens = np.linalg.norm(np.diff(seed_path, axis=0), axis=1)
    path_len = float(seg_lens.sum())
    T0 = path_len / max(opt.vmax, 1e-6)
    dt0 = max(T0 / max(num_ctrl - p, 1), opt.dt_min * 2.0)
    bs0 = UniformBSpline.fit_path(seed_path, num_ctrl=num_ctrl, degree=p,
                                  dt=dt0, clamp_endpoints=True)
    start_pt = seed_path[0].copy()
    end_pt = seed_path[-1].copy()
    interior_init = bs0.ctrl[p + 1 : -(p + 1)].copy().ravel()
    x0 = np.concatenate([interior_init, [dt0]])
    bounds = [(None, None)] * (x0.size - 1) + [(opt.dt_min, None)]
    T_target = path_len / max(opt.vmax, 1e-6)
    args = (num_ctrl, p, start_pt, end_pt, list(obstacles), avoid_cfg, opt, T_target)
    init_cost = _total_cost(x0, *args)
    result = scipy.optimize.minimize(
        _total_cost, x0, args=args, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": opt.maxiter, "gtol": 1e-6},
    )
    bs = _build_spline(result.x, num_ctrl, p, start_pt, end_pt)
    feas = check_feasibility(bs, obstacles, avoid_cfg, opt)
    return {
        "spline": bs,
        "init_cost": float(init_cost),
        "final_cost": float(result.fun),
        "iter": int(result.nit),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "dt0": float(dt0),
        "dt_final": float(result.x[-1]),
        "T_target": float(T_target),
        "T_final": float((num_ctrl - p) * float(result.x[-1])),
        "feasibility": feas,
    }


def _select_best(candidates: List[dict]) -> dict:
    """Feasible-first; failing that, the 'least bad' candidate."""
    feasible = [c for c in candidates if c["feasibility"]["trajectory_valid"]]
    if feasible:
        return min(feasible, key=lambda c: c["final_cost"])
    return min(candidates, key=lambda c: (
        c["feasibility"]["max_clearance_violation"],
        c["feasibility"]["vel_violation"],
        c["final_cost"],
    ))


def _select_best_minco(candidates: List[dict]) -> dict:
    """MINCO selection. A candidate that breaches a HARD human can NEVER win as
    valid; feasible-first picks the cheapest fully-valid one, otherwise the
    least-bad is ranked HARD-breach FIRST (the violated constraint is a person),
    then soft-wall clearance, then vel, then cost — the explicit STOP ordering."""
    feasible = [c for c in candidates if c["feasibility"]["trajectory_valid"]]
    if feasible:
        return min(feasible, key=lambda c: c["final_cost"])
    return min(candidates, key=lambda c: (
        max(c.get("hard_max_breach", float("-inf")), 0.0),
        c["feasibility"]["max_clearance_violation"],
        c["feasibility"]["vel_violation"],
        c["final_cost"],
    ))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def plan(astar_path: np.ndarray,
         obstacles: Iterable,
         avoid_cfg: Dict[str, AvoidParams],
         *,
         num_ctrl: int = None,
         opt_params: OptParams = None,
         detour_cfg: DetourConfig = None) -> Tuple[UniformBSpline, dict]:
    """Plan a quintic B-spline with detour multi-start + feasibility selection.

    Returns (spline, info). `info` has these keys:
        trajectory_valid    (bool) — business feasibility (clearance / vel / accel)
        optimizer_success   (bool) — scipy L-BFGS-B convergence (independent)
        failure_reason      (str|None) — 'clearance_violation' / 'velocity_violation' /
                                          'acceleration_violation'
        feasibility         (dict) — full feasibility record (see check_feasibility)
        seed_used           (str)  — name of the seed that won selection
        n_seeds_tried       (int)
        init_cost, final_cost, iter, dt0, dt_final, T_target, T_final, num_ctrl
        message             — scipy termination message
        converged           — alias of optimizer_success (legacy, will be removed)
    """
    opt = opt_params or OptParams()
    detour_cfg = detour_cfg or DetourConfig()
    p = opt.degree
    astar_path = np.asarray(astar_path, dtype=np.float64)
    if astar_path.ndim != 2 or astar_path.shape[0] < 2:
        raise ValueError(f"astar_path must be (M≥2, d), got {astar_path.shape}")
    if num_ctrl is None:
        num_ctrl = max(2 * (p + 1), int(round(len(astar_path) / 2)))
    if num_ctrl < 2 * (p + 1) + 1:
        num_ctrl = 2 * (p + 1) + 1

    obstacles_list = list(obstacles)
    seeds = _generate_detour_seeds(astar_path, obstacles_list, avoid_cfg, opt, detour_cfg)

    candidates: List[dict] = []
    for name, seed_path in seeds:
        try:
            rec = _optimise_one(seed_path, obstacles_list, avoid_cfg, opt, num_ctrl)
        except Exception as e:
            # one seed failed numerically — note it, continue with the rest
            continue
        rec["seed_name"] = name
        candidates.append(rec)
    if not candidates:
        raise RuntimeError("all seeds failed to optimise")

    best = _select_best(candidates)
    feas = best["feasibility"]
    info = {
        "trajectory_valid": feas["trajectory_valid"],
        "optimizer_success": best["optimizer_success"],
        "failure_reason": feas["failure_reason"],
        "feasibility": feas,
        "seed_used": best["seed_name"],
        "n_seeds_tried": len(candidates),
        "init_cost": best["init_cost"],
        "final_cost": best["final_cost"],
        "iter": best["iter"],
        "dt0": best["dt0"],
        "dt_final": best["dt_final"],
        "T_target": best["T_target"],
        "T_final": best["T_final"],
        "num_ctrl": int(num_ctrl),
        "message": best["message"],
        "converged": best["optimizer_success"],  # legacy alias
    }
    return best["spline"], info


# ===========================================================================
# MINCO analytic-gradient solve path (M2). Additive: the B-spline plan() above
# stays untouched. Decision vars x = [q.ravel() ((M-1)*3,), T (M,)].
# Cost = w_smooth·energy + w_obs·obstacle + w_vel·vel-hinge + w_accel·acc-hinge
#        + w_time·time-anchor, all with ANALYTIC gradient w.r.t. (q, T).
#
# Obstacle/vel/accel are time-integrals via per-segment midpoint quadrature
# (GCOPTER style): segment i sampled at tau = s_j·T_i, s_j=(j+0.5)/kappa, with
# quadrature weight T_i/kappa. The gradient w.r.t. T has TWO paths:
#   IMPLICIT (c = M(T)^{-1} b)            -> grad_from_dcost_dc(dcost_dc, ...)
#   EXPLICIT (moving sample times + qweight) -> dcost_dT_explicit, see below.
# ===========================================================================


def _signed_dist_and_grad(obs, p: np.ndarray, t: float):
    """Return (d, dd/dp (3,), dd/dt scalar) analytically for sphere / AABB.

    Sphere(moving): d=||p-c(t)||-r, c(t)=centre0+vel·t.
        dd/dp = (p-c)/||p-c||,  dd/dt = -(p-c)/||p-c|| · vel = -(dd/dp)·vel.
    AABB outside: d=||outside_vec||, dd/dp = outside_vec/||outside_vec|| (static, dd/dt=0).
    AABB inside : d = L∞ penetration (negative); subgradient ±e on the active axis.
    Caller must keep samples off the kinks (||p-c||=0, ||outside||=0, AABB face/interior)."""
    if hasattr(obs, "centre0"):  # SphereObstacle
        c = obs.centre0 + obs.vel * float(t)
        diff = np.asarray(p, dtype=np.float64) - c
        nrm = float(np.linalg.norm(diff))
        n = diff / nrm
        d = nrm - obs.radius
        dd_dt = -float(n.dot(obs.vel))
        return d, n, dd_dt
    # AABBObstacle (static -> dd/dt = 0)
    p = np.asarray(p, dtype=np.float64)
    over = np.maximum(obs.lo - p, 0.0) + np.maximum(p - obs.hi, 0.0)
    if np.any(over > 0.0):
        nrm = float(np.linalg.norm(over))
        # dd/dp_axis = sign on the violated face: +1 if p>hi, -1 if p<lo
        sign = np.where(p > obs.hi, 1.0, 0.0) - np.where(p < obs.lo, 1.0, 0.0)
        dd_dp = (over * sign) / nrm  # over is |violation|; restore signed direction
        return nrm, dd_dp, 0.0
    # inside: d = max over axes of max(lo-p, p-hi)  (<= 0); L∞ subgradient
    cand = np.maximum(obs.lo - p, p - obs.hi)  # per-axis penetration (signed, <=0)
    ax = int(np.argmax(cand))
    grad = np.zeros(3)
    # active term is max(lo-p, p-hi) on axis ax; whichever wins sets the sign
    grad[ax] = -1.0 if (obs.lo[ax] - p[ax]) >= (p[ax] - obs.hi[ax]) else 1.0
    return float(cand[ax]), grad, 0.0


def _seg_vander(taus: np.ndarray):
    """Power-basis rows for orders 0..3 at local times taus (kappa,).

    Returns (B0,B1,B2,B3) each (kappa,6): B_o[j] = _basis(taus[j], o).
    Vectorised over the kappa samples (no per-point scalar _basis): column p of
    order o is (p!/(p-o)!)*tau^(p-o), using the same `**` power for bit-parity."""
    taus = np.asarray(taus, dtype=np.float64).ravel()
    k = taus.size
    out = []
    for o in range(4):
        B = np.zeros((k, 6), dtype=np.float64)
        for p in range(o, 6):
            coeff = 1.0
            for m in range(o):
                coeff *= (p - m)
            B[:, p] = coeff * (taus ** (p - o))
        out.append(B)
    return out[0], out[1], out[2], out[3]


_BASIS_CACHE = {}


def _fixed_basis(kappa: int):
    """Cached fixed-fraction basis: s=(j+0.5)/kappa and Bs_o=_seg_vander(s).

    Returns (s, (Bs0,Bs1,Bs2,Bs3), idx) where idx[o][p]=max(p-o,0) selects the
    Ti-power that scales column p of order o (see _scale_basis)."""
    if kappa not in _BASIS_CACHE:
        s = (np.arange(kappa) + 0.5) / kappa
        Bs = _seg_vander(s)
        idx = [np.array([max(p - o, 0) for p in range(6)]) for o in range(4)]
        _BASIS_CACHE[kappa] = (s, Bs, idx)
    return _BASIS_CACHE[kappa]


def _scale_basis(Bs, idx, Ti: float):
    """Scale fixed-fraction basis Bs (at tau=s) to segment time Ti.

    B_o(s·Ti)[:,p] = Bs_o[:,p]·Ti^(p-o).  Columns p<o are zero in Bs, so their
    Ti^0=1 scaling is harmless. Replaces a per-segment _seg_vander rebuild."""
    tp = np.array([1.0, Ti, Ti * Ti, Ti**3, Ti**4, Ti**5])
    return tuple(Bs[o] * tp[idx[o]][None, :] for o in range(4))


def _minco_obstacle_term(tr: MinjerkTraj, obstacles, avoid_cfg, opt: OptParams):
    """(cost, dcost_dc (6M,3), dcost_dT_explicit (M,)) for the obstacle integral.

    cost = Σ_i (T_i/κ) Σ_j Σ_obs W·φ(d),  φ(d)=(d_safe-d)^3 for d<d_safe.
    See header / explicit_dT_term derivation: dcost_dT_explicit collects
      (a) quadrature-weight (1/κ)Σ_j Φ
      (b) moving-local-sample (T_i/κ)Σ_j φ'·(n·vel)·s_j      [tau=s_j·T_i]
      (c1) same-segment abs-time (moving obs) s_j·A_{i,j}
      (c2) cross-segment abs-time (moving obs) A_{i,j} added to every k<i
    where A_{i,j} = (T_i/κ)·W·φ'(d)·dd/dt."""
    M = tr.M
    n = tr.NC * M
    kap = max(int(opt.kappa), 1)
    s, Bs, bidx = _fixed_basis(kap)            # midpoint fractions (κ,) + fixed basis
    cost = 0.0
    dcost_dc = np.zeros((n, 3))
    dT_exp = np.zeros(M)
    if not obstacles:
        return cost, dcost_dc, dT_exp
    cum = tr._cum
    for i in range(M):
        Ti = float(tr.T[i])
        taus = s * Ti
        B0, B1, _, _ = _scale_basis(Bs, bidx, Ti)
        ci = tr.c[6 * i:6 * i + 6]             # (6,3)
        P = B0 @ ci                            # (κ,3) sample positions
        V = B1 @ ci                            # (κ,3) sample velocities
        t_abs = cum[i] + taus                  # (κ,) absolute times
        wq = Ti / kap
        for obs in obstacles:
            params = avoid_cfg[obs.class_name]
            d_safe = params.d_safe
            W = params.weight
            # batched (d, dd/dp, dd/dt) over the κ samples
            dvec, ndp, dddt = _signed_dist_and_grad_batch(obs, P, t_abs)  # (κ,),(κ,3),(κ,)
            diff = d_safe - dvec               # (κ,)
            act = diff > 0.0
            if not np.any(act):
                continue
            diff_a = diff[act]
            phi = diff_a ** 3
            dphi = -3.0 * diff_a * diff_a      # φ'(d), (na,)
            B0a = B0[act]; ndpa = ndp[act]; Va = V[act]; sa = s[act]
            cost += wq * W * float(np.sum(phi))
            # implicit-c: Σ_j (wq W dphi_j) outer(B0_j, ndp_j)
            coef = (wq * W) * dphi             # (na,)
            dcost_dc[6 * i:6 * i + 6] += B0a.T @ (coef[:, None] * ndpa)
            Phi = W * phi
            # (a) quadrature-weight
            dT_exp[i] += (1.0 / kap) * float(np.sum(Phi))
            # (b) moving-local-sample: wq Σ_j W dphi_j (ndp_j·V_j) s_j
            ndp_dot_V = np.einsum("kd,kd->k", ndpa, Va)
            dT_exp[i] += wq * float(np.sum(W * dphi * ndp_dot_V * sa))
            # (c) absolute-time term for moving obstacles
            dddt_a = dddt[act]
            if np.any(dddt_a != 0.0):
                A = (wq * W) * dphi * dddt_a   # (na,) = wq·∂Φ/∂t_abs per sample
                dT_exp[i] += float(np.sum(sa * A))   # (c1) same segment
                if i > 0:
                    dT_exp[:i] += float(np.sum(A))    # (c2) all upstream segments
    return cost, dcost_dc, dT_exp


def _signed_dist_and_grad_batch(obs, P: np.ndarray, t_abs: np.ndarray):
    """Vectorised (d (κ,), dd/dp (κ,3), dd/dt (κ,)) over κ sample points P.

    Same math as _signed_dist_and_grad, batched. Sphere is pure-numpy; AABB
    falls back to the per-point scalar routine (rare, cheap)."""
    P = np.asarray(P, dtype=np.float64)
    k = P.shape[0]
    if hasattr(obs, "centre0"):  # sphere (possibly moving)
        c = obs.centre0[None, :] + obs.vel[None, :] * t_abs[:, None]  # (κ,3)
        diff = P - c
        nrm = np.linalg.norm(diff, axis=1)                            # (κ,)
        n = diff / nrm[:, None]
        d = nrm - obs.radius
        dd_dt = -(n @ obs.vel)                                        # (κ,)
        return d, n, dd_dt
    # AABB (static): vectorised over the k samples, same math as the scalar path.
    lo = obs.lo[None, :]; hi = obs.hi[None, :]
    over = np.maximum(lo - P, 0.0) + np.maximum(P - hi, 0.0)   # (k,3) |violation|
    outside = np.any(over > 0.0, axis=1)                       # (k,)
    d = np.empty(k); ndp = np.zeros((k, 3))
    if np.any(outside):
        ov = over[outside]
        nrm = np.linalg.norm(ov, axis=1)                      # (m,) > 0
        sign = (np.where(P[outside] > hi, 1.0, 0.0)
                - np.where(P[outside] < lo, 1.0, 0.0))         # (m,3)
        d[outside] = nrm
        ndp[outside] = (ov * sign) / nrm[:, None]
    ins = ~outside
    if np.any(ins):
        cand = np.maximum(lo - P[ins], P[ins] - hi)           # (m,3) per-axis (<=0)
        ax = np.argmax(cand, axis=1)                          # L∞ active axis
        m = np.arange(cand.shape[0])
        d[ins] = cand[m, ax]
        loP = (lo - P[ins])[m, ax]; Phi = (P[ins] - hi)[m, ax]
        g = np.zeros((cand.shape[0], 3))
        g[m, ax] = np.where(loP >= Phi, -1.0, 1.0)
        ndp[ins] = g
    return d, ndp, np.zeros(k)


def _minco_dynamic_term(tr: MinjerkTraj, order: int, limit: float,
                        weight: float, opt: OptParams):
    """(cost, dcost_dc, dcost_dT_explicit) for a vel(order=1)/accel(order=2) hinge.

    integrand h = max(||deriv||-limit, 0)^2, cost = Σ_i (T_i/κ) Σ_j weight·h.
    ∂h/∂deriv = 2·over·(deriv/||deriv||).  No abs-time path (purely geometric),
    so explicit-T = (a) qweight + (b) moving-sample via dh/dtau (= 2 over (u·nextderiv))."""
    M = tr.M
    n = tr.NC * M
    kap = max(int(opt.kappa), 1)
    s, Bs, bidx = _fixed_basis(kap)
    cost = 0.0
    dcost_dc = np.zeros((n, 3))
    dT_exp = np.zeros(M)
    for i in range(M):
        Ti = float(tr.T[i])
        taus = s * Ti
        B0, B1, B2, B3 = _scale_basis(Bs, bidx, Ti)
        Bo = {0: B0, 1: B1, 2: B2, 3: B3}[order]
        Bnext = {0: B1, 1: B2, 2: B3}[order]    # d/dtau of the order-th deriv
        ci = tr.c[6 * i:6 * i + 6]
        D = Bo @ ci                              # (κ,3) the deriv being limited
        Dn = Bnext @ ci                          # (κ,3) its tau-derivative
        wq = Ti / kap
        mag = np.linalg.norm(D, axis=1)          # (κ,)
        over = mag - limit
        act = (over > 0.0) & (mag > 0.0)
        if not np.any(act):
            continue
        Da = D[act]; Dna = Dn[act]; Boa = Bo[act]; sa = s[act]
        maga = mag[act]; overa = over[act]
        u = Da / maga[:, None]                    # (na,3)
        h = overa * overa
        dh_dderiv = (2.0 * overa)[:, None] * u   # (na,3)
        cost += wq * weight * float(np.sum(h))
        dcost_dc[6 * i:6 * i + 6] += (wq * weight) * (Boa.T @ dh_dderiv)
        # (a) quadrature-weight
        dT_exp[i] += (1.0 / kap) * weight * float(np.sum(h))
        # (b) moving-sample: dh/dtau = dh_dderiv · d(deriv)/dtau
        dh_dtau = np.einsum("kd,kd->k", dh_dderiv, Dna)
        dT_exp[i] += wq * weight * float(np.sum(dh_dtau * sa))
    return cost, dcost_dc, dT_exp


# ===========================================================================
# Stage 3 — HARD (humans) = continuous-time convex-hull constraint via the
# Bernstein control points + augmented-Lagrangian (PHR) penalty.
#
# For each hard obstacle h, segment i, control point k:
#   g_{i,k,h} = d_safe_h - signed_dist(P_{i,k}, t_rep_i)   (<=0 means clear)
# PHR augmented-Lagrangian term:
#   L_ikh = (rho/2) max(0, lambda/rho + g)^2 - lambda^2/(2 rho)
#   z = lambda + rho*g; if z>0 (active): dL/dg = z, else 0.
#   dg/dP = -dd/dp = -n  (sphere n=(P-c)/||P-c||; AABB-out outside_vec/||.||).
# dL/dP_{i,k} accumulates z*(-n); then chained to dCost/dc and explicit dCost/dT
# through P_i = C2B @ D(T_i) @ c_i (the T_i^j scaling is the explicit-T trap).
# ===========================================================================


def _partition_obstacles(obstacles, avoid_cfg, override):
    """Split obstacles into (soft_obs, hard_obs) per resolve_mode + override."""
    soft_obs, hard_obs = [], []
    for o in obstacles:
        if resolve_mode(o, avoid_cfg, override) == "hard":
            hard_obs.append(o)
        else:
            soft_obs.append(o)
    return soft_obs, hard_obs


def _hard_d_safe(obs, avoid_cfg) -> float:
    """d_safe for a hard obstacle; fail-safe to the human default 0.8."""
    params = avoid_cfg.get(obs.class_name)
    if params is None:
        return 0.8
    return float(params.d_safe)


def _seg_rep_time(tr: MinjerkTraj, i: int) -> float:
    """Representative wall-clock time for segment i (Stage 3: segment END — the
    most conservative single time for a forward-moving human, and reduces to
    static for vel=0). Full per-control-point wall-clock is Stage 4."""
    return float(tr._cum[i + 1])


def _is_moving(obs) -> bool:
    """True for a sphere-human with nonzero CV velocity. This is the GATE that
    keeps STATIC humans byte-identical to Stage 3: the per-(i,k) space-time
    normal differs geometrically from the centroid normal even at vel=0, so a
    static human MUST run the old single-time centroid path."""
    return hasattr(obs, "centre0") and float(np.linalg.norm(obs.vel)) > 0.0


def _trust_mask(tr: MinjerkTraj, opt: OptParams) -> np.ndarray:
    """(M,6) bool: control point (i,k) is trusted-HARD iff t_{i,k} <= tau_trust.

    Offline harness: t_now = 0 so t_{i,k} = control_point_times() directly.
    FROZEN per outer ALM iter (same discipline as the supporting-halfspace
    normals); recomputed only between outer iters. An untrusted point is
    excluded from the hard set entirely (its g/w/S all go to zero)."""
    return tr.control_point_times() <= float(opt.tau_trust)


def _seg_normals(tr: MinjerkTraj, hard_obs, P=None, opt: OptParams = None):
    """Frozen supporting-halfspace normals per (segment i, control pt k, human h).

    Stage 3 (static human / spacetime off): ONE normal per (segment, obstacle)
        a_{i,h} = unit(centroid_k(P_{i,k}) - c_h(t_rep_i)).
    Stage 4 (S4a, MOVING human + opt.spacetime_hard): per-control-point
        a_{i,k,h} = unit(P_{i,k} - c_h(t_{i,k})) using control_point_times(),
        the "avoid at the right moment" tight normal.

    ALWAYS returns shape (M, 6, H, 3) for a uniform downstream contract: the
    static branch simply broadcasts the per-segment centroid normal across the
    6 control points (geometrically identical to the Stage-3 (M,H,3) form, just
    repeated), so g and dg/dP are unchanged for static obstacles."""
    M = tr.M
    H = len(hard_obs)
    if P is None:
        P = tr.control_points()
    spacetime = opt is not None and bool(opt.spacetime_hard)
    centroid = P.mean(axis=1)                                # (M,3) per-segment centroid
    t_rep = np.array([_seg_rep_time(tr, i) for i in range(M)])  # (M,)
    t_pt = tr.control_point_times()                         # (M,6)
    A = np.zeros((M, 6, H, 3))
    fallback = np.array([1.0, 0.0, 0.0])
    for h, obs in enumerate(hard_obs):
        if spacetime and _is_moving(obs):
            # per-control-point: anchor each P_{i,k} at the human's own time
            c = obs.centre0[None, None, :] + obs.vel[None, None, :] * t_pt[:, :, None]
            a = P - c                                        # (M,6,3)
        else:
            if hasattr(obs, "centre0"):
                c = obs.centre0[None, :] + obs.vel[None, :] * t_rep[:, None]  # (M,3)
            else:
                c = np.broadcast_to(0.5 * (obs.lo + obs.hi), (M, 3))
            a_seg = centroid - c                             # (M,3) one per segment
            a = np.broadcast_to(a_seg[:, None, :], (M, 6, 3))
        nrm = np.linalg.norm(a, axis=2)                      # (M,6)
        ok = nrm > 1e-12
        A[:, :, h, :] = np.where(ok[:, :, None],
                                 a / np.where(ok[:, :, None], nrm[:, :, None], 1.0),
                                 fallback)
    return A


def _alm_constraints(tr: MinjerkTraj, hard_obs, avoid_cfg, normals=None,
                     opt: OptParams = None, trust_mask=None):
    """Evaluate every (segment i, ctrl-pt k, hard-obs h) TIGHT constraint.

    SPHERE (static human / spacetime off) -> Stage-3 supporting-halfspace at the
      segment-END single time t_rep_i:
        g = R - a_{i,h}^T (P_{i,k} - c_h(t_rep)),  R = r_h + d_safe, dg/dP = -a.
    SPHERE (MOVING human, opt.spacetime_hard, S4a) -> per-control-point space-time
      halfspace at each point's OWN time t_{i,k}=control_point_times():
        g = R - a_{i,k,h}^T (P_{i,k} - c_h(t_{i,k})),  R = r_h + d_safe.
      This anchors every control point against the human's predicted position at
      the moment the drone reaches it ("avoid at the right moment", metric #1).
      It also carries dg/dt_{i,k} = +a_{i,k,h}^T vel_h (the moving-human absolute-
      time sensitivity that feeds Source 2 of dCost/dT in _alm_term).
    AABB (wall, all-hard ablation) -> per-control-point signed-dist surrogate:
        g = d_safe - signed_dist(P_ik),  dg/dP = -dd/dp,  dg/dt = 0.

    S4b TRUST WINDOW: under spacetime + a moving human, a control point whose
    time t_{i,k} > opt.tau_trust is EXCLUDED from the hard set (g set to a large-
    negative so w=0, trust=False); the stale-prediction tail is covered by the
    SOFT EGO field. The mask is FROZEN per outer iter via the call site.

    Returns (dict of flat arrays, P (M,6,3)).  `n` holds dg/dP's negative
    (i.e. the outward unit normal) so dL/dP = -w*n elsewhere stays uniform.
    `dgdt` holds dg/dt_{i,k} (nonzero only for trusted moving-sphere points);
    `trust` is the per-constraint boolean S4b mask."""
    M = tr.M
    H = len(hard_obs)
    P = tr.control_points()                      # (M,6,3)
    if normals is None:
        normals = _seg_normals(tr, hard_obs, P, opt=opt)
    spacetime = opt is not None and bool(opt.spacetime_hard)
    Nc = M * 6 * H
    # flat index m = (i*6 + k)*H + h
    seg = np.repeat(np.arange(M), 6 * H)
    kpt = np.tile(np.repeat(np.arange(6), H), M)
    hidx = np.tile(np.arange(H), M * 6)
    g = np.zeros(Nc); nrm = np.zeros((Nc, 3)); dist = np.zeros(Nc); Rarr = np.zeros(Nc)
    dgdt = np.zeros(Nc); trust = np.ones(Nc, dtype=bool)
    t_rep = np.array([_seg_rep_time(tr, i) for i in range(M)])   # (M,)
    t_pt = tr.control_point_times()                             # (M,6)
    idx_all = (np.arange(M)[:, None] * 6 + np.arange(6)[None, :]) * H  # (M,6) base idx
    G_NEG = -1.0e9   # large-negative g -> forced inactive (untrusted points)
    for h, obs in enumerate(hard_obs):
        d_safe = _hard_d_safe(obs, avoid_cfg)
        idx = idx_all + h                              # (M,6) flat indices for this h
        if hasattr(obs, "centre0"):                   # sphere -> tight halfspace
            R = float(obs.radius) + d_safe
            moving_st = spacetime and _is_moving(obs)
            if moving_st:
                # per-control-point times + per-control-point normals (S4a)
                c = (obs.centre0[None, None, :]
                     + obs.vel[None, None, :] * t_pt[:, :, None])    # (M,6,3)
            else:
                # Stage-3 single segment-END time, broadcast across k
                c = np.broadcast_to(
                    (obs.centre0[None, :] + obs.vel[None, :] * t_rep[:, None])[:, None, :],
                    (M, 6, 3))
            a = normals[:, :, h, :]                    # (M,6,3) frozen normals
            diff = P - c                               # (M,6,3)
            proj = np.einsum("mkd,mkd->mk", a, diff)   # a^T(P-c) per (i,k)
            g_h = R - proj                             # (M,6)
            dist_h = np.linalg.norm(diff, axis=2) - float(obs.radius)
            g[idx] = g_h
            nrm[idx] = a
            dist[idx] = dist_h
            Rarr[idx] = R
            if moving_st:
                # dg/dt_{i,k} = +a_{i,k,h} . vel_h  (POSITIVE sign; opposite dL/dP)
                dgdt[idx] = np.einsum("mkd,d->mk", a, obs.vel)
                # S4b trust mask: FROZEN form (trust_mask) preferred so a point
                # crossing tau_trust does NOT flip HARD<->SOFT inside L-BFGS-B
                # (keeps the inner cost smooth). Fall back to the current-T mask
                # only when no frozen mask is supplied (T-fixed contexts).
                if trust_mask is not None:
                    tmask = np.asarray(trust_mask, dtype=bool)
                elif opt is not None:
                    tmask = t_pt <= float(opt.tau_trust)   # (M,6)
                else:
                    tmask = np.ones((M, 6), dtype=bool)
                trust[idx] = tmask
                g[idx] = np.where(tmask, g_h, G_NEG)
                dgdt[idx] = np.where(tmask, dgdt[idx].reshape(M, 6), 0.0)
        else:                                          # AABB -> per-pt signed dist
            for i in range(M):
                for k in range(6):
                    m = (i * 6 + k) * H + h
                    d, ddp, _ = _signed_dist_and_grad(obs, P[i, k], float(t_rep[i]))
                    g[m] = d_safe - d; nrm[m] = ddp; dist[m] = d; Rarr[m] = d_safe
    return {"g": g, "n": nrm, "seg": seg, "kpt": kpt, "hidx": hidx,
            "dist": dist, "R": Rarr, "dgdt": dgdt, "trust": trust}, P


def _alm_term(tr: MinjerkTraj, hard_obs, avoid_cfg, lam: np.ndarray, rho: float,
              normals=None, opt: OptParams = None, trust_mask=None):
    """(cost, dcost_dc (6M,3), dcost_dT_explicit (M,), cons) for the ALM penalty.

    `normals` are the FROZEN supporting-halfspace normals (per outer iter); if
    None they are recomputed from tr (used for the standalone check_grad gate at
    a fixed (lambda,rho)). cons is reused for warm-start / certificate.

    dCost/dc is UNCHANGED from Stage 3 (c_h depends only on T, not on the MINCO
    coeffs c). dCost/dT_explicit has TWO sources:
      Source 1 (curve geometry): G . dP/dT  (Stage 3, always).
      Source 2 (NEW, moving-human absolute time): w * (a.vel) chained through
        dt_{i,k}/dT_j -> same-segment fraction k/5 + cross-segment UNIT for all
        j<i (reverse-cumsum). Identically 0 for static humans (dgdt=0)."""
    M = tr.M
    n6 = tr.NC * M
    dcost_dc = np.zeros((n6, 3))
    dT_exp = np.zeros(M)
    if not hard_obs:
        return 0.0, dcost_dc, dT_exp, None
    cons, P = _alm_constraints(tr, hard_obs, avoid_cfg, normals=normals, opt=opt,
                               trust_mask=trust_mask)
    g = cons["g"]
    z = lam + rho * g                            # (Nc,)
    active = z > 0.0
    # PHR cost: (rho/2) z^2 - lambda^2/(2 rho) on active, else -lambda^2/(2 rho)
    cost = 0.0
    cost += float(np.sum((0.5 / rho) * (z[active] ** 2)))
    cost -= float(np.sum((lam ** 2) / (2.0 * rho)))
    # dL/dg = z on active (0 otherwise); dL/dP_ik = z * dg/dP = z * (-n)
    w = np.where(active, z, 0.0)                 # (Nc,) — untrusted rows already w=0
    dLdP_seg = np.zeros((M, 6, 3))               # accumulate per-(seg,ctrl)
    seg = cons["seg"]; kpt = cons["kpt"]; nrm = cons["n"]
    contrib = (-w)[:, None] * nrm                # (Nc,3) = z * (-n)
    np.add.at(dLdP_seg, (seg, kpt), contrib)
    # chain dP -> dc and explicit dP/dT per segment (vectorized over M):
    # (A) implicit: dL/dc_i = D(T_i) * (C2B^T @ G_i)
    T = tr.T
    Dvec = np.stack([np.ones(M), T, T**2, T**3, T**4, T**5], axis=1)  # (M,6)
    CtG = np.einsum("kj,mkd->mjd", C2B, dLdP_seg)            # (M,6,3) = C2B^T @ G
    dc = Dvec[:, :, None] * CtG                              # (M,6,3)
    dcost_dc += dc.reshape(n6, 3)
    # (B) explicit-T Source 1: sum_{k,d} G_i[k,d] * dP_i/dT_i[k,d]
    dP_dT = tr.control_points_dT_explicit()                 # (M,6,3)
    dT_exp += np.einsum("mkd,mkd->m", dLdP_seg, dP_dT)
    # (B) explicit-T Source 2: moving-human absolute-time chain (S4a).
    # dL/dt_{i,k} = sum_h w * (a.vel) = w * dgdt; chain via dt_{i,k}/dT_j.
    dgdt = cons.get("dgdt")
    if dgdt is not None and np.any(dgdt != 0.0):
        S_flat = w * dgdt                                    # (Nc,) = dCost/dt per (i,k,h)
        S = np.zeros((M, 6))
        np.add.at(S, (seg, kpt), S_flat)                    # (M,6) summed over h
        kfrac = np.arange(6) / float(MinjerkTraj.DEG)       # k/5
        dT_abs = np.einsum("k,mk->m", kfrac, S)             # same-segment local fraction
        seg_sum = S.sum(axis=1)                             # (M,)
        # cross-segment: dt_{i,k}/dT_j = 1 for ALL j<i -> dT_abs[j] += sum_{i>j} seg_sum[i]
        if M > 1:
            dT_abs[:-1] += np.cumsum(seg_sum[::-1])[::-1][1:]
        dT_exp += dT_abs
    return cost, dcost_dc, dT_exp, cons


def _minco_cost_grad(x: np.ndarray, start: np.ndarray, goal: np.ndarray, M: int,
                     obstacles, avoid_cfg, opt: OptParams, T_target: float,
                     v0=None, a0=None, vf=None, af=None,
                     hard_obs=None, alm_lambda=None, alm_rho=0.0, alm_normals=None,
                     alm_trust_mask=None):
    """THE analytic value+gradient. x = [q.ravel(), T]; returns (f, grad) with
    grad ordered identically to x: [dCost/dq.ravel(), dCost/dT].

    `obstacles` here is the SOFT set (EGO field). `hard_obs`/`alm_lambda`/
    `alm_rho`/`alm_normals` activate the augmented-Lagrangian penalty for the
    HARD set (with FROZEN supporting-halfspace normals). `alm_trust_mask` is the
    FROZEN (M,6) S4b hard/soft mask (frozen alongside the normals per outer
    iter so the inner cost stays smooth as T moves)."""
    nq = 3 * (M - 1)
    q = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    T = x[nq:].astype(np.float64)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T, v0=v0, a0=a0, vf=vf, af=af)

    # accumulate the shared-hook terms (obstacle + vel + accel) into one adjoint
    n = tr.NC * M
    dcost_dc = np.zeros((n, 3))
    dT_exp = np.zeros(M)

    c_obs, gc, gT = _minco_obstacle_term(tr, obstacles, avoid_cfg, opt)
    dcost_dc += opt.w_obs * gc
    dT_exp += opt.w_obs * gT
    f = opt.w_obs * c_obs

    c_vel, gc, gT = _minco_dynamic_term(tr, 1, opt.vmax, opt.w_vel, opt)
    dcost_dc += gc; dT_exp += gT; f += c_vel

    c_acc, gc, gT = _minco_dynamic_term(tr, 2, opt.amax, opt.w_accel, opt)
    dcost_dc += gc; dT_exp += gT; f += c_acc

    # time anchor: pure explicit-T term
    Ttot = float(np.sum(T))
    rel = (Ttot - T_target) / max(T_target, 1e-9)
    f += opt.w_time * rel * rel
    dT_exp += 2.0 * opt.w_time * (Ttot - T_target) / (max(T_target, 1e-9) ** 2)

    # Stage 3: augmented-Lagrangian penalty for the HARD set (one more
    # contributor stacked into the SAME dcost_dc / dT_exp before the adjoint)
    if hard_obs and alm_rho > 0.0:
        c_alm, gc, gT, _ = _alm_term(tr, hard_obs, avoid_cfg, alm_lambda, alm_rho,
                                     normals=alm_normals, opt=opt,
                                     trust_mask=alm_trust_mask)
        dcost_dc += gc; dT_exp += gT; f += c_alm

    gq, gT_total = tr.grad_from_dcost_dc(dcost_dc, dcost_dT_explicit=dT_exp)

    # smoothness via the verified energy_grad (its own adjoint + explicit dJ/dT)
    J, gq_e, gT_e = tr.energy_grad()
    f += opt.w_smooth * J
    gq = gq + opt.w_smooth * gq_e
    gT_total = gT_total + opt.w_smooth * gT_e

    grad = np.concatenate([gq.ravel(), gT_total]) if M > 1 else gT_total.copy()
    return float(f), grad


# ---- MINCO seed + per-seed solve ------------------------------------------

def _minco_seed(seed_path: np.ndarray, opt: OptParams):
    """Init from an A* polyline: arc-length resample to M+1 pts, T0=len/vmax.

    Returns (start, goal, q0 (M-1,3), T0 (M,))."""
    seed_path = np.asarray(seed_path, dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(seed_path, axis=0), axis=1)
    path_len = float(seg_lens.sum())
    M = int(np.clip(round(len(seed_path) / 2.0), 2, opt.M_cap))
    wp = _resample_polyline(seed_path, M + 1)
    wp[0] = seed_path[0]; wp[-1] = seed_path[-1]   # pin exact endpoints
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    T0 = np.maximum(seg / max(opt.vmax, 1e-6), opt.dt_min * 2.0)
    start = wp[0].copy(); goal = wp[-1].copy()
    q0 = wp[1:-1].copy()
    return start, goal, q0, T0


# ---- Stage 3 feasibility = continuous-time convex-hull / dense clearance ---

def hard_clearance(tr: MinjerkTraj, hard_obs, avoid_cfg, K_eval: int = 2000):
    """Densely-sampled continuous-time min HARD-human clearance (signed_dist) and
    the worst per-class breach (d_safe - signed_dist).  Walls/soft excluded.

    Returns (min_clearance, max_breach). max_breach <= 0 means every hard
    obstacle is cleared by its d_safe at every one of the K_eval samples."""
    if not hard_obs:
        return float("inf"), float("-inf")
    ts = np.linspace(tr.t_start, tr.t_end, K_eval)
    pts = tr.eval(ts)
    min_clr = float("inf"); max_breach = float("-inf")
    for obs in hard_obs:
        d_safe = _hard_d_safe(obs, avoid_cfg)
        for j in range(K_eval):
            d = obs.signed_dist(pts[j], float(ts[j]))
            if d < min_clr:
                min_clr = d
            br = d_safe - d
            if br > max_breach:
                max_breach = br
    return float(min_clr), float(max_breach)


def _certificate_margin(cons) -> float:
    """Analytic convex-hull certificate margin = min over ctrl-pts of
    (signed_dist - d_safe) = -max_m g_m.  >= 0 => control polygon clears."""
    if cons is None:
        return float("inf")
    return float(-np.max(cons["g"]))


def _certificate_margin_spacetime(tr: MinjerkTraj, hard_obs, avoid_cfg,
                                  opt: OptParams, trust_mask=None) -> float:
    """CONTINUOUS-TIME clearance certificate for MOVING humans (S4a/S4b).

    Per segment i use ONE inflated-radius supporting halfspace:
        a_i = unit(centroid_k(P_{i,k}) - c_h(t_i^0)),  t_i^0 = _cum[i]
        R_i = r_h + d_safe + ||vel_h|| * T_i
        margin_i = min over trusted k of (a_i^T (P_{i,k} - c_h(t_i^0)) - R_i)
    By the inflated-radius theorem (start-time sphere Minkowski-inflated by
    ||vel||*T_i), margin >= 0 over a segment implies signed_dist(p(t),t)>=d_safe
    for EVERY t in that segment (not just the 6 nodes). Asserted only over
    TRUSTED control points (the trust window). Static humans / spacetime-off use
    _certificate_margin (the Stage-3 tight per-point form) instead.

    Returns the minimum margin over all (segment, trusted-k, moving-human)."""
    M = tr.M
    P = tr.control_points()                                  # (M,6,3)
    centroid = P.mean(axis=1)                                # (M,3)
    t0 = tr._cum[:M]                                         # (M,) segment-start times
    if trust_mask is None:
        trust_mask = _trust_mask(tr, opt)                   # (M,6)
    trust_mask = np.asarray(trust_mask, dtype=bool)
    margin = float("inf")
    for obs in hard_obs:
        if not (hasattr(obs, "centre0") and bool(opt.spacetime_hard)
                and _is_moving(obs)):
            continue
        d_safe = _hard_d_safe(obs, avoid_cfg)
        vnorm = float(np.linalg.norm(obs.vel))
        c0 = obs.centre0[None, :] + obs.vel[None, :] * t0[:, None]   # (M,3)
        a = centroid - c0                                   # (M,3)
        nrm = np.linalg.norm(a, axis=1)
        ok = nrm > 1e-12
        a = np.where(ok[:, None], a / np.where(ok[:, None], nrm[:, None], 1.0),
                     np.array([1.0, 0.0, 0.0]))
        diff = P - c0[:, None, :]                           # (M,6,3)
        proj = np.einsum("md,mkd->mk", a, diff)             # (M,6)
        R = float(obs.radius) + d_safe + vnorm * tr.T       # (M,)
        m_ik = proj - R[:, None]                            # (M,6)
        if np.any(trust_mask):
            margin = min(margin, float(np.min(m_ik[trust_mask])))
    return margin


def _build_certificates(tr: MinjerkTraj, hard_obs, avoid_cfg,
                        lam: np.ndarray, rho: float, normals=None,
                        opt: OptParams = None, trust_mask=None) -> list:
    """Per-(human,segment,ctrl-pt) drawable records for the interpretability API.

    For a MOVING human under spacetime the record anchors c_h at the control
    point's OWN time t_{i,k} (the avoid-at-the-right-moment instant) and exposes
    a `trust` bool = whether the point was a TRUSTED-hard ALM constraint (vs a
    stale-prediction point demoted to the SOFT field). Static humans keep the
    Stage-3 segment-END time (trust always True)."""
    if not hard_obs:
        return []
    cons, P = _alm_constraints(tr, hard_obs, avoid_cfg, normals=normals, opt=opt,
                               trust_mask=trust_mask)
    g = cons["g"]; z = lam + rho * g
    t_pt = tr.control_point_times()                          # (M,6)
    trust = cons["trust"]
    recs = []
    for m in range(g.size):
        i = int(cons["seg"][m]); k = int(cons["kpt"][m]); h = int(cons["hidx"][m])
        obs = hard_obs[h]
        if hasattr(obs, "centre0"):
            moving_st = (opt is not None and bool(opt.spacetime_hard)
                         and _is_moving(obs))
            t_eval = float(t_pt[i, k]) if moving_st else _seg_rep_time(tr, i)
            c_h = obs.predict(t_eval)
        else:
            c_h = 0.5 * (obs.lo + obs.hi)
        lam_m = float(lam[m])
        recs.append({
            "human_idx": h,
            "class": obs.class_name,
            "seg": i,
            "ctrl_pt": k,
            "P": P[i, k].copy(),
            "c_h": np.asarray(c_h, dtype=np.float64).copy(),
            "R": float(cons["R"][m]),
            "dist": float(cons["dist"][m]),
            "clearance": float(cons["dist"][m] - _hard_d_safe(obs, avoid_cfg)),
            "g": float(g[m]),
            "lambda": lam_m,
            "rho": float(rho),
            "active": bool(z[m] > 0.0),
            "trust": bool(trust[m]),
            "force": (lam_m * cons["n"][m]).copy(),
        })
    return recs


def _alm_solve(start, goal, q0, T0, soft_obs, hard_obs, avoid_cfg, opt: OptParams,
               v0=None, a0=None):
    """Augmented-Lagrangian outer loop wrapping the L-BFGS-B inner solve.

    Outer: lambda <- max(0, lambda + rho*g); grow rho on stalled violation;
    stop at max-violation < tol or outer cap. Warm-start x and lambda.

    Returns a candidate dict with the ALM telemetry + certificate."""
    M = T0.size
    nq = 3 * (M - 1)
    x = np.concatenate([q0.ravel(), T0]) if M > 1 else T0.copy()
    T_target = float(np.sum(T0))
    bounds = [(None, None)] * nq + [(opt.dt_min, None)] * M

    tr0 = MinjerkTraj.from_endpoints(start, goal, q0, T0, v0=v0, a0=a0)
    if hard_obs:
        cons0, _ = _alm_constraints(tr0, hard_obs, avoid_cfg)
        Nc = cons0["g"].size
    else:
        Nc = 0
    lam = np.zeros(Nc)
    rho = float(opt.alm_rho0)

    soft_args = (start, goal, M, list(soft_obs), avoid_cfg, opt, T_target, v0, a0)
    f0, _ = _minco_cost_grad(x, *soft_args)

    lambda_hist = []; rho_hist = []
    prev_viol = float("inf")
    outer_iters = 0
    total_inner_iters = 0
    result = None
    if not hard_obs:
        # no hard constraints -> a single unconstrained solve (pure soft / smooth)
        result = scipy.optimize.minimize(
            _minco_cost_grad, x, args=soft_args, jac=True, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": opt.maxiter, "gtol": 1e-6})
        x = result.x
        total_inner_iters = int(result.nit)
    else:
        normals = None
        for outer in range(opt.alm_outer_iters):
            outer_iters = outer + 1
            # FREEZE the supporting-halfspace normals from the current iterate
            # (GCOPTER SFC discipline) so g is linear in P during the inner solve
            qf = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
            Tf = x[nq:]
            tr = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0)
            normals = _seg_normals(tr, hard_obs, opt=opt)
            # FREEZE the S4b trust mask from the same iterate (so a point crossing
            # tau_trust during the inner solve does NOT flip HARD<->SOFT -> the
            # inner cost stays smooth; the active hard set changes only here).
            tmask = _trust_mask(tr, opt)
            # inner unconstrained L-BFGS-B at the current (lambda, rho, normals);
            # the ALM kwargs go through a closure (minimize passes positional args)
            result = scipy.optimize.minimize(
                lambda xx, lam=lam, rho=rho, nrm=normals, tm=tmask: _minco_cost_grad(
                    xx, *soft_args, hard_obs=hard_obs, alm_lambda=lam, alm_rho=rho,
                    alm_normals=nrm, alm_trust_mask=tm),
                x, jac=True, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": opt.alm_inner_maxiter, "gtol": 1e-7})
            x = result.x
            total_inner_iters += int(result.nit)
            qf = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
            Tf = x[nq:]
            tr = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0)
            # re-evaluate g at the SAME frozen normals + trust mask
            cons, _ = _alm_constraints(tr, hard_obs, avoid_cfg, normals=normals,
                                       opt=opt, trust_mask=tmask)
            g = cons["g"]
            # multiplier update (PHR): lambda <- max(0, lambda + rho*g)
            lam = np.maximum(0.0, lam + rho * g)
            max_viol = float(np.max(np.maximum(g, 0.0)))
            lambda_hist.append(float(np.max(lam)) if lam.size else 0.0)
            rho_hist.append(float(rho))
            if max_viol < opt.alm_viol_tol:
                break
            # STALL early-exit: if rho is already maxed AND the violation barely
            # improved, further outer iters won't help -> stop (surfaces as STOP).
            if rho >= opt.alm_rho_max and max_viol > 0.98 * prev_viol:
                break
            # grow rho only if the violation did not shrink enough (stalled)
            if max_viol > 0.5 * prev_viol:
                rho = min(rho * opt.alm_rho_grow, opt.alm_rho_max)
            prev_viol = max_viol

    qf = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    Tf = x[nq:]
    tr = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0)

    cons_final = None; max_viol_final = 0.0; cert_margin = float("inf")
    final_normals = None
    final_trust = None
    if hard_obs:
        # certificate margin is computed with FRESHLY-recomputed normals at the
        # final iterate (the tight halfspace test on the achieved curve)
        final_normals = _seg_normals(tr, hard_obs, opt=opt)
        final_trust = _trust_mask(tr, opt)
        cons_final, _ = _alm_constraints(tr, hard_obs, avoid_cfg,
                                         normals=final_normals, opt=opt,
                                         trust_mask=final_trust)
        max_viol_final = float(np.max(np.maximum(cons_final["g"], 0.0)))
        # MOVING humans -> the conservative inflated-radius single-halfspace gate
        # (continuous-time sound); static / spacetime-off -> the Stage-3 tight
        # per-point margin. Walls (AABB) are reported via the tight margin too.
        has_moving = any(opt.spacetime_hard and _is_moving(o) for o in hard_obs)
        if has_moving:
            cert_margin = _certificate_margin_spacetime(
                tr, hard_obs, avoid_cfg, opt, trust_mask=final_trust)
        else:
            cert_margin = _certificate_margin(cons_final)

    return {
        "tr": tr, "x": x, "lam": lam, "rho": rho,
        "outer_iters": outer_iters,
        "total_inner_iters": total_inner_iters,
        "max_violation": max_viol_final,
        "certificate_margin": cert_margin,
        "final_normals": final_normals,
        "final_trust": final_trust,
        "lambda_history": lambda_hist, "rho_history": rho_hist,
        "result": result, "f0": float(f0), "T_target": T_target,
        "T0": T0,
    }


def _optimise_one_minco(seed_path: np.ndarray, obstacles, avoid_cfg,
                        opt: OptParams, v0=None, a0=None) -> dict:
    """One ALM(hard) + soft-field solve from a seed polyline. Candidate dict.

    Class dispatch: obstacles split by resolve_mode(obs, avoid_cfg, override)
    into soft (EGO field) and hard (ALM convex-hull constraint)."""
    start, goal, q0, T0 = _minco_seed(seed_path, opt)
    M = T0.size
    nq = 3 * (M - 1)
    soft_obs, hard_obs = _partition_obstacles(obstacles, avoid_cfg, opt.avoid_override)

    sol = _alm_solve(start, goal, q0, T0, soft_obs, hard_obs, avoid_cfg, opt,
                     v0=v0, a0=a0)
    tr = sol["tr"]
    result = sol["result"]
    Tf = sol["x"][nq:]

    # feasibility = continuous-time clearance + kinematics, NOT the scipy flag.
    feas = dict(check_feasibility(tr, obstacles, avoid_cfg, opt))
    K_dense = max(int(opt.K), 200)
    # The SAFETY metric / STOP decision use the NATURAL hard set (override=None)
    # so the ablation arms are comparable: all-soft still reports the human
    # breach (a person is a person regardless of the ablation switch).
    _, safety_obs = _partition_obstacles(obstacles, avoid_cfg, None)
    min_clr, max_breach = hard_clearance(tr, safety_obs, avoid_cfg, K_eval=K_dense)

    # Clearance validity is judged ONLY against HARD obstacles. Soft obstacles
    # (walls) are *meant* to be grazed -- that is the whole per-class point --
    # so their proximity is recorded in feas metrics but must NOT invalidate the
    # trajectory. check_feasibility measures every obstacle uniformly, so a soft
    # graze can trip its clearance_violation; re-derive the verdict here from the
    # hard set, falling back to the (class-agnostic) kinematic checks.
    hard_violation = bool(safety_obs) and (max_breach > opt.clearance_tol)
    if hard_violation:
        feas["trajectory_valid"] = False
        feas["failure_reason"] = "clearance_violation"
    elif feas["failure_reason"] == "clearance_violation":
        if feas["max_v"] > opt.vmax * (1.0 + opt.vel_tol):
            feas["failure_reason"] = "velocity_violation"
        elif feas["max_a"] > opt.amax * (1.0 + opt.accel_tol):
            feas["failure_reason"] = "acceleration_violation"
        else:
            feas["failure_reason"] = None
        feas["trajectory_valid"] = feas["failure_reason"] is None

    certs = _build_certificates(tr, hard_obs, avoid_cfg, sol["lam"], sol["rho"],
                                normals=sol.get("final_normals"), opt=opt,
                                trust_mask=sol.get("final_trust"))
    avoid_modes = {}
    for o in obstacles:
        avoid_modes[o.class_name] = resolve_mode(o, avoid_cfg, opt.avoid_override)

    return {
        "spline": tr,
        "init_cost": float(sol["f0"]),
        "final_cost": float(result.fun) if result is not None else float(sol["f0"]),
        "iter": int(sol["total_inner_iters"]),
        "optimizer_success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "no-op",
        "dt0": float(T0.min()),
        "dt_final": float(Tf.min()),
        "T_target": float(sol["T_target"]),
        "T_final": float(np.sum(Tf)),
        "feasibility": feas,
        "M": int(M),
        "hard_violation": hard_violation,
        "hard_max_breach": float(max_breach),
        "hard_min_clearance": float(min_clr),
        "n_hard": len(hard_obs),
        "n_soft": len(soft_obs),
        "alm": {
            "outer_iters": sol["outer_iters"],
            "rho_final": float(sol["rho"]),
            "max_violation": float(sol["max_violation"]),
            "lambda_max": float(np.max(sol["lam"])) if sol["lam"].size else 0.0,
            "lambda_history": sol["lambda_history"],
            "rho_history": sol["rho_history"],
        },
        "hard_certificates": certs,
        "continuous_min_clearance": float(min_clr),
        "certificate_margin": float(sol["certificate_margin"]),
        "avoid_modes": avoid_modes,
    }


def plan_minco(astar_path: np.ndarray,
               obstacles: Iterable,
               avoid_cfg: Dict[str, AvoidParams],
               *,
               opt_params: OptParams = None,
               detour_cfg: DetourConfig = None,
               v0=None, a0=None) -> Tuple[MinjerkTraj, dict]:
    """MINCO analytic-gradient solve. Mirrors plan()'s signature + info keys.

    Decision vars = (q, T); endpoints pinned structurally (not penalised).
    Detour multi-start + feasibility-first selection reused verbatim from the
    B-spline path (they operate on the A* polyline, not the trajectory type).

    v0/a0 clamp the START velocity/acceleration (default rest). Pass the current
    executed state here for receding-horizon continuity, so a replan does not
    restart from rest each tick.
    """
    opt = opt_params or OptParams()
    detour_cfg = detour_cfg or DetourConfig()
    astar_path = np.asarray(astar_path, dtype=np.float64)
    if astar_path.ndim != 2 or astar_path.shape[0] < 2:
        raise ValueError(f"astar_path must be (M>=2, d), got {astar_path.shape}")

    obstacles_list = list(obstacles)
    seeds = _generate_detour_seeds(astar_path, obstacles_list, avoid_cfg, opt, detour_cfg)

    candidates: List[dict] = []
    for name, seed_path in seeds:
        try:
            rec = _optimise_one_minco(seed_path, obstacles_list, avoid_cfg, opt,
                                      v0=v0, a0=a0)
        except Exception:
            continue
        rec["seed_name"] = name
        candidates.append(rec)
    if not candidates:
        raise RuntimeError("all seeds failed to optimise")

    best = _select_best_minco(candidates)
    feas = best["feasibility"]
    info = {
        "trajectory_valid": feas["trajectory_valid"],
        "optimizer_success": best["optimizer_success"],
        "failure_reason": feas["failure_reason"],
        "feasibility": feas,
        "seed_used": best["seed_name"],
        "n_seeds_tried": len(candidates),
        "init_cost": best["init_cost"],
        "final_cost": best["final_cost"],
        "iter": best["iter"],
        "dt0": best["dt0"],
        "dt_final": best["dt_final"],
        "T_target": best["T_target"],
        "T_final": best["T_final"],
        "M": best["M"],
        "message": best["message"],
        "converged": best["optimizer_success"],  # legacy alias
        # --- Stage 3 interpretability + clearance certificate ---
        "n_hard": best["n_hard"],
        "n_soft": best["n_soft"],
        "hard_violation": best["hard_violation"],
        "hard_max_breach": best["hard_max_breach"],
        "alm": best["alm"],
        "hard_certificates": best["hard_certificates"],
        "continuous_min_clearance": best["continuous_min_clearance"],
        "certificate_margin": best["certificate_margin"],
        "avoid_modes": best["avoid_modes"],
    }
    return best["spline"], info
