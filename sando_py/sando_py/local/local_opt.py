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

from .avoid_config import AvoidParams
from .bspline import UniformBSpline
from .cost import obstacle_cost


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
        d_safe = avoid_cfg[obs.class_name].d_safe
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

def check_feasibility(spline: UniformBSpline, obstacles, avoid_cfg,
                      opt: OptParams, K_eval: int = 200) -> dict:
    """Compute trajectory-level feasibility, independent of optimiser convergence.

    Returns dict with min_clearance / max_clearance_violation / max_v / max_a /
    vel_violation / accel_violation / trajectory_valid / failure_reason.
    """
    ts = np.linspace(spline.t_start, spline.t_end, K_eval)
    pts = spline.eval(ts)
    min_clr = float("inf")
    max_violation = 0.0
    for obs in obstacles:
        d_safe = avoid_cfg[obs.class_name].d_safe
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
