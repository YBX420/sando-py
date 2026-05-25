"""End-to-end module-level harness for the dynamic-forest scenario.

Reproduces (in isolation, without ROS) the exact pipeline the live node
runs per replan tick — HGP global plan → time-layered SFC → Gurobi
solver — against a synthetic obstacle set that matches what
``dynamic_forest_node`` publishes for ``run_sim.py -m dynamic -d easy``.

Goal: identify which module produces a failure under realistic load.
The expectation, per the user-stated requirement, is that this Python
pipeline matches the C++ behaviour qualitatively (success rate, corridor
geometry, trajectory shape). The only acceptable drift is wall-clock
speed.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import pytest

# Skip the whole module if gurobipy isn't available — the solver harness
# is meaningless without it.
gp = pytest.importorskip("gurobipy")

from sando_py.core.config_loader import load_parameters_from_yaml
from sando_py.core.dyn_traj import DynTraj, TrajMode
from sando_py.core.types import Parameters, RobotState
from sando_py.core.utils import projectPointToSphere
from sando_py.decomp.sfc import sfc, sfc_time_layered
from sando_py.hgp import HGPPlanner
from sando_py.solver import GurobiSolver, estimate_initial_dt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _trefoil_strs(x0, y0, z0, sx, sy, sz, offset, slower) -> Tuple[str, ...]:
    """Same template as ``dyn_obstacles.launch.py::trefoil_expr_with_vel``."""
    tt = f"t/{slower}+{offset}"
    inv = f"(1/{slower})"
    x = f"{sx / 6.0}*(sin({tt})+2*sin(2*{tt}))+{x0}"
    y = f"{sy / 5.0}*(cos({tt})-2*cos(2*{tt}))+{y0}"
    z = f"{(sz / 2.0)}*(-sin(3*{tt}))+{z0}"
    vx = f"{sx / 6.0}*{inv}*(cos({tt})+4*cos(2*{tt}))"
    vy = f"{sy / 5.0}*{inv}*(-sin({tt})+4*sin(2*{tt}))"
    vz = f"-{3 * sz / 2.0}*{inv}*cos(3*{tt})"
    return x, y, z, vx, vy, vz


def _make_dynamic_forest(
    num_obstacles: int,
    *,
    x_range=(5.0, 105.0),
    y_range=(-6.0, 6.0),
    z_range=(0.5, 4.5),
    dynamic_ratio: float = 0.5,
    seed: int = 0,
) -> List[DynTraj]:
    """Replicate dyn_obstacles.launch.py spawn logic. Dynamic obstacles
    get trefoil analytic expressions; static obstacles get a constant
    position. Bbox sizes match the launch defaults (small cubes for
    dynamic, narrow pillars/walls for static)."""
    rng = random.Random(seed)
    bbox_dyn = [0.8, 0.8, 0.8]
    bbox_vert = [0.4, 0.4, 4.0]
    bbox_horiz = [0.4, 4.0, 0.4]
    num_dynamic = int(num_obstacles * dynamic_ratio)
    out: List[DynTraj] = []
    for i in range(num_obstacles):
        x = rng.uniform(*x_range); y = rng.uniform(*y_range); z = rng.uniform(*z_range)
        is_dyn = i < num_dynamic
        if is_dyn:
            sx = rng.uniform(2.0, 4.0); sy = rng.uniform(2.0, 4.0); sz = rng.uniform(2.0, 4.0)
            offset = rng.uniform(0.0, 3.0); slower = rng.uniform(10.0, 12.0)
            tx, ty, tz, tvx, tvy, tvz = _trefoil_strs(x, y, z, sx, sy, sz, offset, slower)
            bbox = bbox_dyn
        else:
            static_idx = i - num_dynamic
            num_static = num_obstacles - num_dynamic
            is_vert = static_idx < (num_static * 0.35)
            if is_vert:
                z = bbox_vert[2] / 2.0
                bbox = bbox_vert
            else:
                bbox = bbox_horiz
            tx, ty, tz = str(x), str(y), str(z)
            tvx, tvy, tvz = "0.0", "0.0", "0.0"

        dt_obj = DynTraj()
        dt_obj.id = i
        dt_obj.mode = TrajMode.Analytic
        dt_obj.traj_x = tx; dt_obj.traj_y = ty; dt_obj.traj_z = tz
        dt_obj.traj_vx = tvx; dt_obj.traj_vy = tvy; dt_obj.traj_vz = tvz
        dt_obj.bbox = np.array(bbox, dtype=float)
        dt_obj.is_agent = False
        ok = dt_obj.compileAnalytic()
        assert ok, f"trefoil compile failed for obstacle {i}"
        # Snapshot the t=0 position into current_pos (the live node refreshes
        # this each replan tick from the analytic expression).
        dt_obj.current_pos = dt_obj.evalAnalyticPos(0.0)
        out.append(dt_obj)
    return out


def _refresh_current_pos(obstacles: List[DynTraj], t_now: float) -> None:
    for o in obstacles:
        o.current_pos = o.evalAnalyticPos(t_now)


def _load_dynamic_params() -> Parameters:
    p, _ = load_parameters_from_yaml("config/sando.yaml")
    # Force the dynamic environment branch + skip viz-only knobs.
    p.environment_assumption = "dynamic"
    return p


def _make_state(pos) -> RobotState:
    s = RobotState()
    s.setPos(np.asarray(pos, dtype=float))
    return s


def _run_one_tick(params, planner, solver, obstacles, start_pos, goal, t_now,
                  *, project_to_horizon: bool = True):
    """Return (hgp_ok, sfc_ok, solver_ok, info).

    When ``project_to_horizon`` is True (default), the goal is projected
    onto a sphere of radius ``params.horizon`` around ``start_pos`` —
    matching what ``SandoNode._replan_cb`` does in live sim. Without
    this, a single tick is asked to plan a 30 m+ trajectory in one
    shot, which dynamics-wise needs ``v_max >> 2 m/s``.
    """
    _refresh_current_pos(obstacles, t_now)

    start_state = _make_state(start_pos)
    if project_to_horizon and params.horizon and params.horizon > 0:
        goal = projectPointToSphere(
            np.asarray(start_pos, dtype=float),
            np.asarray(goal, dtype=float),
            float(params.horizon),
        )
    path = planner.plan(start_state, goal, obstacles, t_now=t_now)
    if path is None:
        return False, False, False, {"reason": "hgp_none"}

    vm = planner.last_voxel_map
    num_N = max(1, int(params.num_N))
    A = RobotState()
    A.setPos(np.asarray(start_pos, dtype=float))

    initial_dt = estimate_initial_dt(A, np.asarray(path[-1]), params, num_N)
    factor_hint = float(params.dynamic_factor_initial_mean or 1.0)
    dt_layer = max(1e-3, initial_dt * factor_hint)

    polys = sfc_time_layered(
        path, vm, obstacles, params,
        num_time_layers=num_N,
        dt_layer=dt_layer,
        t_now=t_now,
    )
    if not polys or any(len(layer) == 0 for layer in polys):
        return True, False, False, {"reason": "sfc_empty", "n_path": len(path)}

    # Check the warm-start sub-segment endpoints actually live in *some*
    # polytope per time layer. If a layer has no polytope containing a
    # segment, the MIQP will be infeasible — surface that here.
    contained_layers = 0
    for n, layer in enumerate(polys):
        ok = any(
            np.all(p.A @ path[i] - p.b.flatten() <= 1e-6) and
            np.all(p.A @ path[i + 1] - p.b.flatten() <= 1e-6)
            for p in layer for i in range(min(1, len(path) - 1))
        )
        if ok:
            contained_layers += 1

    pwp, info = solver.solve(A, path, polys)
    return True, True, bool(info.success), {
        "reason": "ok" if info.success else "solver_fail",
        "info": info,
        "n_path": len(path),
        "n_polys_per_layer": len(polys[0]) if polys else 0,
        "n_layers": len(polys),
        "factors_tried": list(solver.time_alloc.factors_to_try()) if hasattr(solver, "time_alloc") else None,
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup_easy_forest():
    params = _load_dynamic_params()
    planner = HGPPlanner(params)
    solver = GurobiSolver(params)
    obstacles = _make_dynamic_forest(num_obstacles=50, seed=42)
    return params, planner, solver, obstacles


def test_pipeline_one_tick(setup_easy_forest, capsys):
    """Replan once from a start outside the obstacle cluster so the
    horizon-projected goal lands in free space. Catches structural
    pipeline regressions where HGP+SFC succeed but the MIQP is
    infeasible (the bug the dynamic-forest harness was written for)."""
    params, planner, solver, obstacles = setup_easy_forest
    # Start at y=-7 (outside the [-6,6] obstacle band) heading along +x.
    hgp_ok, sfc_ok, solver_ok, info = _run_one_tick(
        params, planner, solver, obstacles,
        start_pos=(0.0, -12.0, 2.0),
        goal=np.array([105.0, -12.0, 2.0]),
        t_now=0.0,
    )
    print(f"[one_tick] hgp_ok={hgp_ok} sfc_ok={sfc_ok} solver_ok={solver_ok} info={info}")
    assert hgp_ok, info
    assert sfc_ok, info
    assert solver_ok, info


def test_pipeline_many_ticks(setup_easy_forest, capsys):
    """20 replans along the open y=-7 lane while obstacles evolve via
    their trefoil expressions. Goals are horizon-projected each tick
    (matches ``SandoNode._replan_cb``). HGP failures here would mean
    the lane projection landed in an obstacle — legitimate in dense
    clusters but should be rare in the open lane. Solver failures here
    are the actual regression target."""
    params, planner, solver, obstacles = setup_easy_forest
    n_total = 0
    hgp_fail = sfc_fail = solver_fail = 0
    rows = []
    for i in range(20):
        t_now = float(i) * 0.25
        # Walk along the clear lane at y=-7 to avoid the obstacle band.
        x_now = 1.5 * i
        hgp_ok, sfc_ok, solver_ok, info = _run_one_tick(
            params, planner, solver, obstacles,
            start_pos=(x_now, -12.0, 2.0),
            goal=np.array([105.0, -12.0, 2.0]),
            t_now=t_now,
        )
        n_total += 1
        if not hgp_ok:
            hgp_fail += 1
        elif not sfc_ok:
            sfc_fail += 1
        elif not solver_ok:
            solver_fail += 1
        rows.append((i, t_now, hgp_ok, sfc_ok, solver_ok, info.get("reason")))

    print("\n[many_ticks] tick t_now hgp sfc solver reason")
    for r in rows:
        print(r)
    print(f"[many_ticks] hgp_fail={hgp_fail} sfc_fail={sfc_fail} solver_fail={solver_fail} / {n_total}")
    # Open-lane scenario: HGP and SFC should always succeed; solver may
    # fail in <= 10% of ticks (matches the C++ benchmark success rate
    # for dynamic-easy).
    assert hgp_fail == 0
    assert sfc_fail == 0
    assert solver_fail <= max(1, n_total // 10), f"solver fails {solver_fail}/{n_total}"
