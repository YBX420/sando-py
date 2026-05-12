# solver/ — local trajectory optimizer

Given a start state `A`, the global path, and the SFCs, find a smooth
committed trajectory that
  - starts at `A` (matching pos, vel, accel),
  - stays inside each corridor for the duration assigned to that
    corridor's segment,
  - respects `v_max` / `a_max` / `j_max`,
  - minimizes integrated jerk.

## Approach

The C++ project uses a **Gurobi QP**:
  - Hermite-spline parameterization of the trajectory,
  - variable elimination so only intermediate waypoints are free,
  - linear inequality constraints per corridor (Ax <= b),
  - dynamic time-allocation factor that adapts on convergence failure.

The Python port lands in two passes:

| version | implementation                                                                       | status        |
|---------|--------------------------------------------------------------------------------------|---------------|
| v1      | closed-form **cubic Hermite** through waypoints (no inequality constraints).         | implemented   |
| v2      | full QP via `gurobipy` — cubic per polynomial-segment, full state matching, corridor inequality. | implemented |

### v1 caveats (all fixed by v2)
  - cubic per segment ⇒ start *acceleration* is dropped (only 4 DOFs per axis, all spent on pos+vel at both endpoints);
  - corridors are not enforced — the cubic spline can clip sharp corners of an SFC;
  - `v_max` is enforced only at junction velocities, not along the intermediate cubic.

### v2 details
  - Cubic per polynomial-segment (matches `PieceWisePol`).
  - Variables: 4 coefficients × 3 axes × N segments.
  - Each path edge is subdivided into `K = ceil(num_N / P)` polynomial sub-segments so the QP has free DOFs for jerk minimization (with all 6 boundary + 3·(n-1) continuity equality constraints in place, DOF per axis = `4n − 3n − 3 = n − 3`).
  - Start and end states fully anchored (pos, vel, accel).
  - C0 / C1 / C2 continuity at every internal junction.
  - `v_max` / `a_max` sampled at `_SAMPLES_PER_SEG` control points per segment per axis. `j_max` is exact: jerk of a cubic is constant per segment, one constraint per axis suffices.
  - **Corridor membership via Bezier convex-hull**: each cubic is converted to its 4 Bezier control points (a linear transformation) and each control point is constrained to lie in the corridor. The convex-hull property guarantees the *entire* segment stays in the polytope — strictly tighter than sampling, and with fewer constraints (4 per segment per polytope row vs `_SAMPLES_PER_SEG + 1`). `parent_of_seg` routes each sub-segment to its parent path-edge's corridor.
  - Objective: `sum_seg sum_axis 36·a²·T` plus an optional `jerk_smooth_weight · b²` regulariser.
  - Warm-started with a v1 cubic-Hermite seed (faster Gurobi convergence; the QP is convex so the optimum is unchanged).
  - **Persistent model**: the `Model` + decision variables are cached per `n_seg` and reused across replan ticks; only the constraints are cleared (batched `model.remove(...)`) and rebuilt per call. Mirrors the C++ `setDynamicConstraintsFaster_` pattern and preserves warm-start info via the variables' `.X`. In Python the win is small in practice (constraint construction dominates over variable allocation), but the structure makes the C++ MINVO / variable-elimination ports cleaner to slot in later.
  - Falls back to `success=False` on infeasibility / timeout / no Gurobi license — the runner switches to v1 (`MinJerkSolver`) in that branch.

### v2 alignment with C++

  - **Bezier corridor enforcement**: matches the C++ `SolverGurobi::setPolyConsts` / `getCP0..3`. Both use the Bezier convex-hull property: 4 control points per cubic segment, each constrained to lie in the corridor. The C++ separately computes MINVO control points in `checkCollisionViolation` for **post-solve verification only** — not for the Gurobi inequality constraints. We follow the same split: Bezier for enforcement, MINVO available in `core.BasisConverter` for verifiers.

  - **Variable elimination**: the C++ parameterises by interior waypoints only (3 free vars × 3 axes for N=6) via hardcoded `computeDependentCoefficientsN4` / `N6` derivations. The Python port keeps all 4 coefficients × 3 axes × N segments as free Gurobi variables and adds the boundary + continuity constraints as equalities. **Gurobi's presolve eliminates equality-constrained variables automatically** before optimisation — the resulting reduced QP is mathematically identical to the C++ formulation, so the optimum is the same. The difference is purely build-time perf (more `addVar` / `addConstr` calls in Python), not optimisation behaviour.

  - **Parallel factor search**: matches the C++ `sando.cpp::tryToCompleteOpt` pattern — when the adapter window has multiple factors, all candidates run concurrently via a `ThreadPoolExecutor` (capped at `_MAX_PARALLEL_WORKERS = 8`); the lowest-factor success wins. Each worker builds a fresh `Model` (the cached model is single-thread only). In Python the wall-clock benefit over sequential is small in practice — Gurobi serialises operations through the shared `Env`, and there's no per-worker license overhead but also no true parallelism — but the structure is identical to the C++ `std::async` fan-out so the behaviour matches when concurrency does kick in.

## Contract

```python
solve(A:      RobotState,
      path:   List[np.ndarray],
      polys:  List[Polytope],
      params: Parameters) -> Tuple[PieceWisePol, SolverInfo]
```

`solve` is the v1 free function; `solve_qp` is the v2 free function;
`MinJerkSolver` and `GurobiSolver` are the class forms (stateful across
replan ticks for the shared Gurobi env).

`SolverInfo` carries `success: bool`, `cost: float`, and
`wall_time_s: float`. The runner uses `success` to decide whether to
commit the new trajectory or hold the previous one (mirroring the C++
`sando.cpp` behaviour on Gurobi solve failure).

## Files

| file               | status        | role                                                                              |
|--------------------|---------------|-----------------------------------------------------------------------------------|
| `types.py`         | implemented   | `SolverInfo` dataclass (`success` / `cost` / `wall_time_s`)                       |
| `min_jerk.py`      | implemented   | cubic-Hermite spline through waypoints (v1)                                       |
| `gurobi_solver.py` | implemented   | full QP via gurobipy (v2)                                                         |
| `time_alloc.py`    | implemented   | dynamic-factor adapter — window of factors that recenters on success / shifts up on failure |

v1 covered by `tests/test_solver.py`, v2 by `tests/test_solver_gurobi.py`, adapter by `tests/test_time_alloc.py`.

Parameters consumed (from `core.Parameters`):
  `v_max`, `a_max`, `j_max`, `num_P`, `num_N`, `horizon`,
  `max_gurobi_comp_time_sec`, `jerk_smooth_weight`,
  `using_variable_elimination`, `use_dynamic_factor`,
  `dynamic_factor_initial_mean`, `factor_initial`, `factor_final`,
  `factor_constant_step_size`, `dynamic_factor_k_radius`.

Of these, v2 currently uses `v_max`, `a_max`, `j_max`, `num_N`,
`max_gurobi_comp_time_sec`, `jerk_smooth_weight`, and the full
`use_dynamic_factor` / `dynamic_factor_initial_mean` /
`dynamic_factor_k_radius` / `factor_initial` / `factor_final` /
`factor_constant_step_size` group via `TimeAlloc`.
`using_variable_elimination` is a no-op in our port because Gurobi
presolve already performs the elimination automatically (see the
alignment section above).
