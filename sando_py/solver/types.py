"""Dataclasses shared across solver variants.

Kept in its own file so callers can ``from sando_py.solver import SolverInfo``
without pulling in numpy / gurobipy / etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolveTimingBreakdown:
    """Per-stage wall-clock breakdown of the *winning* Gurobi solve.

    Mirrors the C++ ``SolverGurobi::SolveTimingBreakdown`` struct so the
    step-timing CSV columns line up 1:1 with the C++ output. All
    fields in **milliseconds**, default 0.
    """
    findDT_ms: float = 0.0
    setX_ms: float = 0.0
    polytopes_ms: float = 0.0
    dynamic_ms: float = 0.0
    objective_ms: float = 0.0
    mapsize_ms: float = 0.0
    callOptimizer_ms: float = 0.0
    postsolve_ms: float = 0.0


@dataclass
class SolverInfo:
    """Diagnostic info from a single ``solve(...)`` call.

    ``success`` mirrors the boolean return of the C++ Gurobi pass. The
    runner uses it to decide whether to commit the new trajectory or
    hold the previous one. ``cost`` (= QP objective value) and
    ``wall_time_s`` are informational — they show up in benchmark CSVs
    but are not part of the per-tick contract.

    ``timing`` is the per-stage breakdown of the *winning* Gurobi solve
    in the multi-factor parallel search. ``successful_factor`` is the
    time-allocation factor that won — both mirror the C++
    ``winning_solve_timing_`` / ``successful_factor_`` member vars.
    """

    success: bool = False
    cost: float = 0.0
    wall_time_s: float = 0.0
    timing: SolveTimingBreakdown = field(default_factory=SolveTimingBreakdown)
    successful_factor: float = 1.0
