"""Dataclasses shared across solver variants.

Kept in its own file so callers can ``from sando_py.solver import SolverInfo``
without pulling in numpy / gurobipy / etc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolverInfo:
    """Diagnostic info from a single ``solve(...)`` call.

    ``success`` mirrors the boolean return of the C++ Gurobi pass. The
    runner uses it to decide whether to commit the new trajectory or
    hold the previous one. ``cost`` and ``wall_time_s`` are
    informational — they show up in benchmark CSVs but are not part of
    the per-tick contract.
    """

    success: bool = False
    cost: float = 0.0
    wall_time_s: float = 0.0
