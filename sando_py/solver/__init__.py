"""Local trajectory optimizer — solve for a smooth committed trajectory
through the global path's waypoints.

The C++ project uses a Gurobi QP (Hermite-spline parameterized, with
variable elimination and a dynamic time-allocation factor). The Python
port lands the v1 variant first — a closed-form cubic-Hermite spline
through the waypoints — and the full QP follows as v2.

See ``solver/README.md``.

Contract
--------
    solve(A:      RobotState,
          path:   List[np.ndarray],
          polys:  List[Polytope],
          params: Parameters) -> Tuple[PieceWisePol, SolverInfo]

``SolverInfo.success`` tells the runner whether to commit or hold the
previous trajectory.

Public API
----------
solve              — functional shim (v1 cubic Hermite), matches the contract above
solve_qp           — functional shim (v2 Gurobi QP), same signature
MinJerkSolver      — v1 class form (cubic Hermite through waypoints)
GurobiSolver       — v2 class form (QP, full state matching + corridor inequality)
SolverInfo         — dataclass with ``success`` / ``cost`` / ``wall_time_s``

v1 limitations (all fixed by v2)
--------------------------------
* Cubic per segment ⇒ start *acceleration* is not preserved.
* ``polys`` is accepted but the corridors are not enforced as
  constraints — the cubic spline can cut a sharp corner.
* ``v_max`` is enforced only at junction velocities, not along the
  intermediate spline.
"""

from .gurobi_solver import GurobiSolver, estimate_initial_dt, solve_qp
from .min_jerk import MinJerkSolver, solve
from .types import SolveTimingBreakdown, SolverInfo

__all__ = [
    "GurobiSolver",
    "estimate_initial_dt",
    "MinJerkSolver",
    "SolverInfo",
    "SolveTimingBreakdown",
    "solve",
    "solve_qp",
]
