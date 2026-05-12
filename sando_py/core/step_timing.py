"""Step-timing CSV writer — Python-end equivalent of the C++
``SANDO::writeStepTimingRow`` (sando.cpp:574-618).

Schema parity
-------------
Same 25 columns as the C++ output, same header order, same units (ms),
same NaN-on-failure convention. The two files can be diffed directly or
fed through the upstream ``src/sando/scripts/plot_step_timing.py``.

Default output path is ``/tmp/sando_py_step_timing.csv`` (distinct from
C++ to avoid clobbering when both run in the same process tree). Override
with the ``SANDO_PY_STEP_TIMING_CSV`` env var (set to empty string to
disable writing entirely).

Truncate-on-first-write
-----------------------
First call from a fresh process re-creates the file with the header;
subsequent calls append. Matches the C++ ``first_write_in_process``
flag so a long demo produces a single growing CSV per process.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# Column order must stay locked to the C++ schema — downstream tooling
# (plot_step_timing.py, pandas readers) keys on header text.
_HEADER = (
    "replan_id,t_wall_s,success,"
    "housekeeping_ms,global_planning_ms,"
    "hgp_static_jps_ms,hgp_check_path_ms,hgp_dynamic_astar_ms,hgp_recover_path_ms,"
    "cvx_decomp_ms,local_traj_opt_ms,"
    "gurobi_findDT_ms,gurobi_setX_ms,gurobi_polytopes_ms,gurobi_dynamic_ms,"
    "gurobi_objective_setup_ms,gurobi_mapsize_ms,gurobi_callOptimizer_ms,gurobi_postsolve_ms,"
    "append_ms,final_housekeeping_ms,total_replan_ms,"
    "successful_factor,hgp_final_g,gurobi_objective_value"
)


@dataclass
class StepTimingRow:
    """One row of step-timing data. Mirrors the 25-column C++ schema.

    Stages that don't apply to a particular failure mode are set to 0.0
    (C++ convention) — ``gurobi_objective_value`` is ``nan`` on any
    failure that occurred before Gurobi succeeded.
    """
    replan_id: int = 0
    t_wall_s: float = 0.0
    success: bool = False
    housekeeping_ms: float = 0.0
    global_planning_ms: float = 0.0
    hgp_static_jps_ms: float = 0.0
    hgp_check_path_ms: float = 0.0
    hgp_dynamic_astar_ms: float = 0.0
    hgp_recover_path_ms: float = 0.0
    cvx_decomp_ms: float = 0.0
    local_traj_opt_ms: float = 0.0
    gurobi_findDT_ms: float = 0.0
    gurobi_setX_ms: float = 0.0
    gurobi_polytopes_ms: float = 0.0
    gurobi_dynamic_ms: float = 0.0
    gurobi_objective_setup_ms: float = 0.0
    gurobi_mapsize_ms: float = 0.0
    gurobi_callOptimizer_ms: float = 0.0
    gurobi_postsolve_ms: float = 0.0
    append_ms: float = 0.0
    final_housekeeping_ms: float = 0.0
    total_replan_ms: float = 0.0
    successful_factor: float = 0.0
    hgp_final_g: float = float("nan")
    gurobi_objective_value: float = float("nan")

    def to_csv_line(self) -> str:
        """Render the row to one CSV line, matching the C++
        ``std::fixed << std::setprecision(6)`` formatting and the
        ``nan`` literal for failures."""
        def _f(x: float) -> str:
            if isinstance(x, bool):
                return "1" if x else "0"
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "nan"
            return f"{float(x):.6f}"
        return ",".join([
            str(int(self.replan_id)),
            _f(self.t_wall_s),
            "1" if self.success else "0",
            _f(self.housekeeping_ms),
            _f(self.global_planning_ms),
            _f(self.hgp_static_jps_ms),
            _f(self.hgp_check_path_ms),
            _f(self.hgp_dynamic_astar_ms),
            _f(self.hgp_recover_path_ms),
            _f(self.cvx_decomp_ms),
            _f(self.local_traj_opt_ms),
            _f(self.gurobi_findDT_ms),
            _f(self.gurobi_setX_ms),
            _f(self.gurobi_polytopes_ms),
            _f(self.gurobi_dynamic_ms),
            _f(self.gurobi_objective_setup_ms),
            _f(self.gurobi_mapsize_ms),
            _f(self.gurobi_callOptimizer_ms),
            _f(self.gurobi_postsolve_ms),
            _f(self.append_ms),
            _f(self.final_housekeeping_ms),
            _f(self.total_replan_ms),
            _f(self.successful_factor),
            _f(self.hgp_final_g),
            _f(self.gurobi_objective_value),
        ])


class StepTimingWriter:
    """Per-process appender for the step-timing CSV.

    The C++ writer is a static-singleton with a "truncate on first write"
    flag. We do the same with a module-level instance keyed on the file
    path, guarded by a lock so the writer is safe if multiple agents in
    the same Python process publish to the same path.
    """
    _instances: dict = {}
    _lock = threading.Lock()

    def __init__(self, path: str):
        self.path = path
        self._t_origin = time.perf_counter()
        self._first_write = True
        self._replan_id = 0
        self._write_lock = threading.Lock()

    @classmethod
    def get(cls, path: Optional[str] = None) -> Optional["StepTimingWriter"]:
        """Return the singleton writer for ``path``, or ``None`` if
        writing is disabled (path is empty / unset)."""
        if path is None:
            path = os.environ.get(
                "SANDO_PY_STEP_TIMING_CSV", "/tmp/sando_py_step_timing.csv",
            )
        if not path:
            return None
        with cls._lock:
            if path not in cls._instances:
                cls._instances[path] = cls(path)
            return cls._instances[path]

    def write(self, row: StepTimingRow) -> None:
        """Append ``row`` to the CSV, populating ``replan_id`` and
        ``t_wall_s`` from the writer's internal counter / timer."""
        with self._write_lock:
            self._replan_id += 1
            row.replan_id = self._replan_id
            row.t_wall_s = time.perf_counter() - self._t_origin
            mode = "w" if self._first_write else "a"
            try:
                with open(self.path, mode) as f:
                    if self._first_write:
                        f.write(_HEADER + "\n")
                        self._first_write = False
                    f.write(row.to_csv_line() + "\n")
            except OSError:
                # Disk failures shouldn't kill the planner; drop the row.
                pass

    def next_replan_id(self) -> int:
        """Peek at the next ``replan_id`` that will be assigned (useful
        for tests / debugging)."""
        return self._replan_id + 1
