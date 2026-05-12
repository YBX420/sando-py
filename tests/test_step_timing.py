"""Regression tests for the step-timing CSV writer — schema parity with
C++ ``SANDO::writeStepTimingRow`` (sando.cpp:574-618)."""

from __future__ import annotations

import math
import os
from pathlib import Path

from sando_py.core.step_timing import (
    StepTimingRow,
    StepTimingWriter,
    _HEADER,
)


# The C++ schema. If the column ordering ever drifts the upstream
# ``plot_step_timing.py`` will silently parse wrong fields, so we lock it.
_EXPECTED_COLUMNS = [
    "replan_id", "t_wall_s", "success",
    "housekeeping_ms", "global_planning_ms",
    "hgp_static_jps_ms", "hgp_check_path_ms",
    "hgp_dynamic_astar_ms", "hgp_recover_path_ms",
    "cvx_decomp_ms", "local_traj_opt_ms",
    "gurobi_findDT_ms", "gurobi_setX_ms",
    "gurobi_polytopes_ms", "gurobi_dynamic_ms",
    "gurobi_objective_setup_ms", "gurobi_mapsize_ms",
    "gurobi_callOptimizer_ms", "gurobi_postsolve_ms",
    "append_ms", "final_housekeeping_ms", "total_replan_ms",
    "successful_factor", "hgp_final_g", "gurobi_objective_value",
]


def test_header_has_exactly_25_columns():
    cols = _HEADER.split(",")
    assert len(cols) == 25
    assert cols == _EXPECTED_COLUMNS


def test_csv_line_has_25_fields():
    row = StepTimingRow(replan_id=1, success=True, total_replan_ms=12.345)
    line = row.to_csv_line()
    fields = line.split(",")
    assert len(fields) == 25


def test_success_renders_as_1_failure_as_0():
    row_ok = StepTimingRow(success=True)
    row_fail = StepTimingRow(success=False)
    assert row_ok.to_csv_line().split(",")[2] == "1"
    assert row_fail.to_csv_line().split(",")[2] == "0"


def test_failure_objective_value_renders_as_nan():
    row = StepTimingRow(success=False)
    line = row.to_csv_line()
    fields = line.split(",")
    # gurobi_objective_value is the last column (index 24)
    assert fields[24] == "nan"
    # hgp_final_g defaults to nan too (column 23, index 23)
    assert fields[23] == "nan"


def test_floats_use_6_decimal_places():
    row = StepTimingRow(total_replan_ms=1.5)
    line = row.to_csv_line()
    # total_replan_ms is column 21 (0-indexed)
    fields = line.split(",")
    assert fields[21] == "1.500000"


def test_writer_creates_file_with_header(tmp_path):
    """First write truncates + adds header."""
    p = tmp_path / "test_timing.csv"
    w = StepTimingWriter(str(p))
    row = StepTimingRow(success=True, total_replan_ms=10.0)
    w.write(row)
    assert p.exists()
    content = p.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == _HEADER
    # replan_id auto-incremented from 0 → 1
    assert lines[1].startswith("1,")


def test_writer_appends_on_subsequent_writes(tmp_path):
    p = tmp_path / "test_timing.csv"
    w = StepTimingWriter(str(p))
    for _ in range(3):
        w.write(StepTimingRow(success=True))
    lines = p.read_text().strip().split("\n")
    # header + 3 data rows
    assert len(lines) == 4
    # replan_id 1, 2, 3
    ids = [int(line.split(",")[0]) for line in lines[1:]]
    assert ids == [1, 2, 3]


def test_writer_get_returns_singleton_per_path(tmp_path):
    """``StepTimingWriter.get(path)`` should return the same instance
    across calls so the counter / first-write flag stay consistent."""
    p = tmp_path / "test_timing.csv"
    w1 = StepTimingWriter.get(str(p))
    w2 = StepTimingWriter.get(str(p))
    assert w1 is w2


def test_writer_disabled_when_path_empty():
    """Empty path = writer is None = no CSV written (env var override
    contract)."""
    assert StepTimingWriter.get("") is None


def test_writer_env_var_override(tmp_path, monkeypatch):
    p = tmp_path / "via_env.csv"
    monkeypatch.setenv("SANDO_PY_STEP_TIMING_CSV", str(p))
    # Reset the cached instance so the env var is re-read.
    StepTimingWriter._instances.pop(str(p), None)
    w = StepTimingWriter.get()
    assert w is not None
    assert w.path == str(p)
