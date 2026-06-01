"""Smoke import test — verify the package and its core modules import cleanly.

Excludes ROS-dependent node modules; those are exercised separately by the
launch graph. Gurobi is also lazy-imported so we don't require a license here.
"""
import importlib

import pytest


def test_package_imports():
    """Top-level package must import."""
    pkg = importlib.import_module("sando_py")
    assert pkg is not None


@pytest.mark.parametrize("name", [
    "sando_py.types",
    "sando_py.utils",
    "sando_py.solver_gurobi",
    "sando_py.planner",
    "sando_py.hgp.voxel_map",
    "sando_py.hgp.graph_search",
    "sando_py.hgp.hgp_planner",
    "sando_py.hgp.hgp_manager",
])
def test_module_imports(name):
    """Each core module must import without ROS or Gurobi installed."""
    m = importlib.import_module(name)
    assert m is not None


def test_planner_construction():
    """SANDO must instantiate with default parameters."""
    from sando_py.planner import SANDO
    from sando_py.types import Parameters

    par = Parameters()
    par.num_N = 5
    sando = SANDO(par)
    assert sando is not None
    assert sando.factors  # default sweep should not be empty
    assert sando.get_drone_status() == 3  # GOAL_REACHED
