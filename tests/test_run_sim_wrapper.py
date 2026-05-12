"""Tests for src/sando/scripts/run_sim.py — the demo launcher wrapper.

The wrapper has two backends (cpp / py). We test the YAML extraction
and the Python-backend injection without spawning subprocesses or
requiring the upstream colcon workspace to exist at test time.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Load the wrapper as a module (it lives outside the package tree)
# ---------------------------------------------------------------------------

_WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "sando" / "scripts" / "run_sim.py"
)


@pytest.fixture(scope="module")
def wrapper():
    spec = importlib.util.spec_from_file_location("run_sim_wrapper", _WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# YAML extraction
# ---------------------------------------------------------------------------

_SAMPLE_DRY_RUN = textwrap.dedent(
    """\
    [run_sim wrapper] forwarding to ...
    [INFO] Using setup.bash: install/setup.bash
    [INFO] Mode: Interactive (click goals in RViz)

    [DRY RUN] Generated YAML:
    ------------------------------------------------------------
    session_name: sando_sim
    windows:
    - window_name: main
      layout: tiled
      panes:
      - shell_command:
        - echo hello
    ------------------------------------------------------------
    """
)


def test_extract_yaml_basic(wrapper):
    yaml_text = wrapper.extract_yaml_from_dry_run(_SAMPLE_DRY_RUN)
    assert "session_name: sando_sim" in yaml_text
    assert "echo hello" in yaml_text
    # Should not include the surrounding [INFO] lines or the delimiters
    assert "INFO" not in yaml_text
    assert "------" not in yaml_text


def test_extract_setup_bash_short_flag(wrapper, tmp_path):
    """``-s install/setup.bash`` resolves against the workspace."""
    ws = tmp_path / "ws"
    (ws / "install").mkdir(parents=True)
    target = ws / "install" / "setup.bash"
    target.write_text("# stub\n")
    out = wrapper._extract_setup_bash(["-s", "install/setup.bash"], ws)
    assert out == target


def test_extract_setup_bash_long_flag(wrapper, tmp_path):
    ws = tmp_path / "ws"
    (ws / "install").mkdir(parents=True)
    target = ws / "install" / "setup.bash"
    target.write_text("# stub\n")
    out = wrapper._extract_setup_bash(["--setup-bash", "install/setup.bash"], ws)
    assert out == target


def test_extract_setup_bash_eq_form(wrapper, tmp_path):
    ws = tmp_path / "ws"
    out = wrapper._extract_setup_bash(
        ["--setup-bash=/abs/path/setup.bash"], ws,
    )
    assert str(out) == "/abs/path/setup.bash"


def test_extract_setup_bash_absent_returns_none(wrapper, tmp_path):
    assert wrapper._extract_setup_bash(["-m", "dynamic"], tmp_path) is None


def test_delay_goal_sender_pushes_sleep_past_warmup(wrapper):
    panes = [
        {"shell_command": [
            "sleep 8",
            "ros2 launch sando goal_sender.launch.py list_agents:=NX01",
        ]},
        {"shell_command": [
            "sleep 3",
            "ros2 launch sando rviz_only.launch.py num_obstacles:=50",
        ]},
    ]
    wrapper._delay_goal_sender(panes, 13)
    # goal_sender pane: leading sleep now 13
    assert panes[0]["shell_command"][0] == "sleep 13"
    # rviz_only pane: untouched
    assert panes[1]["shell_command"][0] == "sleep 3"


def test_delay_goal_sender_inserts_sleep_if_missing(wrapper):
    panes = [
        {"shell_command": [
            "ros2 launch sando goal_sender.launch.py list_agents:=NX01",
        ]},
    ]
    wrapper._delay_goal_sender(panes, 13)
    assert panes[0]["shell_command"][0] == "sleep 13"
    assert "goal_sender" in panes[0]["shell_command"][1]


def test_extract_yaml_missing_delimiters_raises(wrapper):
    with pytest.raises(RuntimeError, match="delimiters"):
        wrapper.extract_yaml_from_dry_run("no delimiters here, sorry")


# ---------------------------------------------------------------------------
# Python-backend YAML injection
# ---------------------------------------------------------------------------

_SAMPLE_YAML = textwrap.dedent(
    """\
    session_name: sando_sim
    windows:
    - window_name: main
      layout: tiled
      shell_command_before:
      - source /opt/ros/humble/setup.bash
      panes:
      - shell_command:
        - ros2 launch sando rviz_only.launch.py
      - shell_command:
        - sleep 3
        - ros2 launch sando onboard_sando.launch.py namespace:=NX01
    """
)


def test_inject_python_backend_appends_pane(wrapper, tmp_path):
    sando_py_root = tmp_path / "sando-py"
    sando_py_root.mkdir()
    config = sando_py_root / "config" / "sando.yaml"
    config.parent.mkdir()
    config.write_text("dummy: 1\n")

    out = wrapper.inject_python_backend(
        _SAMPLE_YAML,
        sando_py_root=sando_py_root,
        config_file=config,
        namespace="NX01",
        warmup_s="10",
    )

    import yaml
    data = yaml.safe_load(out)
    panes = data["windows"][0]["panes"]
    # Original 2 + injected 1
    assert len(panes) == 3
    last = panes[-1]["shell_command"]
    joined = "\n".join(last)
    assert "sleep 10" in joined
    assert "pkill -9 -x sando" in joined
    assert "python3 -m sando_py.sim_io" in joined
    assert "__ns:=/NX01" in joined
    assert f"config_file:={config}" in joined
    assert f"PYTHONPATH={sando_py_root}" in joined


def test_inject_python_backend_warmup_carries_through(wrapper, tmp_path):
    """The ``warmup_s`` kwarg appears verbatim as the injected
    ``sleep`` value."""
    config = tmp_path / "sando.yaml"
    config.write_text("x: 1\n")
    out = wrapper.inject_python_backend(
        _SAMPLE_YAML,
        sando_py_root=tmp_path,
        config_file=config,
        warmup_s="7",
    )
    # _SAMPLE_YAML has ``namespace:=NX01`` so the injected pane uses NX01
    # via auto-detect (which overrides the namespace= fallback arg). The
    # warmup, however, comes from the explicit kwarg.
    import yaml
    panes = yaml.safe_load(out)["windows"][0]["panes"]
    injected = panes[-1]["shell_command"]
    assert injected[0] == "sleep 7"
    assert "__ns:=/NX01" in "\n".join(injected)


def test_inject_python_backend_multi_agent_spawns_one_pane_per_namespace(wrapper, tmp_path):
    """When the upstream YAML mentions multiple agents
    (``namespace:=NX01``, ``namespace:=NX02``, ...), inject one Python
    sando_node pane per agent — the multi-agent ``--backend py`` flow."""
    import yaml

    multi_yaml = textwrap.dedent(
        """\
        session_name: sando_sim
        windows:
        - window_name: main
          layout: tiled
          panes:
          - shell_command:
            - ros2 launch sando simulator.launch.py
          - shell_command:
            - sleep 10
            - ros2 launch sando onboard_sando.launch.py namespace:=NX01 x:=10.0
          - shell_command:
            - sleep 10
            - ros2 launch sando onboard_sando.launch.py namespace:=NX02 x:=-5.0
          - shell_command:
            - sleep 10
            - ros2 launch sando onboard_sando.launch.py namespace:=NX03 x:=-5.0
        """
    )
    config = tmp_path / "sando.yaml"
    config.write_text("x: 1\n")
    out = wrapper.inject_python_backend(
        multi_yaml,
        sando_py_root=tmp_path,
        config_file=config,
    )
    data = yaml.safe_load(out)
    panes = data["windows"][0]["panes"]
    # Original 4 + 3 injected (one per agent)
    assert len(panes) == 7

    py_panes = panes[-3:]
    # Injected panes use ``__ns:=/NXnn`` (the ROS arg form), not the
    # upstream ``namespace:=`` form. Match that.
    import re
    ros_ns_re = re.compile(r"__ns:=/(\S+)")
    namespaces = []
    for pane in py_panes:
        joined = "\n".join(pane["shell_command"])
        m = ros_ns_re.search(joined)
        assert m, f"no namespace in pane: {pane}"
        namespaces.append(m.group(1))
    assert namespaces == ["NX01", "NX02", "NX03"]

    # Only the first injected pane carries the pkill — subsequent are
    # just sleep+launch
    assert "pkill -9 -x sando" in "\n".join(py_panes[0]["shell_command"])
    for p in py_panes[1:]:
        assert "pkill" not in "\n".join(p["shell_command"])

    # Per-agent agent_id assignment: 0, 1, 2
    for i, pane in enumerate(py_panes):
        joined = "\n".join(pane["shell_command"])
        assert f"agent_id:={i}" in joined


def test_inject_python_backend_falls_back_to_default_namespace(wrapper, tmp_path):
    """YAMLs without any ``namespace:=`` references (rare but possible
    for custom upstream tweaks) should still get a single Python pane
    using the configured fallback namespace."""
    import yaml

    plain_yaml = textwrap.dedent(
        """\
        session_name: sando_sim
        windows:
        - window_name: main
          layout: tiled
          panes:
          - shell_command:
            - echo hello
        """
    )
    config = tmp_path / "sando.yaml"
    config.write_text("x: 1\n")
    out = wrapper.inject_python_backend(
        plain_yaml,
        sando_py_root=tmp_path,
        config_file=config,
        namespace="CUSTOM_NS",
    )
    data = yaml.safe_load(out)
    panes = data["windows"][0]["panes"]
    assert len(panes) == 2  # original + 1 injected
    assert "__ns:=/CUSTOM_NS" in "\n".join(panes[-1]["shell_command"])


def test_inject_python_backend_rejects_malformed_yaml(wrapper, tmp_path):
    """Inputs that don't have ``windows`` should raise rather than write
    a silently broken YAML."""
    config = tmp_path / "sando.yaml"
    config.write_text("x: 1\n")
    with pytest.raises(RuntimeError, match="Unexpected upstream YAML"):
        wrapper.inject_python_backend(
            "not_a_yaml: true\n",
            sando_py_root=tmp_path,
            config_file=config,
        )


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def test_find_sando_py_root_resolves_to_repo_root(wrapper):
    root = wrapper._find_sando_py_root()
    # The repo root should contain pyproject.toml
    assert (root / "pyproject.toml").is_file()
    assert (root / "sando_py").is_dir()
