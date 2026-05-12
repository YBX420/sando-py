#!/usr/bin/env python3
"""Wrapper around the upstream SANDO C++ demo launcher (task #16).

Two backends:

  --backend cpp  (default)
      Forward to the upstream ``scripts/run_sim.py`` unchanged. tmuxp
      brings up RViz / Gazebo / fake_sim / dynamic_forest_node and the
      C++ ``sando_node`` planner. Pre-existing behaviour.

  --backend py
      Same demo world, but the C++ planner is replaced by the Python
      ``sando_py.sim_io`` node. We do this by:
        1. invoking the upstream launcher with ``--dry-run`` to capture
           its generated tmuxp YAML,
        2. appending a pane that waits for the upstream launch to
           settle, kills the C++ ``sando`` executable, and starts
           ``python -m sando_py.sim_io`` with the same namespace and
           config-file parameters,
        3. handing the modified YAML to ``tmuxp load``.

Usage (from /home/boxuan/code/sando-py):
    python3 src/sando/scripts/run_sim.py -m dynamic -d hard -s install/setup.bash
    python3 src/sando/scripts/run_sim.py -m dynamic -d hard -s install/setup.bash --backend py
    python3 src/sando/scripts/run_sim.py -m interactive -s install/setup.bash --backend py --dry-run

Environment overrides:
    $SANDO_WS_DIR        — upstream colcon workspace location
    $SANDO_PY_CONFIG     — config file for the Python sando_node (default: <repo>/config/sando.yaml)
    $SANDO_PY_NAMESPACE  — ROS namespace (default: NX01 — matches the upstream single-agent launches)
    $SANDO_PY_WARMUP_S   — seconds to wait before swapping in the Python planner (default: 10)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

# The upstream launcher fences the YAML in its --dry-run output between
# two ``-`` × 60 dashed lines.
_DRY_RUN_DELIM = "-" * 60

# Default namespace / warmup. Both overridable via env so the wrapper can
# stay short while still being adapted to multi-agent / slow-boot setups.
_DEFAULT_NAMESPACE = "NX01"
_DEFAULT_WARMUP_S = "10"

# Pattern for ``namespace:=NX01`` arguments inside ``ros2 launch sando
# onboard_sando.launch.py ...`` invocations. Used to auto-detect
# multi-agent setups so we spawn one Python sando_node per agent.
_NS_RE = re.compile(r"namespace:=(\S+)")


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def _find_sando_ws() -> Path:
    """Locate the colcon workspace that contains the upstream launcher."""
    env_override = os.environ.get("SANDO_WS_DIR")
    if env_override:
        ws = Path(env_override).expanduser().resolve()
        if (ws / "src" / "sando" / "scripts" / "run_sim.py").is_file():
            return ws
        print(
            f"[run_sim wrapper] $SANDO_WS_DIR={ws} does not contain "
            f"src/sando/scripts/run_sim.py",
            file=sys.stderr,
        )
        sys.exit(2)

    candidates = [
        Path.home() / "code" / "sando_ws",
        Path(__file__).resolve().parents[4] / "sando_ws",
    ]
    for ws in candidates:
        if (ws / "src" / "sando" / "scripts" / "run_sim.py").is_file():
            return ws.resolve()

    print(
        "[run_sim wrapper] could not locate the sando colcon workspace.\n"
        "  Searched:\n    " + "\n    ".join(str(c) for c in candidates) + "\n"
        "  Set $SANDO_WS_DIR to override.",
        file=sys.stderr,
    )
    sys.exit(2)


def _find_sando_py_root() -> Path:
    """Repo root for sando-py (the directory containing ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# YAML extraction / injection
# ---------------------------------------------------------------------------

def extract_yaml_from_dry_run(stdout: str) -> str:
    """Pull the YAML body out of the upstream launcher's ``--dry-run``
    stdout. Raises ``RuntimeError`` if the expected delimiters aren't
    found — easier to fix than a silent mis-extraction.
    """
    parts = stdout.split(_DRY_RUN_DELIM)
    if len(parts) < 3:
        raise RuntimeError(
            f"Could not find YAML delimiters '{_DRY_RUN_DELIM}' in upstream output. "
            f"Got:\n{stdout}"
        )
    # The YAML is the middle chunk. Surrounding chunks are pre/post text.
    return parts[1].strip("\n")


def _extract_namespaces(panes: list) -> List[str]:
    """Find every ``namespace:=NXnn`` referenced in the YAML's panes
    so we can spawn one Python sando_node per agent. Preserves order
    of first appearance; deduplicates."""
    seen: List[str] = []
    for pane in panes:
        cmds = pane.get("shell_command", [])
        if isinstance(cmds, str):
            cmds = [cmds]
        for cmd in cmds:
            for m in _NS_RE.finditer(str(cmd)):
                ns = m.group(1)
                if ns not in seen:
                    seen.append(ns)
    return seen


def inject_python_backend(
    yaml_text: str,
    *,
    sando_py_root: Path,
    config_file: Path,
    namespace: str = _DEFAULT_NAMESPACE,
    warmup_s: str = _DEFAULT_WARMUP_S,
) -> str:
    """Append panes to the tmuxp YAML that run the Python sando_node
    in place of the C++ planner.

    Auto-detects all ``namespace:=NXnn`` references in the upstream
    YAML (single-agent → one namespace, multi-agent → one per agent).
    Falls back to ``namespace`` (the ``$SANDO_PY_NAMESPACE`` default)
    when nothing is detected — handy for non-standard upstream setups.

    Layout: one extra pane per agent, each of which waits ``warmup_s``,
    pkills the C++ ``sando`` executable (one shared kill that covers
    all agents, redundant for the first pane only), then runs
    ``python3 -m sando_py.sim_io`` with the right namespace and
    config-file parameters. The upstream support nodes (fake_sim,
    rviz, dyn_obstacles, goal_relay) are left untouched — they don't
    conflict with the Python planner.
    """
    import yaml

    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict) or "windows" not in data or not data["windows"]:
        raise RuntimeError(f"Unexpected upstream YAML shape: {data!r}")

    panes = data["windows"][0].setdefault("panes", [])
    namespaces = _extract_namespaces(panes) or [namespace]

    # ``goal_sender`` only publishes once and then exits. The upstream
    # YAML schedules it before ``warmup_s`` so the C++ ``sando`` planner
    # can pick the goal up — but the Python planner only starts AFTER
    # warmup, so it misses the latched-but-not-actually-latched goal.
    # Push any ``goal_sender`` panes a few seconds past ``warmup_s`` so
    # the Python node has time to subscribe first.
    _delay_goal_sender(panes, int(warmup_s) + 3)

    for i, ns in enumerate(namespaces):
        # Stagger pkill so it doesn't run more than once per pkill window
        # (first pane carries the pkill; subsequent panes only sleep
        # and launch). pkill is idempotent so duplicates are harmless,
        # but skipping them avoids spurious warnings in the logs.
        cmds = [f"sleep {warmup_s}"]
        if i == 0:
            cmds.append("pkill -9 -x sando 2>/dev/null || true")
        cmds.append(f"export PYTHONPATH={sando_py_root}:${{PYTHONPATH:-}}")
        cmds.append(
            f"python3 -m sando_py.sim_io --ros-args "
            f"-r __ns:=/{ns} "
            f"-p config_file:={config_file} "
            f"-p agent_id:={i} "
            f"-p frame_id:=map"
        )
        panes.append({"shell_command": cmds})

    return yaml.safe_dump(data, sort_keys=False)


def _delay_goal_sender(panes, total_delay_s: int) -> None:
    """Rewrite any pane that runs ``goal_sender.launch.py`` so its
    leading ``sleep N`` is at least ``total_delay_s`` seconds. Without
    this the Python planner misses the goal because goal_sender is a
    one-shot publisher that exits after publishing."""
    for pane in panes:
        cmds = pane.get("shell_command") if isinstance(pane, dict) else None
        if not cmds:
            continue
        has_goal_sender = any(
            isinstance(c, str) and "goal_sender" in c for c in cmds
        )
        if not has_goal_sender:
            continue
        new_cmds = []
        seen_sleep = False
        for c in cmds:
            if isinstance(c, str) and c.strip().startswith("sleep "):
                # Replace the first sleep with our delayed value
                if not seen_sleep:
                    new_cmds.append(f"sleep {total_delay_s}")
                    seen_sleep = True
                    continue
            new_cmds.append(c)
        if not seen_sleep:
            new_cmds.insert(0, f"sleep {total_delay_s}")
        pane["shell_command"] = new_cmds


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _run_cpp_backend(rest: List[str]) -> None:
    """Exec the upstream launcher with ``rest`` verbatim."""
    ws = _find_sando_ws()
    upstream = ws / "src" / "sando" / "scripts" / "run_sim.py"
    if shutil.which("tmuxp") is None and "--dry-run" not in rest:
        print(
            "[run_sim wrapper] WARNING: `tmuxp` is not on PATH. "
            "The upstream launcher needs it to start the demo "
            "(install with `pip install tmuxp`). Continuing anyway; "
            "--dry-run will still work.",
            file=sys.stderr,
        )
    os.chdir(ws)
    print(
        f"[run_sim wrapper] forwarding to {upstream} (cwd={ws}, backend=cpp)",
        file=sys.stderr,
    )
    argv = [sys.executable, str(upstream), *rest]
    os.execvp(argv[0], argv)


def _run_py_backend(rest: List[str]) -> None:
    """Generate the upstream YAML, inject our Python sando_node pane,
    and either print it (``--dry-run``) or hand it to ``tmuxp load``."""
    ws = _find_sando_ws()
    upstream = ws / "src" / "sando" / "scripts" / "run_sim.py"
    sando_py_root = _find_sando_py_root()

    config_file = Path(
        os.environ.get(
            "SANDO_PY_CONFIG", str(sando_py_root / "config" / "sando.yaml"),
        )
    ).expanduser().resolve()
    if not config_file.is_file():
        # Fall back to upstream config if our copy is missing.
        config_file = (ws / "src" / "sando" / "config" / "sando.yaml").resolve()
    namespace = os.environ.get("SANDO_PY_NAMESPACE", _DEFAULT_NAMESPACE)
    warmup_s = os.environ.get("SANDO_PY_WARMUP_S", _DEFAULT_WARMUP_S)

    print(
        f"[run_sim wrapper] backend=py, ns=/{namespace}, "
        f"config={config_file}, warmup={warmup_s}s",
        file=sys.stderr,
    )

    # Capture the upstream's --dry-run output
    dry_run_argv = [sys.executable, str(upstream), *rest]
    user_dry_run = "--dry-run" in rest
    if not user_dry_run:
        dry_run_argv.append("--dry-run")

    try:
        res = subprocess.run(
            dry_run_argv,
            cwd=str(ws),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"[run_sim wrapper] upstream launcher failed (exit={e.returncode}):\n"
            f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}",
            file=sys.stderr,
        )
        sys.exit(e.returncode)

    yaml_text = extract_yaml_from_dry_run(res.stdout)
    modified = inject_python_backend(
        yaml_text,
        sando_py_root=sando_py_root,
        config_file=config_file,
        namespace=namespace,
        warmup_s=warmup_s,
    )

    if user_dry_run:
        print(
            "\n[run_sim wrapper] --dry-run + --backend py: modified YAML below.",
            file=sys.stderr,
        )
        print(_DRY_RUN_DELIM)
        print(modified)
        print(_DRY_RUN_DELIM)
        return

    if shutil.which("tmuxp") is None:
        print(
            "[run_sim wrapper] tmuxp is not on PATH; install with "
            "`pip install tmuxp`.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Write to a temp file and hand off to tmuxp
    fd, temp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="sando_py_")
    os.close(fd)
    Path(temp_yaml).write_text(modified)
    print(
        f"[run_sim wrapper] launching tmuxp with modified YAML: {temp_yaml}",
        file=sys.stderr,
    )
    # The YAML's shell_command_before contains ``. "$SETUP_BASH"`` — the
    # upstream sets this env var when it execs tmuxp itself, but here
    # we're exec'ing tmuxp directly so we must populate it from the
    # ``-s/--setup-bash`` flag found in ``rest``. Resolve relative paths
    # against the colcon workspace (matches the upstream's behaviour).
    setup_bash = _extract_setup_bash(rest, ws)
    if setup_bash is not None:
        os.environ["SETUP_BASH"] = str(setup_bash)
    os.execvp("tmuxp", ["tmuxp", "load", "-d", temp_yaml])


def _extract_setup_bash(rest: List[str], ws: Path):
    """Pull ``-s <path>`` / ``--setup-bash <path>`` out of ``rest`` and
    resolve it against ``ws`` if it's relative. Returns ``None`` when
    the flag is absent (the YAML's ``. "$SETUP_BASH"`` then fails loudly
    inside the pane, which is the right behaviour)."""
    for i, arg in enumerate(rest):
        if arg in ("-s", "--setup-bash") and i + 1 < len(rest):
            p = Path(rest[i + 1]).expanduser()
            if not p.is_absolute():
                p = (ws / p).resolve()
            return p
        if arg.startswith("--setup-bash="):
            p = Path(arg.split("=", 1)[1]).expanduser()
            if not p.is_absolute():
                p = (ws / p).resolve()
            return p
        if arg.startswith("-s="):
            p = Path(arg.split("=", 1)[1]).expanduser()
            if not p.is_absolute():
                p = (ws / p).resolve()
            return p
    return None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # Carve out --backend before forwarding the rest to upstream.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--backend",
        choices=["cpp", "py"],
        default="cpp",
        help="Which planner backend to run. cpp (default) forwards to "
             "the upstream launcher unchanged; py replaces the C++ "
             "sando_node with the Python sando_py.sim_io module.",
    )
    known, rest = parser.parse_known_args(sys.argv[1:])

    if known.backend == "cpp":
        _run_cpp_backend(rest)
    else:
        _run_py_backend(rest)


if __name__ == "__main__":
    main()
