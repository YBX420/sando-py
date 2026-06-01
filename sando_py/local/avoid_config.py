"""Per-class obstacle-avoidance parameters.

Open-ended schema: a config is a dict[class_name -> AvoidParams]. Add new
classes by adding new entries — cost.py looks each obstacle's class_name up
in the config and applies its (mode, d_safe, weight) at evaluation time.

mode:
  "soft"  — gentle field, low weight (walls, ceilings)
  "hard"  — very stiff field via a huge weight (people, robots)
  Same EGO-Planner cubic penalty shape for both; the only difference is
  weight magnitude (so optimisation stays unconstrained / differentiable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class AvoidParams:
    name: str
    mode: str       # "soft" or "hard"
    d_safe: float   # metres — penalty kicks in when point-to-obstacle distance < d_safe
    weight: float   # cost weight


def default_config() -> Dict[str, AvoidParams]:
    """Two-class baseline. Tune via overrides; add classes by adding entries."""
    return {
        "wall":  AvoidParams("wall",  mode="soft", d_safe=0.4, weight=1.0e1),
        "human": AvoidParams("human", mode="hard", d_safe=0.8, weight=1.0e4),
    }


def resolve_mode(obs, avoid_cfg: Dict[str, AvoidParams], override) -> str:
    """Resolve an obstacle's avoidance mode -> 'hard' | 'soft'.

    override: None -> per-class (mode from avoid_cfg[class]);
              'soft'/'hard' -> GLOBAL ablation force on every obstacle.
    FAIL-SAFE: an unknown/missing class, or a malformed mode, defaults to
    'hard' — the violated constraint could be a person, so the safe default
    is the stiff continuous-time constraint, never a silent soft trade-off.
    """
    if override in ("soft", "hard"):
        return override
    params = avoid_cfg.get(obs.class_name)
    if params is None:
        return "hard"
    if params.mode not in ("soft", "hard"):
        return "hard"
    return params.mode
