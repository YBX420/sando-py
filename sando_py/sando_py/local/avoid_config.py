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
