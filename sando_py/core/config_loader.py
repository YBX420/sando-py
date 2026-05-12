"""Load SANDO YAML config into a Parameters dataclass.

The YAML follows ROS 2 parameter conventions:

    <node_name>:
      ros__parameters:
        v_max: 10.0
        ...

We strip those two wrapping levels and assign each leaf key to the matching
``Parameters`` field. Type coercion is done according to the dataclass
annotation so YAML floats land in float fields, ints in int fields, etc.

Unknown keys (in YAML but not in Parameters) are *kept aside* and returned
in ``extras`` so callers can spot drift between sando.yaml and the struct.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, get_type_hints

import yaml

from .types import Parameters


# YAML keys that map onto Parameters fields under a different name.
# The C++ ROS-param namespace uses some keys that don't exactly match
# the struct field names; alias them here so the YAML stays compatible
# with the upstream C++ launcher without forcing renames in either
# direction.
_ALIASES: Dict[str, str] = {
    # `sando_map_res` in YAML -> `res` in Parameters (matches the C++
    # ROS-param name; the struct field is short because everything
    # else in the planner already lives in a "map" namespace).
    "sando_map_res": "res",
}


def _coerce(field_name: str, field_type, value: Any) -> Any:
    """Best-effort coercion of YAML scalar to dataclass field type."""
    if value is None:
        return value
    # List types
    origin = getattr(field_type, "__origin__", None)
    if origin in (list, List):
        if not isinstance(value, list):
            raise TypeError(
                f"{field_name}: expected list, got {type(value).__name__}"
            )
        elem_type = field_type.__args__[0] if getattr(field_type, "__args__", None) else None
        if elem_type is float:
            return [float(x) for x in value]
        if elem_type is int:
            return [int(x) for x in value]
        return list(value)

    if field_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    if field_type is int:
        return int(value)

    if field_type is float:
        return float(value)

    if field_type is str:
        return str(value)

    # Fallback: return as-is
    return value


def load_parameters_from_yaml(path: str | Path,
                              node_name: str | None = None,
                              strict: bool = False) -> Tuple[Parameters, Dict[str, Any]]:
    """Parse a SANDO YAML config into a Parameters instance.

    Args:
        path: path to YAML file (e.g. ``config/sando.yaml``).
        node_name: top-level key under which ``ros__parameters`` lives. If None,
            the first top-level key is used.
        strict: if True, raise on unknown YAML keys instead of returning them.

    Returns:
        (params, extras) — extras maps YAML keys with no matching field to their values.
    """
    path = Path(path)
    with open(path, "r") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{path}: empty or non-dict YAML")

    # Strip the <node_name> / ros__parameters wrappers if present
    if node_name is None:
        node_name = next(iter(doc.keys()))
    node = doc[node_name]
    flat = node.get("ros__parameters", node) if isinstance(node, dict) else node
    if not isinstance(flat, dict):
        raise ValueError(f"{path}: could not locate flat parameter dict under '{node_name}'")

    hints = get_type_hints(Parameters)
    known = set(hints.keys())
    extras: Dict[str, Any] = {}
    p = Parameters()

    for key, value in flat.items():
        target = _ALIASES.get(key, key)
        if target in known:
            try:
                setattr(p, target, _coerce(target, hints[target], value))
            except Exception as e:
                raise ValueError(f"{path}: failed to coerce {key}={value!r}: {e}") from e
        else:
            extras[key] = value

    if strict and extras:
        raise KeyError(
            f"{path}: unknown YAML keys with no matching Parameters field: "
            f"{sorted(extras.keys())}"
        )

    return p, extras


def warn_if_extras(extras: Dict[str, Any]) -> None:
    """Print a warning about unmapped keys (drift between YAML and the dataclass)."""
    if not extras:
        return
    print(
        f"[config_loader] {len(extras)} YAML keys not in Parameters: "
        f"{sorted(extras.keys())}",
        file=sys.stderr,
    )
