"""Tests for sando_py.core.config_loader."""

import textwrap

import pytest

from sando_py.core.config_loader import load_parameters_from_yaml
from sando_py.core.types import Parameters


REPO_YAML = "config/sando.yaml"


def test_load_real_yaml():
    p, extras = load_parameters_from_yaml(REPO_YAML)
    assert isinstance(p, Parameters)
    # spot-check a value from each tier — v_max is user-tunable in the yaml,
    # just sanity check it's a positive float.
    assert isinstance(p.v_max, float) and p.v_max > 0.0
    assert p.num_N == 5
    assert p.num_P == 3
    assert p.drone_bbox == [0.2, 0.2, 0.2]
    assert p.sfc_size == [3.0, 3.0, 3.0]
    # default boolean field
    assert isinstance(p.use_dynamic_factor, bool)
    # extras are obstacle_tracker / mapping keys, not planner core
    assert isinstance(extras, dict)


def test_load_minimal_inline(tmp_path):
    yaml_text = textwrap.dedent(
        """
        sando_node:
            ros__parameters:
                v_max: 7.5
                num_N: 6
                num_P: 2
                drone_bbox: [0.3, 0.3, 0.1]
                debug_verbose: true
        """
    )
    p_yaml = tmp_path / "x.yaml"
    p_yaml.write_text(yaml_text)
    p, extras = load_parameters_from_yaml(p_yaml)
    assert p.v_max == 7.5
    assert p.num_N == 6
    assert p.num_P == 2
    assert p.drone_bbox == [0.3, 0.3, 0.1]
    assert p.debug_verbose is True
    assert extras == {}


def test_strict_raises_on_unknown(tmp_path):
    yaml_text = textwrap.dedent(
        """
        sando_node:
            ros__parameters:
                v_max: 1.0
                some_unknown_key: 42
        """
    )
    p_yaml = tmp_path / "x.yaml"
    p_yaml.write_text(yaml_text)
    with pytest.raises(KeyError):
        load_parameters_from_yaml(p_yaml, strict=True)


def test_type_coercion(tmp_path):
    """int/float/bool/string coercion follows the dataclass annotation."""
    yaml_text = textwrap.dedent(
        """
        x:
            ros__parameters:
                v_max: 5            # int in yaml, should become float
                num_N: 7.0          # float in yaml, should become int
                use_hardware: "true"  # string -> bool
        """
    )
    p_yaml = tmp_path / "x.yaml"
    p_yaml.write_text(yaml_text)
    p, _ = load_parameters_from_yaml(p_yaml)
    assert isinstance(p.v_max, float) and p.v_max == 5.0
    assert isinstance(p.num_N, int) and p.num_N == 7
    assert p.use_hardware is True
