"""Tests for sando_py.solver.time_alloc — TimeAlloc adapter."""

from __future__ import annotations

import pytest

from sando_py.core import Parameters
from sando_py.solver.time_alloc import TimeAlloc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _params(**overrides) -> Parameters:
    p = Parameters()
    # Defaults that match config/sando.yaml
    p.use_dynamic_factor = True
    p.dynamic_factor_initial_mean = 1.5
    p.factor_initial = 1.0
    p.factor_final = 2.5
    p.factor_constant_step_size = 0.1
    p.dynamic_factor_k_radius = 0.4
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------------------
# Static behaviour (use_dynamic_factor off)
# ---------------------------------------------------------------------------

def test_static_sweeps_factor_range_when_disabled():
    p = _params(use_dynamic_factor=False, dynamic_factor_initial_mean=1.5)
    ta = TimeAlloc(p)
    factors = list(ta.factors_to_try())
    assert factors[0] == pytest.approx(p.factor_initial)
    assert factors[-1] == pytest.approx(p.factor_final)
    assert len(factors) == 16


def test_static_yields_at_least_one_factor():
    """Always yields ≥1 factor — no caller branch needed."""
    p = _params(use_dynamic_factor=False, dynamic_factor_initial_mean=0.0)
    ta = TimeAlloc(p)
    assert len(list(ta.factors_to_try())) >= 1


# ---------------------------------------------------------------------------
# Dynamic window construction
# ---------------------------------------------------------------------------

def test_dynamic_window_centered_on_initial_mean():
    p = _params()
    ta = TimeAlloc(p)
    factors = list(ta.factors_to_try())
    # Window of half-width 0.4 around 1.5, stepping by 0.1, clipped to [1.0, 2.5]
    # -> { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 }
    assert factors[0] == pytest.approx(1.1, abs=1e-6)
    assert factors[-1] == pytest.approx(1.9, abs=1e-6)
    assert all(factors[i] < factors[i + 1] for i in range(len(factors) - 1))


def test_dynamic_window_clipped_to_lo_hi():
    """If the initial mean sits near factor_final, the window must clip."""
    p = _params(dynamic_factor_initial_mean=2.5, factor_final=2.5)
    ta = TimeAlloc(p)
    factors = list(ta.factors_to_try())
    assert all(1.0 <= f <= 2.5 for f in factors)


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def test_on_success_recenters_window():
    p = _params()
    ta = TimeAlloc(p)
    ta.on_success(2.0)
    assert ta.current_mean == pytest.approx(2.0)
    factors = list(ta.factors_to_try())
    # Window now centred on 2.0, clipped to [1.0, 2.5]
    assert factors[0] >= 1.6 - 1e-6
    assert factors[-1] <= 2.4 + 1e-6


def test_on_failure_shifts_window_up():
    p = _params()
    ta = TimeAlloc(p)
    mean_before = ta.current_mean
    ta.on_failure()
    assert ta.current_mean > mean_before


def test_on_failure_resets_when_past_factor_final():
    """Repeated failures should eventually reset the window."""
    p = _params()
    ta = TimeAlloc(p)
    # Drive the window up well past factor_final
    for _ in range(50):
        ta.on_failure()
    # Should have wrapped back at least once
    assert ta.current_mean <= p.factor_final


def test_on_success_clamps_to_window_bounds():
    """Calling on_success with a wild factor must not break the window."""
    p = _params()
    ta = TimeAlloc(p)
    ta.on_success(100.0)
    assert ta.current_mean <= p.factor_final
    ta.on_success(-100.0)
    assert ta.current_mean >= max(1.0, p.factor_initial)


# ---------------------------------------------------------------------------
# Off-mode no-ops
# ---------------------------------------------------------------------------

def test_disabled_adapter_does_not_mutate_on_success_or_failure():
    p = _params(use_dynamic_factor=False)
    ta = TimeAlloc(p)
    factors_before = list(ta.factors_to_try())
    ta.on_success(2.0)
    ta.on_failure()
    factors_after = list(ta.factors_to_try())
    assert factors_after == factors_before
