"""Dynamic time-allocation adapter.

Wraps a window of candidate time-scale factors around a current mean.
On a successful solve, the window is recentered on the successful
factor. On a failed solve, the window shifts up (toward
``factor_final``); when it would exceed ``factor_final`` it resets to
the configured initial window.

This is a single-threaded version of the C++
``factors_`` adaptation in ``sando.cpp::tryToCompleteOpt`` (the C++
fires all candidate factors at multiple Gurobi workers in parallel and
picks the lowest successful one; we just iterate the window sequentially
inside :class:`GurobiSolver`).

The class is stateless across processes — long-running planning loops
should keep a single instance alive so the adapter actually adapts.

Public API
----------
    TimeAlloc(params)
    .factors_to_try()    -> Iterable[float]  (lowest-first)
    .on_success(factor)  -> None             (call after success)
    .on_failure()        -> None             (call after exhausting window)
    .current_mean        -> float            (for logging / debug)

When ``use_dynamic_factor`` is off, the adapter always yields a single
factor (``dynamic_factor_initial_mean``, or 1.0 if unset) — the same
behaviour you get by hard-coding the time allocation.
"""

from __future__ import annotations

from typing import Iterable, List

from sando_py.core import Parameters


_FACTOR_FLOOR = 1.0   # never below the v_max cruise allocation
_FALLBACK_INITIAL = 1.0


class TimeAlloc:
    def __init__(self, params: Parameters) -> None:
        self.params = params
        # Resolve config — fall back to safe defaults when keys are unset
        self._initial_mean = float(params.dynamic_factor_initial_mean or _FALLBACK_INITIAL)
        self._k_radius = max(0.0, float(params.dynamic_factor_k_radius or 0.0))
        self._step = max(1e-6, float(params.factor_constant_step_size or 0.1))
        self._lo = max(_FACTOR_FLOOR, float(params.factor_initial or _FALLBACK_INITIAL))
        self._hi = max(self._lo + self._step, float(params.factor_final or self._lo))
        self._use_dynamic = bool(params.use_dynamic_factor)

        self._current_mean = self._initial_mean
        self._window: List[float] = self._build_window(self._current_mean)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_mean(self) -> float:
        """Center of the current factor window. For logs / debug only."""
        return self._current_mean

    def factors_to_try(self) -> Iterable[float]:
        """Yield the next batch of factors to try, lowest first.

        When ``use_dynamic_factor`` is off, yields a single factor
        (``initial_mean``) so the caller doesn't need a branch.
        """
        if not self._use_dynamic:
            yield max(_FACTOR_FLOOR, self._initial_mean)
            return
        for f in self._window:
            yield f

    def on_success(self, factor: float) -> None:
        """Recenter the window on the successful factor (matches the C++
        ``successful_factor`` branch in ``sando.cpp``)."""
        if not self._use_dynamic:
            return
        factor = max(self._lo, min(self._hi, float(factor)))
        self._current_mean = factor
        self._window = self._build_window(factor)

    def on_failure(self) -> None:
        """Shift the window up by one step. Reset to the initial mean if
        we would step past ``factor_final``."""
        if not self._use_dynamic:
            return
        # Mean of the current window
        if not self._window:
            self._current_mean = self._initial_mean
        else:
            self._current_mean = sum(self._window) / len(self._window)

        if self._current_mean + self._step > self._hi:
            # Reset to initial window
            self._current_mean = self._initial_mean
            self._window = self._build_window(self._initial_mean)
        else:
            self._current_mean += self._step
            self._window = [min(self._hi, f + self._step) for f in self._window]
            if not self._window:
                self._window = self._build_window(self._current_mean)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_window(self, mean: float) -> List[float]:
        """Window of factors around ``mean``, clipped to [lo, hi], with
        the configured radius and step size.

        Returns at least one element so ``factors_to_try`` always yields
        something even when k_radius is zero.
        """
        if self._k_radius <= 0 or self._step <= 0:
            return [max(self._lo, min(self._hi, mean))]

        n = max(1, int((2.0 * self._k_radius) / self._step) + 1)
        out: List[float] = []
        for i in range(n):
            f = mean - self._k_radius + i * self._step
            if self._lo - 1e-9 <= f <= self._hi + 1e-9:
                out.append(max(self._lo, min(self._hi, f)))
        if not out:
            out.append(max(self._lo, min(self._hi, mean)))
        return out
