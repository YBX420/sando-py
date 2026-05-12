"""Symbolic variable elimination for the cubic-Bezier QP, mirroring the
C++ ``computeDependentCoefficientsN4/N5/N6`` schemes in
``sando_ws/src/sando/src/sando/gurobi_solver.cpp``.

Goal
----
A cubic piecewise polynomial trajectory has ``4 * N`` coefficients per
axis (a, b, c, d for each of the ``N`` segments). Boundary conditions
(initial + final position/velocity/accel) plus ``C^2`` continuity at the
``N - 1`` junctions kill ``3 * (N - 1) + 6 = 3*N + 3`` degrees of
freedom. The remaining ``4*N - (3*N + 3) = N - 3`` coefficients per axis
are *free*. C++ takes those to be ``d_3, d_4, ..., d_{N-1}`` — the
constant term of segments 3 through N-1. Every other coefficient is then
a linear combination of those free vars plus boundary constants.

This module derives those linear combinations symbolically (with sympy)
once at solver init, lambdifies them to numpy callables, and provides a
unified ``CoefficientFormulas`` object that the Gurobi model uses to
emit linear expressions in the free Gurobi variables.

For N = 5 (the default ``num_N`` in sando.yaml), there are 2 free
coefficients per axis (d3, d4) — total 6 free Gurobi decision variables
for the 3D problem, down from 60 in the naive formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Symbolic derivation
# ---------------------------------------------------------------------------

@dataclass
class CoefficientFormulas:
    """Per-axis dependent-coefficient evaluator.

    For an N-segment cubic trajectory with free vars ``d_3, ..., d_{N-1}``
    and boundary state ``(P0, V0, A0, Pf, Vf, Af)``, this object lets the
    caller compute the linear map

        coeff[i, k] = sum_j M_jk[i,k] * d_{3+j} + bias[i,k]

    where ``coeff[i, k]`` is the k-th coefficient (k=0 → a_i cubic, k=1 →
    b_i quadratic, k=2 → c_i linear, k=3 → d_i constant) of segment i.

    ``M_free`` shape: ``(N, 4, num_free)``
    ``M_boundary`` shape: ``(N, 4, 6)`` — coefficients of [P0, V0, A0, Pf, Vf, Af]

    The maps depend only on the segment durations ``dt`` (sized ``(N,)``)
    so they're rebuilt each ``solve()`` call once the time allocation is
    fixed.
    """
    N: int
    num_free: int
    M_free: np.ndarray
    M_boundary: np.ndarray

    @classmethod
    def build(cls, N: int, dt: np.ndarray) -> "CoefficientFormulas":
        if N < 4:
            raise ValueError(
                "Variable elimination requires N >= 4 (got {})".format(N)
            )
        num_free = N - 3
        # Cache by (N, rounded dt) so multiple identical solves don't
        # re-derive the system.
        key = (N, tuple(np.round(dt, 10).tolist()))
        M_free, M_bnd = _build_cached(key)
        return cls(N=N, num_free=num_free, M_free=M_free, M_boundary=M_bnd)


def _build_cached(key: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """LRU-cached version of the heavy sympy derivation. The key is
    (N, rounded-dt-tuple) so repeat calls with the same time allocation
    skip the work."""
    N, dt_tuple = key
    dt = np.asarray(dt_tuple, dtype=float)
    return _derive(N, dt)


_BUILD_CACHE: dict = {}


def _derive(N: int, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the boundary + continuity linear system symbolically and
    return numeric (M_free, M_boundary) matrices."""
    cache_key = (N, tuple(np.round(dt, 10).tolist()))
    if cache_key in _BUILD_CACHE:
        return _BUILD_CACHE[cache_key]

    # Unknowns: x = [a_0, b_0, c_0, d_0, a_1, ..., d_{N-1}], length 4N.
    coeff_syms = sp.symbols(
        " ".join(f"x{i}" for i in range(4 * N)), real=True,
    )
    coeff_syms = list(coeff_syms)

    # Parameters: P0, V0, A0, Pf, Vf, Af (6) + free d_3..d_{N-1} (N - 3)
    P0, V0, A0, Pf, Vf, Af = sp.symbols("P0 V0 A0 Pf Vf Af", real=True)
    boundary_syms = [P0, V0, A0, Pf, Vf, Af]
    if N - 3 == 0:
        free_syms: List[sp.Symbol] = []
    elif N - 3 == 1:
        free_syms = [sp.Symbol("df0", real=True)]
    else:
        free_syms = list(sp.symbols(
            " ".join(f"df{j}" for j in range(N - 3)), real=True,
        ))

    dt_syms = [sp.Float(float(t)) for t in dt]

    def a(i): return coeff_syms[4 * i + 0]
    def b(i): return coeff_syms[4 * i + 1]
    def c(i): return coeff_syms[4 * i + 2]
    def d(i): return coeff_syms[4 * i + 3]

    eqs = []
    # --- Initial boundary at u=0 of segment 0 ---
    # p(0) = d_0, v(0) = c_0, accel(0) = 2*b_0
    eqs.append(sp.Eq(d(0), P0))
    eqs.append(sp.Eq(c(0), V0))
    eqs.append(sp.Eq(2 * b(0), A0))

    # --- C^2 continuity at junctions ---
    # at end of segment i (u = dt_i):
    #   p_i(dt) = a_i*dt^3 + b_i*dt^2 + c_i*dt + d_i = d_{i+1}
    #   v_i(dt) = 3 a_i dt^2 + 2 b_i dt + c_i       = c_{i+1}
    #   acc_i(dt) = 6 a_i dt + 2 b_i                = 2 b_{i+1}
    for i in range(N - 1):
        T = dt_syms[i]
        eqs.append(sp.Eq(
            a(i) * T ** 3 + b(i) * T ** 2 + c(i) * T + d(i), d(i + 1),
        ))
        eqs.append(sp.Eq(
            3 * a(i) * T ** 2 + 2 * b(i) * T + c(i), c(i + 1),
        ))
        eqs.append(sp.Eq(6 * a(i) * T + 2 * b(i), 2 * b(i + 1)))

    # --- Final boundary at end of segment N-1 ---
    T_last = dt_syms[N - 1]
    eqs.append(sp.Eq(
        a(N - 1) * T_last ** 3 + b(N - 1) * T_last ** 2
        + c(N - 1) * T_last + d(N - 1),
        Pf,
    ))
    eqs.append(sp.Eq(
        3 * a(N - 1) * T_last ** 2 + 2 * b(N - 1) * T_last + c(N - 1),
        Vf,
    ))
    eqs.append(sp.Eq(6 * a(N - 1) * T_last + 2 * b(N - 1), Af))

    # --- Pin free variables: d_3, d_4, ..., d_{N-1} ---
    for j in range(N - 3):
        eqs.append(sp.Eq(d(3 + j), free_syms[j]))

    # Solve the linear system. The unknowns are the 4N coefficients;
    # parameters are boundary_syms + free_syms. sympy.linsolve is the
    # fastest path for a square linear system over the rationals.
    sol = sp.solve(eqs, coeff_syms, dict=True)
    if not sol:
        raise RuntimeError(
            f"Variable elimination: failed to solve N={N} system"
        )
    sol = sol[0]

    # Build the (4N x num_free) and (4N x 6) coefficient matrices.
    num_free = N - 3
    M_free = np.zeros((N, 4, num_free), dtype=float)
    M_bnd = np.zeros((N, 4, 6), dtype=float)

    all_params = free_syms + boundary_syms  # df0..df_{N-4}, P0, V0, A0, Pf, Vf, Af
    for i in range(N):
        for k in range(4):
            expr = sp.expand(sol[coeff_syms[4 * i + k]])
            # Each expr is a linear combination of free_syms + boundary_syms.
            for j, sym in enumerate(free_syms):
                coeff_val = expr.coeff(sym)
                M_free[i, k, j] = float(coeff_val)
            for j, sym in enumerate(boundary_syms):
                coeff_val = expr.coeff(sym)
                M_bnd[i, k, j] = float(coeff_val)

    out = (M_free, M_bnd)
    _BUILD_CACHE[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Linear-expression evaluator for the Gurobi model
# ---------------------------------------------------------------------------

class LinearCoeff:
    """Linear function of free variables + boundary state, packaged so the
    Gurobi model can multiply / add / negate without constructing
    Gurobi LinExpr until the final ``to_linexpr`` call. Keeps the
    boundary contribution as a scalar that is *known* at solve time
    (boundary state is fixed at the start of each replan tick) so the
    constant part collapses to a number.
    """
    __slots__ = ("coef", "const")

    def __init__(self, coef: np.ndarray, const: float = 0.0):
        # coef shape: (num_free,)
        self.coef = np.asarray(coef, dtype=float).reshape(-1)
        self.const = float(const)

    def __add__(self, other):
        if isinstance(other, LinearCoeff):
            if self.coef.shape != other.coef.shape:
                # broadcast if one is empty
                if self.coef.size == 0:
                    return LinearCoeff(other.coef.copy(), self.const + other.const)
                if other.coef.size == 0:
                    return LinearCoeff(self.coef.copy(), self.const + other.const)
                raise ValueError("LinearCoeff size mismatch")
            return LinearCoeff(self.coef + other.coef, self.const + other.const)
        # scalar
        return LinearCoeff(self.coef.copy(), self.const + float(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, LinearCoeff):
            return LinearCoeff(self.coef - other.coef, self.const - other.const)
        return LinearCoeff(self.coef.copy(), self.const - float(other))

    def __rsub__(self, other):
        return LinearCoeff(-self.coef, float(other) - self.const)

    def __neg__(self):
        return LinearCoeff(-self.coef, -self.const)

    def __mul__(self, scalar: float):
        s = float(scalar)
        return LinearCoeff(self.coef * s, self.const * s)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        return self.__mul__(1.0 / float(scalar))

    def to_linexpr(self, free_vars):
        """Build a Gurobi LinExpr from this linear combination."""
        import gurobipy as gp  # type: ignore
        expr = gp.LinExpr(self.const)
        for j, c in enumerate(self.coef):
            if c != 0.0:
                expr.add(free_vars[j], c)
        return expr

    @classmethod
    def from_free(cls, j: int, num_free: int) -> "LinearCoeff":
        coef = np.zeros(num_free, dtype=float)
        coef[j] = 1.0
        return cls(coef, 0.0)

    @classmethod
    def constant(cls, value: float, num_free: int) -> "LinearCoeff":
        return cls(np.zeros(num_free, dtype=float), float(value))


# ---------------------------------------------------------------------------
# Top-level helper used by GurobiSolver
# ---------------------------------------------------------------------------

@dataclass
class SegmentExpressions:
    """Linear expressions (in the free vars + boundary scalars) for the
    4 polynomial coefficients of each of N segments. ``coeffs[i][k]`` is
    a ``LinearCoeff`` for the k-th coefficient (a=0, b=1, c=2, d=3) of
    segment i. ``num_free`` is the dimensionality of the free-variable
    block."""

    N: int
    num_free: int
    coeffs: List[List[LinearCoeff]] = field(default_factory=list)

    @classmethod
    def build(cls, N: int, dt: np.ndarray, boundary: np.ndarray) -> "SegmentExpressions":
        """``boundary`` is a length-6 numpy array ``[P0, V0, A0, Pf, Vf, Af]``
        (for a single axis). The free variables remain symbolic; the
        boundary state collapses into the constant term."""
        formulas = CoefficientFormulas.build(N, dt)
        num_free = formulas.num_free
        out: List[List[LinearCoeff]] = []
        for i in range(N):
            row: List[LinearCoeff] = []
            for k in range(4):
                # coefficient[i, k] = M_free[i, k] · free_vars + M_boundary[i, k] · boundary
                free_part = formulas.M_free[i, k].copy()
                const_part = float(np.dot(formulas.M_boundary[i, k], boundary))
                row.append(LinearCoeff(free_part, const_part))
            out.append(row)
        return cls(N=N, num_free=num_free, coeffs=out)

    # Convenience accessors ---------------------------------------------------

    def a(self, i: int) -> LinearCoeff: return self.coeffs[i][0]
    def b(self, i: int) -> LinearCoeff: return self.coeffs[i][1]
    def c(self, i: int) -> LinearCoeff: return self.coeffs[i][2]
    def d(self, i: int) -> LinearCoeff: return self.coeffs[i][3]

    def pos(self, i: int, u: float) -> LinearCoeff:
        """Position at parameter ``u`` in segment ``i``: ``a u^3 + b u^2 + c u + d``."""
        return (self.a(i) * (u ** 3)
                + self.b(i) * (u ** 2)
                + self.c(i) * u
                + self.d(i))

    def vel(self, i: int, u: float) -> LinearCoeff:
        """Velocity: ``3 a u^2 + 2 b u + c``."""
        return self.a(i) * (3 * u * u) + self.b(i) * (2 * u) + self.c(i)

    def accel(self, i: int, u: float) -> LinearCoeff:
        """Acceleration: ``6 a u + 2 b``."""
        return self.a(i) * (6 * u) + self.b(i) * 2

    def jerk(self, i: int) -> LinearCoeff:
        """Jerk (constant per segment): ``6 a``."""
        return self.a(i) * 6
