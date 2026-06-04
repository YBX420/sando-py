"""Golden-data generator for types.py C++ port verification (types.hpp).

不许欺骗、必须还原: dumps the REAL Python types.py outputs over diverse + random +
edge cases. The C++ port (types.hpp) reads the same inputs and must match within tol.

Sections dumped (each a self-contained CASE/END block, key->numbers lines):
  - PWP   : PieceWisePol eval / velocity / acceleration over sample times (incl. edge:
            t before start, t at/after end, exactly on a knot).
  - EXPR  : _AnalyticExpr eval of many expressions (incl. the required
            "3.5*sin(0.6*t)", "10.0 + 0.5*t", "2.0", trefoil-style, ^ operator,
            nested funcs, unary minus, ** right-assoc) at sample t.
  - DYNA  : DynTraj mode "Analytic" eval / velocity / accel (with and without
            explicit velocity expr strings -> numeric central-difference fallback).
  - DYNP  : DynTraj mode "Piecewise" eval / velocity / accel.

Run: python cpp/golden/gen_types_golden.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.types import PieceWisePol, DynTraj, _AnalyticExpr  # noqa: E402

rng = np.random.default_rng(2026)


def rf(x):
    return repr(float(x))


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []


# ---------------------------------------------------------------------------
# helper to build a random PieceWisePol with `nseg` cubic segments
# ---------------------------------------------------------------------------
def make_pwp(nseg, t0):
    times = [t0]
    for _ in range(nseg):
        times.append(times[-1] + float(rng.uniform(0.2, 1.5)))
    cx = [rng.normal(0, 2.0, 4) for _ in range(nseg)]
    cy = [rng.normal(0, 2.0, 4) for _ in range(nseg)]
    cz = [rng.normal(0, 2.0, 4) for _ in range(nseg)]
    p = PieceWisePol()
    p.times = times
    p.coeff_x = cx
    p.coeff_y = cy
    p.coeff_z = cz
    return p


# ===========================================================================
# PWP cases (PieceWisePol)
# ===========================================================================
pwp_specs = [
    (1, 0.0),   # single segment
    (2, 0.0),
    (3, 0.5),   # nonzero start time
    (4, -1.0),  # negative start time
    (6, 0.0),
]
for ci, (nseg, t0) in enumerate(pwp_specs):
    p = make_pwp(nseg, t0)
    tstart = p.times[0]
    tend = p.times[-1]
    # sample times: include before-start, after-end, exact knots, and dense interior
    interior = list(np.linspace(tstart, tend, 23))
    extra = [tstart - 0.7, tend + 0.9, tend, tstart] + list(p.times)
    ts = sorted(set([round(float(x), 12) for x in interior + extra]))
    e0 = np.array([p.eval(t) for t in ts])
    e1 = np.array([p.velocity(t) for t in ts])
    e2 = np.array([p.acceleration(t) for t in ts])

    lines.append("CASE")
    lines.append("KIND PWP")
    lines.append(f"NSEG {nseg}")
    lines.append(f"TIMES {fmt(p.times)}")
    lines.append(f"CX {fmt(np.array(p.coeff_x))}")
    lines.append(f"CY {fmt(np.array(p.coeff_y))}")
    lines.append(f"CZ {fmt(np.array(p.coeff_z))}")
    lines.append(f"GET_END_TIME {rf(p.get_end_time())}")
    lines.append(f"GET_DURATION {rf(p.get_duration())}")
    lines.append(f"NT {len(ts)}")
    lines.append(f"TS {fmt(ts)}")
    lines.append(f"E0 {fmt(e0)}")
    lines.append(f"E1 {fmt(e1)}")
    lines.append(f"E2 {fmt(e2)}")
    lines.append("END")


# ===========================================================================
# EXPR cases (_AnalyticExpr) — diverse expressions, evaluated at many t.
# ===========================================================================
exprs = [
    "2.0",
    "10.0 + 0.5*t",
    "3.5*sin(0.6*t)",
    "cos(t)",
    "sin(t)*cos(t)",
    "-2.0*t + 3.0",
    "sqrt(abs(t) + 1.0)",
    "exp(-0.1*t)",
    "log(t*t + 1.0)",
    "pow(t, 3)",
    "t^2 + 2*t + 1",            # ExprTk-style caret
    "2**t",                     # ** operator
    "2**-1 + t",                # ** with unary-minus RHS (right assoc)
    "-2**2 + t",                # unary minus looser than ** -> -4 + t
    "(t + 1.0)*(t - 1.0)",
    "tan(0.3*t)",
    "atan2(t, 2.0)",
    "pi*t",
    "e*1.0 + 0.0*t",
    "2.0 + 3.0*sin(t) - 1.5*cos(2.0*t) + 0.5*sin(3.0*t)",   # trefoil-ish
    "3.0*sin(t + 2.0*0.7) + 2.0*sin(2.0*t)",                 # trefoil x-ish
    "1.0e1 + 2.5e-1*t",         # scientific notation literals
    "sin(0.6*t)*sqrt(2.0)",
    "",                          # empty -> returns 0.0
]
ts_expr = [-3.3, -1.0, -0.25, 0.0, 0.1, 0.7, 1.0, 1.5,
           2.3456, 3.14159, 4.0, 5.5, 6.28318, 9.87]
for idx, src in enumerate(exprs):
    ex = _AnalyticExpr(src)
    vals = [ex(t) for t in ts_expr]
    lines.append("CASE")
    lines.append("KIND EXPR")
    lines.append(f"SRC {src}")          # raw string (may be empty -> nothing after key)
    lines.append(f"NT {len(ts_expr)}")
    lines.append(f"TS {fmt(ts_expr)}")
    lines.append(f"VALS {fmt(vals)}")
    lines.append("END")


# ===========================================================================
# DYNA cases (DynTraj mode "Analytic")
# ===========================================================================
# (1) with explicit velocity strings (velocity uses traj_v*), (2) without (numeric fallback)
dyn_analytic_specs = [
    # x, y, z, vx, vy, vz
    ("3.0*sin(t)", "3.0*cos(t)", "0.5*t", "3.0*cos(t)", "-3.0*sin(t)", "0.5"),
    ("3.5*sin(0.6*t)", "2.0*cos(0.4*t)", "1.0 + 0.2*t", "", "", ""),   # numeric fallback
    # trefoil knot style
    ("sin(t) + 2.0*sin(2.0*t)", "cos(t) - 2.0*cos(2.0*t)", "-sin(3.0*t)", "", "", ""),
    ("10.0 + 0.5*t", "2.0", "3.0*sin(0.6*t)", "", "", ""),
    ("t^2", "2*t", "5.0", "2*t", "2.0", "0.0"),
]
ts_dyn = [-2.0, -0.5, 0.0, 0.3, 1.0, 1.7, 2.5, 3.3, 4.4, 6.28]
for si, (sx, sy, sz, svx, svy, svz) in enumerate(dyn_analytic_specs):
    d = DynTraj()
    d.mode = "Analytic"
    d.traj_x, d.traj_y, d.traj_z = sx, sy, sz
    d.traj_vx, d.traj_vy, d.traj_vz = svx, svy, svz
    d.compile_analytic()
    ev = np.array([d.eval(t) for t in ts_dyn])
    vv = np.array([d.velocity(t) for t in ts_dyn])
    av = np.array([d.accel(t) for t in ts_dyn])
    lines.append("CASE")
    lines.append("KIND DYNA")
    lines.append(f"SX {sx}")
    lines.append(f"SY {sy}")
    lines.append(f"SZ {sz}")
    lines.append(f"SVX {svx}")
    lines.append(f"SVY {svy}")
    lines.append(f"SVZ {svz}")
    lines.append(f"NT {len(ts_dyn)}")
    lines.append(f"TS {fmt(ts_dyn)}")
    lines.append(f"EV {fmt(ev)}")
    lines.append(f"VV {fmt(vv)}")
    lines.append(f"AV {fmt(av)}")
    lines.append("END")


# ===========================================================================
# DYNP cases (DynTraj mode "Piecewise")
# ===========================================================================
for ci, (nseg, t0) in enumerate([(2, 0.0), (4, 0.3), (1, -0.5)]):
    p = make_pwp(nseg, t0)
    d = DynTraj()
    d.set_piecewise(p)
    tstart = p.times[0]
    tend = p.times[-1]
    ts = sorted(set(
        [round(float(x), 12) for x in np.linspace(tstart, tend, 17)]
        + [tstart - 0.4, tend + 0.6, tend, tstart]
        + [float(x) for x in p.times]
    ))
    ev = np.array([d.eval(t) for t in ts])
    vv = np.array([d.velocity(t) for t in ts])
    av = np.array([d.accel(t) for t in ts])
    lines.append("CASE")
    lines.append("KIND DYNP")
    lines.append(f"NSEG {nseg}")
    lines.append(f"TIMES {fmt(p.times)}")
    lines.append(f"CX {fmt(np.array(p.coeff_x))}")
    lines.append(f"CY {fmt(np.array(p.coeff_y))}")
    lines.append(f"CZ {fmt(np.array(p.coeff_z))}")
    lines.append(f"NT {len(ts)}")
    lines.append(f"TS {fmt(ts)}")
    lines.append(f"EV {fmt(ev)}")
    lines.append(f"VV {fmt(vv)}")
    lines.append(f"AV {fmt(av)}")
    lines.append("END")


out = os.path.join(os.path.dirname(__file__), "types_cases.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len([l for l in lines if l == 'CASE'])} cases)")
