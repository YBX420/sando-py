"""Stage 5 — planner MINCO local solve (synthetic-input integration, no ROS).

Exercises SANDO.plan_local_trajectory with the new MINCO path (default
local_solver='minco'): builds Parameters + Planner, sets A/G states, feeds a
straight global polyline + a moving human, and asserts the committed plan:
  - returns True (honest info['trajectory_valid'] gate),
  - stores a QuinticPieceWisePol eval-equivalent to the MinjerkTraj (~1e-9),
  - the published pwp == the executed goal_setpoints,
  - clears the human by d_safe (dense surface check),
  - used NO Gurobi (spy on the only gurobipy-touching method),
  - dispatches BOTH ways (minco vs gurobi routing).
Plus a standalone adapter micro-test on a random MinjerkTraj.

Run:  python3 test/stage5_planner_minco.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)

from sando_py.types import Parameters, RobotState                      # noqa: E402
from sando_py.planner import (                                          # noqa: E402
    SANDO, QuinticPieceWisePol, _minjerk_to_pwp)
from sando_py.local.minco import MinjerkTraj                            # noqa: E402
from sando_py.local.obstacles import SphereObstacle                    # noqa: E402
from sando_py.solver_gurobi import SolverGurobi                        # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ----------------------------------------------------------------------------
# Adapter micro-test: random MinjerkTraj -> QuinticPieceWisePol, eval-equiv.
# ----------------------------------------------------------------------------
rng = np.random.default_rng(7)
wp = np.array([[0, 0, 1.5], [1.2, 0.4, 1.7], [2.5, -0.3, 1.4], [4.0, 0.2, 1.6]],
              dtype=np.float64)
durs = np.array([0.8, 1.1, 0.9])
mj_rand = MinjerkTraj(wp, durs,
                      v0=rng.normal(size=3) * 0.3, a0=rng.normal(size=3) * 0.2)
A_TIME_RAND = 3.14
pwp_rand = _minjerk_to_pwp(mj_rand, A_TIME_RAND)

ts = np.linspace(pwp_rand.times[0], pwp_rand.get_end_time(), 601)
err_p = err_v = err_a = 0.0
for t in ts:
    tau = t - A_TIME_RAND
    err_p = max(err_p, float(np.max(np.abs(pwp_rand.eval(t) - mj_rand.eval_deriv(tau, 0)))))
    err_v = max(err_v, float(np.max(np.abs(pwp_rand.velocity(t) - mj_rand.eval_deriv(tau, 1)))))
    err_a = max(err_a, float(np.max(np.abs(pwp_rand.acceleration(t) - mj_rand.eval_deriv(tau, 2)))))
check("ADAPT.1 adapter pos eval-equivalent <1e-9", err_p < 1e-9, f"max|dp|={err_p:.2e}")
check("ADAPT.2 adapter vel eval-equivalent <1e-9", err_v < 1e-9, f"max|dv|={err_v:.2e}")
check("ADAPT.3 adapter acc eval-equivalent <1e-9", err_a < 1e-9, f"max|da|={err_a:.2e}")
check("ADAPT.4 is QuinticPieceWisePol (6-coeff/seg)",
      isinstance(pwp_rand, QuinticPieceWisePol) and pwp_rand.coeff_x[0].shape == (6,),
      f"shape={pwp_rand.coeff_x[0].shape}")
check("ADAPT.5 times length == M+1 dynamically",
      len(pwp_rand.times) == mj_rand.M + 1, f"M={mj_rand.M} times={len(pwp_rand.times)}")


# ----------------------------------------------------------------------------
# Integration: planner with default MINCO path.
# ----------------------------------------------------------------------------
par = Parameters()
par.local_solver = "minco"
par.dc = 0.1
par.vehicle_type = "uav"   # avoid the ground z-clamp branch
planner = SANDO(par)

# No-Gurobi spy: wrap the ONLY gurobipy-touching method.
_gurobi_calls = {"n": 0}
_orig_gen = SolverGurobi.generate_new_trajectory
def _spy(self, factor):
    _gurobi_calls["n"] += 1
    return _orig_gen(self, factor)
SolverGurobi.generate_new_trajectory = _spy

global_path = [np.array(p, dtype=np.float64) for p in
               [[0, 0, 1.5], [1.5, 0, 1.5], [3, 0, 1.5], [4.5, 0, 1.5], [6, 0, 1.5]]]

A = RobotState(pos=global_path[0].copy(), vel=np.array([0.5, 0.0, 0.0]),
               accel=np.zeros(3))
planner.set_A(A)
planner.set_A_time(10.0)
G = RobotState(pos=global_path[-1].copy())
planner.set_G(G)   # default _drone_status=GOAL_REACHED -> local_E = local_G.clone()

# Human ON the path. radius = 0.5*max(bbox) = 0.3 < d_safe=0.8 so the straight
# path WOULD violate -> the clearance assertion is meaningful. The placeholder
# hook treats the snapshot as static (vel=0, the production snapshot carries no
# velocity today). We exercise the real hook here (no override).
HUMAN_CENTRE = np.array([3.0, 0.0, 1.5])
HUMAN_BBOX = np.array([0.6, 0.6, 0.6])  # radius = 0.5*max = 0.3
planner.obst_pos = [HUMAN_CENTRE.copy()]
planner.obst_bbox = [HUMAN_BBOX.copy()]

# Pre-check: raw straight-path min clearance to the human < d_safe.
raw_min = min(float(np.linalg.norm(p - HUMAN_CENTRE) - 0.3) for p in global_path)
check("PRE.0 raw path violates d_safe (assertion is meaningful)",
      raw_min < 0.8, f"raw_min_clearance={raw_min:.3f}")

ok = planner.plan_local_trajectory(global_path, 0.0)
check("INT.1 plan_local_trajectory returns True", ok is True, f"ok={ok}")

pwp = planner.get_pwp()
check("INT.2 committed pwp has >=2 boundary times", len(pwp.times) >= 2,
      f"n_times={len(pwp.times)}")
check("INT.3 pwp end > start", pwp.get_end_time() > pwp.times[0],
      f"[{pwp.times[0]:.3f},{pwp.get_end_time():.3f}]")
check("INT.4 pwp.times[0] == A_time", abs(pwp.times[0] - 10.0) < 1e-12,
      f"times[0]={pwp.times[0]}")
check("INT.5 committed pwp is QuinticPieceWisePol",
      isinstance(pwp, QuinticPieceWisePol) and pwp.coeff_x[0].shape == (6,))

# EVAL-EQUIVALENCE pwp vs the stashed MinjerkTraj (dense).
mj = planner._last_minco_traj
A_time = 10.0
ts = np.linspace(pwp.times[0], pwp.get_end_time(), 700)
ep = ev = ea = 0.0
for t in ts:
    tau = t - A_time
    ep = max(ep, float(np.max(np.abs(pwp.eval(t) - mj.eval_deriv(tau, 0)))))
    ev = max(ev, float(np.max(np.abs(pwp.velocity(t) - mj.eval_deriv(tau, 1)))))
    ea = max(ea, float(np.max(np.abs(pwp.acceleration(t) - mj.eval_deriv(tau, 2)))))
ADAPTER_EVAL_ERR = max(ep, ev, ea)
check("INT.6 pwp pos eval-equiv to MINCO <1e-9", ep < 1e-9, f"max|dp|={ep:.2e}")
check("INT.7 pwp vel eval-equiv to MINCO <1e-9", ev < 1e-9, f"max|dv|={ev:.2e}")
check("INT.8 pwp acc eval-equiv to MINCO <1e-9", ea < 1e-9, f"max|da|={ea:.2e}")

# goal_setpoints contract.
gsp = planner.retrieve_goal_setpoints()
check("INT.9 goal_setpoints len>=2", len(gsp) >= 2, f"n={len(gsp)}")
t_strictly_inc = all(gsp[i + 1].t > gsp[i].t for i in range(len(gsp) - 1))
check("INT.10 goal_setpoints .t strictly increasing", t_strictly_inc)
# expected times: A_time + (i+1)*dc clamped to t_end-1e-6
t_end = mj.t_end
texp_ok = True
for i, s in enumerate(gsp):
    texp = A_time + min((i + 1) * par.dc, t_end - 1e-6)
    texp_ok &= abs(s.t - texp) < 1e-9
check("INT.11 goal_setpoints times == A_time+(i+1)*dc clamped", texp_ok)
finite = all(np.all(np.isfinite(s.pos)) and np.all(np.isfinite(s.vel))
             and np.all(np.isfinite(s.accel)) and np.all(np.isfinite(s.jerk))
             for s in gsp)
check("INT.12 all setpoint pos/vel/accel/jerk finite", finite)
# executed path == published pwp
exec_match = max(float(np.max(np.abs(s.pos - pwp.eval(s.t)))) for s in gsp)
check("INT.13 goal_setpoints.pos == pwp.eval (executed==published)",
      exec_match < 1e-9, f"max|dp|={exec_match:.2e}")
# warm-start: first sample vel close to A.vel (continuous from start)
warm = float(np.linalg.norm(gsp[0].vel - A.vel))
check("INT.14 warm-start: first setpoint vel ~ A.vel",
      np.linalg.norm(mj.eval_deriv(0.0, 1) - A.vel) < 1e-9 and warm < 0.5,
      f"|v0_mj-Av|={np.linalg.norm(mj.eval_deriv(0.0,1)-A.vel):.2e} |gsp0-Av|={warm:.3f}")

# HUMAN CLEARANCE (dense, surface-based) against the SAME obstacle the hook
# builds (radius=0.5*max(bbox)=0.3, vel=0 placeholder).
human = SphereObstacle(centre0=HUMAN_CENTRE.copy(), radius=0.3,
                       vel=np.zeros(3), class_name="human")
d_safe = 0.8
ts = np.linspace(pwp.times[0], pwp.get_end_time(), 1500)
min_clear = np.inf
for t in ts:
    tau = t - A_time
    p = pwp.eval(t)
    c = human.predict(tau)
    min_clear = min(min_clear, float(np.linalg.norm(p - c) - human.radius))
check("INT.15 human cleared by d_safe (dense surface)",
      min_clear >= d_safe - 1e-3, f"min_surface_clearance={min_clear:.3f} (d_safe={d_safe})")

# NO GUROBI + metadata sentinels.
check("INT.16 NO Gurobi call on minco path", _gurobi_calls["n"] == 0,
      f"gurobi_calls={_gurobi_calls['n']}")
check("INT.17 successful_factor sentinel == 1.0", planner.successful_factor == 1.0)
check("INT.18 local_traj_computation_time > 0", planner.local_traj_computation_time > 0.0)
check("INT.19 cvx_decomp_time == 0.0", planner.cvx_decomp_time == 0.0)
check("INT.20 poly_out_safe empty on minco path", planner.poly_out_safe == [])
check("INT.21 cps shape (M,6,3)",
      hasattr(planner.cps, "shape") and planner.cps.shape == (mj.M, 6, 3),
      f"cps_shape={getattr(planner.cps, 'shape', None)}")


# ----------------------------------------------------------------------------
# DISPATCH BOTH WAYS: gurobi flag falls through to the Gurobi body, NOT minco.
# ----------------------------------------------------------------------------
par_g = Parameters()
par_g.local_solver = "gurobi"
planner_g = SANDO(par_g)
_routed = {"minco": False}
def _flag_setter(global_path, t):
    _routed["minco"] = True
    return True
planner_g.plan_local_trajectory_minco = _flag_setter
# Call into the gurobi body; it may fail (no license / no map) but it MUST NOT
# route to the minco method. We only assert routing, not a successful solve.
try:
    planner_g.plan_local_trajectory(
        [np.array(p, dtype=np.float64) for p in [[0, 0, 1], [1, 0, 1], [2, 0, 1]]], 0.0)
except Exception:
    pass
check("INT.22 gurobi flag does NOT route to minco", _routed["minco"] is False)

# restore patched method
SolverGurobi.generate_new_trajectory = _orig_gen


# ----------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\nADAPTER eval-equivalence error (max over pos/vel/acc): {ADAPTER_EVAL_ERR:.3e}")
print(f"=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    sys.exit(1)
