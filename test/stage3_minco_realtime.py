"""Stage 3 — receding-horizon / REAL-TIME closed-loop test.

Offline 'valid' is NOT the same as closed-loop safe: a human moves BETWEEN
replans, so the executed path can breach even when every single plan was valid
at plan-time. This is the test behind the 'dynamic reactivity' metric.

Coverage:
  RT.warm  warm-start: a replan starts from the commanded v0 (state continuity,
           so receding-horizon does not restart from rest each tick).
  RT.react a human crossing the path: plan-ONCE open-loop BREACHES, while
           replanning each tick (reacting to where the human IS now) stays SAFE
           and reaches the goal. Stressed over several crossing scenarios.
  RT.lat   per-replan solve-time stats (informational + a loose regression ceiling).

Run:  python3 test/stage3_minco_realtime.py
"""
import os
import sys
import time

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([8.0, 0.0, 1.5])
WALL = AABBObstacle([3.0, 1.2, 0.0], [5.0, 3.0, 3.0], class_name="wall")
CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall", "soft", 0.4, 1.0e1)}
HR, D_SAFE = 0.4, 0.8
ND = DetourConfig(enabled=False)
OPT = OptParams(vmax=3.0, amax=3.0)
SAFE_TOL = 0.05            # closed-loop is "safe" if min clearance >= d_safe - tol


def _plan_from(dpos, dvel, hpos, hvel_known):
    astar = np.linspace(dpos, GOAL, 8)
    human = SphereObstacle(hpos, HR, vel=hvel_known, class_name="human")
    t0 = time.perf_counter()
    tr, info = plan_minco(astar, [human, WALL], CFG, opt_params=OPT,
                          detour_cfg=ND, v0=dvel)
    return tr, info, (time.perf_counter() - t0) * 1e3


def simulate(h0, hvel, *, replan, predict, dt=0.30, max_ticks=40):
    """Receding-horizon rollout. min_clr is the dense closed-loop clearance to
    the MOVING human along the executed (committed) path."""
    drone = START.copy(); dvel = np.zeros(3); hc = np.asarray(h0, float).copy()
    hknown = np.asarray(hvel, float) if predict else np.zeros(3)
    tr, info, ms = _plan_from(drone, dvel, hc, hknown)
    solve_ms = [ms]; min_clr = np.inf; reached = False; texec = 0.0
    for tick in range(max_ticks):
        if replan and tick > 0:
            tr, info, ms = _plan_from(drone, dvel, hc, hknown)
            solve_ms.append(ms); texec = 0.0          # fresh plan -> clock resets
        a0, a1 = texec, min(texec + dt, tr.t_end)
        for a in np.linspace(a0, a1, 12):             # dense sub-step clearance
            dp = tr.eval(float(a)); hp = hc + hvel * (float(a) - a0)
            min_clr = min(min_clr, float(np.linalg.norm(dp - hp) - HR))
        drone = tr.eval(a1); dvel = tr.eval_deriv(a1, 1)
        hc = hc + hvel * (a1 - a0); texec = a1
        if np.linalg.norm(drone[:2] - GOAL[:2]) < 0.3:
            reached = True; break
    return dict(min_clr=min_clr, reached=reached, ticks=tick + 1,
                ms=np.array(solve_ms))


# ============================================================ RT.warm  warm-start
print("\n--- RT.warm  a replan starts from the commanded v0 (state continuity) ---")
v_cmd = np.array([1.7, 0.4, 0.0])
human_far = SphereObstacle([4.0, -6.0, 1.5], HR, class_name="human")
tr_w, _ = plan_minco(np.linspace(START, GOAL, 8), [human_far, WALL], CFG,
                     opt_params=OPT, detour_cfg=ND, v0=v_cmd)
v_got = tr_w.eval_deriv(tr_w.t_start, 1)
check("RT.warm.0 replan honours v0 (eval_deriv(0,1) == commanded velocity)",
      np.allclose(v_got, v_cmd, atol=1e-6), f"v0 got={np.round(v_got, 4)} cmd={v_cmd}")
tr_r, _ = plan_minco(np.linspace(START, GOAL, 8), [human_far, WALL], CFG,
                     opt_params=OPT, detour_cfg=ND)            # default -> rest
check("RT.warm.1 default start is rest (v0 == 0)",
      np.allclose(tr_r.eval_deriv(tr_r.t_start, 1), 0.0, atol=1e-6),
      f"v0 got={np.round(tr_r.eval_deriv(tr_r.t_start, 1), 4)}")


# ============================================== RT.react  crossing human: react or breach
# headline: a human crossing upward, timed to reach the path as the drone passes.
print("\n--- RT.react  crossing human: plan-once BREACHES, replan stays SAFE ---")
h0 = np.array([4.0, -3.6, 1.5]); hvel = np.array([0.0, 1.5, 0.0])
open_once = simulate(h0, hvel, replan=False, predict=False)
closed = simulate(h0, hvel, replan=True, predict=False)

check("RT.react.0 non-vacuous: plan-ONCE open-loop BREACHES the moving human",
      open_once["min_clr"] < D_SAFE - SAFE_TOL,
      f"open-loop min_clr={open_once['min_clr']:.3f} < {D_SAFE - SAFE_TOL}")
check("RT.react.1 closed-loop replan stays SAFE (min clr >= d_safe - tol)",
      closed["min_clr"] >= D_SAFE - SAFE_TOL,
      f"closed-loop min_clr={closed['min_clr']:.3f}")
check("RT.react.2 closed-loop reaches the goal (warm-start continuity holds)",
      closed["reached"], f"reached={closed['reached']} ticks={closed['ticks']}")
check("RT.react.3 replanning strictly improves closed-loop clearance over plan-once",
      closed["min_clr"] > open_once["min_clr"] + 0.3,
      f"closed={closed['min_clr']:.3f} vs open={open_once['min_clr']:.3f}")

# stress: several distinct crossing geometries/speeds -> closed-loop SAFE in ALL
print("\n--- RT.react  stress: closed-loop safe across crossing scenarios ---")
scenarios = [
    ("up-fast",   np.array([4.5, -3.8, 1.5]), np.array([0.0, 1.7, 0.0])),
    ("diagonal",  np.array([3.0, -3.2, 1.5]), np.array([0.5, 1.4, 0.0])),
    ("slow-late", np.array([4.0, -2.6, 1.5]), np.array([0.0, 1.1, 0.0])),
]
all_ms = list(closed["ms"]) + list(open_once["ms"])
worst = np.inf
for name, s0, sv in scenarios:
    r = simulate(s0, sv, replan=True, predict=False)
    all_ms += list(r["ms"])
    worst = min(worst, r["min_clr"])
    check(f"RT.react.[{name}] closed-loop SAFE + reached",
          r["min_clr"] >= D_SAFE - SAFE_TOL and r["reached"],
          f"min_clr={r['min_clr']:.3f} reached={r['reached']} ticks={r['ticks']}")


# ============================================================ RT.lat  solve-time
print("\n--- RT.lat  per-replan solve-time (Python; C++ deploy ~10-50x faster) ---")
ms = np.array(all_ms)
med, p95, mx = np.median(ms), np.percentile(ms, 95), ms.max()
print(f"   solve-time over {ms.size} replans: median={med:.1f}ms  "
      f"p95={p95:.1f}ms  max={mx:.1f}ms")
check("RT.lat.0 median replan within a loose ceiling (gross-regression guard)",
      med < 1000.0, f"median={med:.1f}ms (informational; not an RT budget claim)")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
