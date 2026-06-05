"""Agile mode — CLOSED-LOOP threading of a DENSE MOVING crowd.

This is the real test of "see the gap, thread the gap": a receding-horizon rollout where
each tick replans from the current drone state (warm-started on the commanded v0), feeding
the crowd's predicted velocities, commits a slice, the humans walk on, and the executed
(committed) path's clearance to EVERY human is sub-step densely sampled. The agile knob
under test is the arrival-time topology seed (minco_time_aware_seed).

What it shows:
  CROWD.<s>.open   plan-ONCE open-loop BREACHES the moving crowd (non-vacuous baseline).
  CROWD.<s>.safe   closed-loop replanning stays SAFE (min clr over all humans >= d_safe-tol).
  CROWD.<s>.reach  closed-loop reaches the goal (threads start->goal through the crowd).

Run:  python test/stage_agile_dynamic_crowd.py
"""
import os
import sys
import time

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402
from sando_py.local.obstacles import SphereObstacle                        # noqa: E402
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([10.0, 0.0, 1.5])
HR, D_SAFE, SAFE_TOL = 0.3, 0.8, 0.05
CFG = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4)}
OPT = OptParams(vmax=3.0, amax=3.0)


def simulate_crowd(crowd, *, ta, replan=True, predict=True, dt=0.30,
                   max_ticks=80, budget=200.0):
    """Receding-horizon rollout through a moving crowd. Returns closed-loop stats."""
    drone = START.copy(); dvel = np.zeros(3)
    hpos = [np.asarray(p, float).copy() for p, _ in crowd]
    hvel = [np.asarray(v, float) for _, v in crowd]
    cfg = DetourConfig(enabled=True, use_topology=True, time_aware_seed=ta)

    def plan(dp, dv, positions):
        astar = np.linspace(dp, GOAL, 9)
        obs = [SphereObstacle(positions[i], HR,
                              vel=(hvel[i] if predict else np.zeros(3)),
                              class_name="human") for i in range(len(crowd))]
        try:
            tr, info = plan_minco(astar, obs, CFG, opt_params=OPT, detour_cfg=cfg,
                                  v0=dv, time_budget_ms=budget)
            return tr, info
        except Exception:
            return None, None

    tr, _ = plan(drone, dvel, hpos)
    min_clr = np.inf; reached = False; ms = []; texec = 0.0; fails = 0; tick = 0
    for tick in range(max_ticks):
        if replan and tick > 0:
            t0 = time.perf_counter()
            ntr, _ = plan(drone, dvel, hpos)
            ms.append((time.perf_counter() - t0) * 1e3)
            if ntr is not None:
                tr = ntr; texec = 0.0
            else:
                fails += 1                      # hold last plan (freeze) — recovery not built yet
        if tr is None:
            break
        a0, a1 = texec, min(texec + dt, tr.t_end)
        for a in np.linspace(a0, a1, 12):       # dense sub-step clearance to ALL humans
            dp = tr.eval(float(a))
            for i in range(len(crowd)):
                hp = hpos[i] + hvel[i] * (float(a) - a0)
                min_clr = min(min_clr, float(np.linalg.norm(dp - hp) - HR))
        drone = tr.eval(a1); dvel = tr.eval_deriv(a1, 1)
        for i in range(len(crowd)):
            hpos[i] = hpos[i] + hvel[i] * (a1 - a0)
        texec = a1
        if np.linalg.norm(drone[:2] - GOAL[:2]) < 0.3:
            reached = True; break
    return dict(min_clr=min_clr, reached=reached, ticks=tick + 1, fails=fails,
                ms=np.array(ms) if ms else np.array([0.0]))


# crowd = list of (pos0, vel)
SCENARIOS = {
    # a slalom of staggered crossers: a gap opens/closes as the drone advances
    "slalom": [([3.0, -2.6, 1.5], [0.0, 1.3, 0.0]),
               ([5.0,  2.6, 1.5], [0.0, -1.3, 0.0]),
               ([7.0, -2.6, 1.5], [0.0, 1.3, 0.0])],
    # a wall of 4 humans sweeping up: the y=0 slot the straight line aims at is closing
    "sweep":  [([5.0, y, 1.5], [0.0, 1.2, 0.0]) for y in (-3.6, -1.2, 1.2, 3.6)],
    # denser: 5 crossers at staggered x and alternating directions
    "dense":  [([3.0, -3.0, 1.5], [0.0, 1.4, 0.0]),
               ([4.5,  3.0, 1.5], [0.0, -1.4, 0.0]),
               ([6.0, -3.0, 1.5], [0.0, 1.4, 0.0]),
               ([7.5,  3.0, 1.5], [0.0, -1.4, 0.0]),
               ([5.2, -4.0, 1.5], [0.0, 1.1, 0.0])],
}

all_ms = []
for name, crowd in SCENARIOS.items():
    print(f"\n--- crowd: {name} ({len(crowd)} moving humans) ---")
    op = simulate_crowd(crowd, ta=True, replan=False)              # plan-once open-loop
    geo = simulate_crowd(crowd, ta=False, replan=True)             # closed-loop, geometric seed
    taw = simulate_crowd(crowd, ta=True, replan=True)              # closed-loop, time-aware seed
    all_ms += list(geo["ms"]) + list(taw["ms"])
    print(f"   open-once : min_clr={op['min_clr']:.3f} reached={op['reached']}")
    print(f"   closed geo: min_clr={geo['min_clr']:.3f} reached={geo['reached']} "
          f"ticks={geo['ticks']} fails={geo['fails']}")
    print(f"   closed t-a: min_clr={taw['min_clr']:.3f} reached={taw['reached']} "
          f"ticks={taw['ticks']} fails={taw['fails']}")
    check(f"CROWD.{name}.open non-vacuous: plan-once open-loop breaches",
          op["min_clr"] < D_SAFE - SAFE_TOL, f"open min_clr={op['min_clr']:.3f}")
    check(f"CROWD.{name}.safe closed-loop time-aware stays SAFE (>= d_safe - tol)",
          taw["min_clr"] >= D_SAFE - SAFE_TOL, f"closed t-a min_clr={taw['min_clr']:.3f}")
    check(f"CROWD.{name}.reach closed-loop time-aware threads to the goal",
          taw["reached"], f"reached={taw['reached']} ticks={taw['ticks']} fails={taw['fails']}")

ms = np.array(all_ms)
print(f"\n   solve-time over {ms.size} replans: median={np.median(ms):.1f}ms "
      f"p95={np.percentile(ms, 95):.1f}ms max={ms.max():.1f}ms")

total = len(results); passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
sys.exit(0 if passed == total else 1)
