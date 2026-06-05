"""Agile mode — VERY congested CORRIDOR (baseline failure probe, pre-recovery).

A narrow lane bounded by HARD walls (so the drone cannot escape sideways — the corridor
analogue of removing the z-flyover cheat) packed with a dense moving crowd. We drive the
CURRENT agile planner closed-loop and HONESTLY record how it copes / dies, to establish the
baseline that motivates the safety half (monitor + recovery). When a replan finds no feasible
trajectory, today's behaviour is to HOLD the last plan (freeze) — exactly the gap to fix.

This is a DIAGNOSTIC probe, not a pass/fail gate: it prints reached / min-human-clearance
(breach?) / min-wall-clearance (wall hit?) / freeze count / how far it got.

Run:  python test/stage_agile_corridor_trap.py
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

START = np.array([0.0, 0.0, 1.5])
GOAL = np.array([11.0, 0.0, 1.5])
HR, D_HUMAN, D_WALL = 0.3, 0.8, 0.15
# walls HARD here, purely to CONFINE the lane (remove the sideways-escape cheat) and force
# the hold-or-thread decision; the per-class design keeps walls soft in normal scenes.
CFG = {"human": AvoidParams("human", "hard", D_HUMAN, 1.0e4),
       "wall":  AvoidParams("wall", "hard", D_WALL, 1.0e4)}
OPT = OptParams(vmax=3.0, amax=3.0)

# narrow lane y in [-1.3, 1.3] (width 2.6), walls x in [2,9], full height
LANE = 1.3
WALLS = [AABBObstacle([2.0, -4.0, 0.0], [9.0, -LANE, 3.0], class_name="wall"),
         AABBObstacle([2.0,  LANE, 0.0], [9.0,  4.0, 3.0], class_name="wall")]

# dense oncoming + offset crowd marching toward the drone inside the tube
CROWD = [([4.0,  0.0, 1.5], [-1.0, 0.0, 0.0]),
         ([5.0,  0.7, 1.5], [-1.0, 0.0, 0.0]),
         ([6.0, -0.7, 1.5], [-1.0, 0.0, 0.0]),
         ([7.0,  0.4, 1.5], [-1.0, 0.0, 0.0]),
         ([8.0, -0.4, 1.5], [-1.0, 0.0, 0.0])]


def run(ta, replan=True, dt=0.30, max_ticks=120, budget=200.0):
    """Production-like: only a trajectory_valid plan is committed; otherwise HOLD the last
    valid plan, and if none exists, HOVER in place (this is exactly today's freeze behaviour).
    Humans keep moving regardless, so a freeze in a congested tube = getting walked into."""
    drone = START.copy(); dvel = np.zeros(3)
    hpos = [np.asarray(p, float).copy() for p, _ in CROWD]
    hvel = [np.asarray(v, float) for _, v in CROWD]
    cfg = DetourConfig(enabled=True, use_topology=True, time_aware_seed=ta)

    def plan(dp, dv, positions):
        astar = np.linspace(dp, GOAL, 9)
        obs = list(WALLS) + [SphereObstacle(positions[i], HR, vel=hvel[i], class_name="human")
                             for i in range(len(CROWD))]
        try:
            tr, info = plan_minco(astar, obs, CFG, opt_params=OPT, detour_cfg=cfg,
                                  v0=dv, time_budget_ms=budget)
            return tr, info
        except Exception:
            return None, None

    committed = None
    tr, info = plan(drone, dvel, hpos)
    if info is not None and info.get("trajectory_valid"):
        committed = tr
    min_h = np.inf; min_w = np.inf; reached = False; fails = 0; froze_hit = False
    texec = 0.0; tick = 0
    for tick in range(max_ticks):
        if replan and tick > 0:
            ntr, ninfo = plan(drone, dvel, hpos)
            if ninfo is not None and ninfo.get("trajectory_valid"):
                committed = ntr; texec = 0.0
            else:
                fails += 1                          # no valid plan -> HOLD/HOVER (today's freeze)
        # advance one tick of real time
        if committed is not None and texec < committed.t_end - 1e-9:
            a0, a1 = texec, min(texec + dt, committed.t_end)
            samples = [committed.eval(float(a)) for a in np.linspace(a0, a1, 12)]
            dt_real = a1 - a0
            nxt = committed.eval(a1); nvel = committed.eval_deriv(a1, 1)
        else:                                        # HOVER: no plan / plan exhausted
            a0, dt_real = 0.0, dt
            samples = [drone.copy() for _ in range(12)]
            nxt = drone.copy(); nvel = np.zeros(3)
        # clearance over the slice (humans move through the slice)
        for k, dp in enumerate(samples):
            tau = dt_real * k / 11.0
            for i in range(len(CROWD)):
                hp = hpos[i] + hvel[i] * tau
                d = float(np.linalg.norm(dp - hp) - HR)
                if d < min_h:
                    min_h = d
                    if committed is None or texec >= committed.t_end - 1e-9:
                        froze_hit = froze_hit or d < D_HUMAN
            for w in WALLS:
                min_w = min(min_w, float(w.signed_dist(dp, 0.0)))
        drone = nxt; dvel = nvel
        for i in range(len(CROWD)):
            hpos[i] = hpos[i] + hvel[i] * dt_real
        texec = (texec + dt_real) if committed is not None else texec
        if np.linalg.norm(drone[:2] - GOAL[:2]) < 0.3:
            reached = True; break
    return dict(min_h=min_h, min_w=min_w, reached=reached, fails=fails, froze_hit=froze_hit,
                ticks=tick + 1, final_x=float(drone[0]), final_y=float(drone[1]))


print("=== VERY congested corridor - current agile, closed-loop baseline ===")
print(f"lane width={2*LANE:.1f}m, {len(CROWD)} humans (d_safe={D_HUMAN}), hard walls\n")
for label, ta, replan in (("open-once (plan once)", True, False),
                          ("closed geometric seed", False, True),
                          ("closed time-aware seed", True, True)):
    r = run(ta, replan=replan)
    hit_h = r["min_h"] < D_HUMAN - 0.05
    pen_h = r["min_h"] < 0.0
    pen_w = r["min_w"] < -0.05
    verdict = ("REACHED" if r["reached"] else f"STUCK@x={r['final_x']:.1f}")
    flags = []
    if pen_h: flags.append("HUMAN-COLLISION")
    elif hit_h: flags.append("human-breach")
    if pen_w: flags.append("WALL-PENETRATION")
    if r["fails"] > 0: flags.append(f"froze x{r['fails']}")
    if r["froze_hit"]: flags.append("HIT-WHILE-FROZEN")
    print(f"  {label:24s}: {verdict:14s} min_human={r['min_h']:+.3f} "
          f"min_wall={r['min_w']:+.3f} ticks={r['ticks']} y_end={r['final_y']:+.2f} "
          f"{'  '.join(flags) if flags else 'clean'}")

print("\n(diagnostic baseline - establishes what the safety half must fix; no pass/fail)")
