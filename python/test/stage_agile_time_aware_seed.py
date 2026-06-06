"""Agile mode — arrival-time-aware topology seed (见缝就穿 / seize the transient gap).

The production topology seed (use_topology) picked the passing side from the obstacle's
pose AT t=0. For a MOVING human, the t=0 view is wrong by the time the drone arrives. The
arrival-time seed (time_aware_seed=True) estimates when the drone reaches each path vertex
(cum arclength / vmax) and picks the side that is open AT THAT TIME, from the obstacle's
PREDICTED pose — model-based, no learning. This is the per-class MINCO analogue of T-MPC's
(x,y,t) space-time homotopy selection (de Groot et al., T-RO 2024).

HONEST scope (verified while building this):
  * In OPEN single-crosser scenes the production space-time ALM already threads from a
    straight seed, so the seed does not change the end-to-end outcome there — it is proven
    at the SEED level (Part 1).
  * The seed's end-to-end payoff shows up in DENSE moving scenes where the optimizer would
    otherwise only graze d_safe; there the time-aware seed lifts the flown clearance from
    BELOW d_safe (grazing) to ABOVE d_safe (Part 2).

Coverage:
  Part 1 (open crosser, SEED-level mechanism):
    AGILE.1  geometric (t=0) topology is BLIND to a future-arriving crosser (no detour).
    AGILE.2  the straight (geometric) seed BREACHES in space-time (non-empty baseline).
    AGILE.3  time-aware seed DETOURS around the future conflict.
    AGILE.4  time-aware seed lifts space-time clearance from breach to positive (right class).
    AGILE.5  detour goes to the side the human is LEAVING (pass behind a crosser).
    AGILE.6  gating: a STATIC human -> time-aware ON == OFF (byte-identical seeds).
  Part 2 (dense moving slit, END-TO-END value):
    AGILE.7  end-to-end plan_minco: time-aware clears strictly better than geometric.
    AGILE.8  time-aware lifts the flown trajectory from grazing-below-d_safe to >= d_safe.

Run:  python test/stage_agile_time_aware_seed.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.local_opt import (plan_minco, OptParams, DetourConfig,        # noqa: E402
                                      _generate_detour_seeds,
                                      _min_clearance_per_obstacle)
from sando_py.local.obstacles import SphereObstacle                               # noqa: E402
from sando_py.local.avoid_config import AvoidParams                               # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


D_SAFE, HR, VMAX = 0.8, 0.4, 3.0
CFG = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4)}
OPT = OptParams(vmax=VMAX, amax=3.0)
# straight seed path: 9 vertices from (0,0,1.5) to (8,0,1.5)
PATH = np.array([[float(x), 0.0, 1.5] for x in range(9)], dtype=np.float64)


def topo_dict(obstacles, time_aware):
    cfg = DetourConfig(enabled=True, use_topology=True, time_aware_seed=time_aware)
    return dict(_generate_detour_seeds(PATH, obstacles, CFG, OPT, cfg))


def solve(obstacles, time_aware, budget=200.0):
    cfg = DetourConfig(enabled=True, use_topology=True, time_aware_seed=time_aware)
    _, info = plan_minco(PATH, obstacles, CFG, v0=np.zeros(3), a0=np.zeros(3),
                         opt_params=OPT, time_budget_ms=budget, detour_cfg=cfg)
    return info


# =========================== Part 1: open crosser (seed mechanism) ===========================
# Human FAR (south) at t=0 but crossing NORTH onto the path exactly when the drone arrives
# near x=4 (reaches x=4 at t~=4/vmax=1.33s; human y = -3 + 2.0*1.33 ~= -0.33).
HUMAN = SphereObstacle([4.0, -3.0, 1.5], HR, vel=[0.0, 2.0, 0.0], class_name="human")

def st_clr(seed_path):  # min space-time clearance of a polyline seed at arrival times
    return _min_clearance_per_obstacle(seed_path, [HUMAN], 200, speed=VMAX)[0][0]

off = topo_dict([HUMAN], False)
check("AGILE.1 geometric (t=0) topology blind to future crosser (no detour seed)",
      "topology" not in off, f"seeds={list(off)}")
clr_off = st_clr(off["original"])
check("AGILE.2 straight (geometric) seed breaches in space-time",
      clr_off < D_SAFE, f"min space-time clearance={clr_off:.3f} < d_safe={D_SAFE}")

on = topo_dict([HUMAN], True)
check("AGILE.3 time-aware seed detours around the future conflict",
      "topology" in on, f"seeds={list(on)}")
topo = on.get("topology", off["original"])
clr_on = st_clr(topo)
check("AGILE.4 time-aware seed lifts clearance from breach to positive (right homotopy class)",
      clr_on > 0.0 and (clr_on - clr_off) > 0.5,
      f"space-time clearance {clr_off:.3f} -> {clr_on:.3f}")
check("AGILE.5 detour goes to the side the +y-crosser is LEAVING (-y)",
      float(topo[:, 1].min()) < -0.5, f"min y of detour seed={float(topo[:, 1].min()):.3f}")

# gating: a STATIC human must be byte-identical ON vs OFF
STATIC = SphereObstacle([4.0, 0.2, 1.5], HR, vel=[0.0, 0.0, 0.0], class_name="human")
s_off = topo_dict([STATIC], False)
s_on = topo_dict([STATIC], True)
same = (set(s_off) == set(s_on)
        and all(np.array_equal(s_off[k], s_on[k]) for k in s_off))
check("AGILE.6 static obstacle: time-aware ON == OFF (byte-identical gating)", same,
      f"keys off={list(s_off)} on={list(s_on)}")

# =========================== Part 2: dense moving slit (end-to-end value) ===========================
# A row of crossing humans that DRIFT +y, so the slot the straight line aims at (y=0) is
# CLOSING by the time the drone arrives. This is the local-minimum regime where the seed
# changes the flown clearance.
SLIT = [SphereObstacle([4.0, y, 1.5], HR, vel=[0.0, 1.8, 0.0], class_name="human")
        for y in (-3.6, -1.2, 1.2, 3.6)]
i_off = solve(SLIT, False)
i_on = solve(SLIT, True)
print(f"   [dense slit] geometric : valid={i_off['trajectory_valid']} "
      f"min_clr={i_off['continuous_min_clearance']:.3f} seed={i_off['seed_used']}")
print(f"   [dense slit] time-aware: valid={i_on['trajectory_valid']} "
      f"min_clr={i_on['continuous_min_clearance']:.3f} seed={i_on['seed_used']}")
co, cn = i_off["continuous_min_clearance"], i_on["continuous_min_clearance"]
check("AGILE.7 dense slit end-to-end: time-aware clears strictly better than geometric",
      cn >= co + 0.05, f"flown min clearance geometric={co:.3f} -> time-aware={cn:.3f}")
check("AGILE.8 time-aware lifts flown trajectory from grazing-below-d_safe to >= d_safe",
      co < D_SAFE <= cn, f"geometric grazes {co:.3f} < {D_SAFE} <= time-aware {cn:.3f}")


# ---------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\n{n_pass}/{len(results)} passed")
sys.exit(0 if n_pass == len(results) else 1)
