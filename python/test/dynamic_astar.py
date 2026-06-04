"""Moving obstacle + A* detour timing.

IMPORTANT: the heat-A* has NO time axis. read_map bakes the obstacle's predicted
trajectory into a STATIC heat tube, time-weighted by exp(-t/tau) (sooner = hotter).
So A* "timing" = avoid the predicted swept corridor, weighted by how soon the
obstacle gets there. True per-time avoidance ("be at X exactly when the obstacle
isn't") is local_opt's job, not the global guide's.

This test verifies the two things the global guide DOES do:
  1. with prediction, A* detours around where the obstacle WILL sweep
     (not just where it is now)
  2. time weighting: an obstacle crossing SOON is avoided more than one
     crossing LATE (same geometry, different crossing time)

Run:  python3 test/dynamic_astar.py
"""
import os
import sys

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.hgp.graph_search import GraphSearch                       # noqa: E402
from sando_py.hgp.voxel_map import VoxelMapUtil                          # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

RES = 0.2
H = 2.0
START = np.array([-3.0, 0.0, 1.0])
GOAL = np.array([3.0, 0.0, 1.0])
CROSS = np.array([0.0, 0.0, 1.0])   # where the drone's straight path crosses x=0

def build_map(samples, times, traj_max_time):
    """A mostly-unknown map whose only feature is the moving obstacle's heat tube.
    The obstacle sweeps in y at x=0, across the drone's straight path."""
    m = VoxelMapUtil(res=RES)
    m.use_soft_cost_obstacles = False
    m.dynamic_as_occupied_current = False   # obstacle represented purely by heat
    m.static_heat_enabled = False           # isolate the dynamic tube
    m.dyn_pred_samples = samples
    m.dyn_pred_times = times
    m.read_map(46, 44, 10, np.array([0.0, 0.0, 1.0]), np.zeros((0, 3)),
               0.0, 3.0, 0.2, [np.array([0.0, -1.2, 1.0])], [np.array([0.2, 0.2, 0.2])],
               traj_max_time)
    return m

def run_astar(m, heat_w):
    gs = GraphSearch(m, eps=1.0, global_planner="astar_heat" if heat_w > 0 else "astar",
                     w_unknown=0.0, heat_weight=heat_w)
    si = m.float_to_int(START); gi = m.float_to_int(GOAL)
    ok = gs.plan(si, gi, 0.0, np.zeros(3), max_expand=400000, timeout_ms=6000)
    return ok, gs.get_path_world()

def detour(path):
    """Max perpendicular deviation from the straight START->GOAL line (y=0, z=1).
    Captures a detour in EITHER y or z (A* may choose to go over in z)."""
    return max(float(np.hypot(p[1], p[2] - 1.0)) for p in path)

def min_dist_cross(path):
    return min(float(np.linalg.norm(np.asarray(p) - CROSS)) for p in path)

times = np.linspace(0.0, H, 10)

def sweep(t_cross_frac):
    """Obstacle sweeps y from -1.2 to +1.2, reaching y=0 at t_cross_frac*H."""
    out = []
    for tt in times:
        f = tt / H
        # piecewise-linear in time so y=0 is hit at t_cross_frac
        if f <= t_cross_frac:
            y = -1.2 + 1.2 * (f / max(t_cross_frac, 1e-6))
        else:
            y = 1.2 * ((f - t_cross_frac) / max(1 - t_cross_frac, 1e-6))
        out.append(np.array([0.0, y, 1.0]))
    return out

print("=" * 60)
print("Moving obstacle + A* detour timing")
print("=" * 60)

# ---- 1. prediction vs current-only ----
m_nopred = build_map(None, None, 0.0)                       # no future tube
m_pred = build_map([sweep(0.5)], times, H)                   # full predicted sweep
_, p_nopred = run_astar(m_nopred, 30.0)
_, p_pred = run_astar(m_pred, 30.0)
d_nopred, d_pred = detour(p_nopred), detour(p_pred)
print(f"\ndetour (max deviation from straight line):  no-prediction={d_nopred:.2f}  "
      f"with-prediction={d_pred:.2f}")
print(f"min dist to crossing: no-prediction={min_dist_cross(p_nopred):.2f}  "
      f"with-prediction={min_dist_cross(p_pred):.2f}")
check("1: prediction makes A* detour around the future sweep (bigger deviation)",
      d_pred > d_nopred + 0.3, f"{d_pred:.2f} > {d_nopred:.2f}")
check("1: with prediction A* keeps further from the crossing point",
      min_dist_cross(p_pred) > min_dist_cross(p_nopred) + 0.2)

# ---- 2. timing: soon-crossing avoided more than late-crossing ----
m_soon = build_map([sweep(0.2)], times, H)   # obstacle at y=0 early (hot)
m_late = build_map([sweep(0.85)], times, H)  # obstacle at y=0 late (decayed)
h_soon = m_soon.get_heat(*[int(v) for v in m_soon.float_to_int(CROSS)])
h_late = m_late.get_heat(*[int(v) for v in m_late.float_to_int(CROSS)])
# heat_weight=2 is in the graded regime: a strong-enough heat fully detours, a
# weak one barely does — so the timing effect shows up in the path, not just the field.
# (At high heat_weight both saturate to the same detour.)
_, p_soon = run_astar(m_soon, 2.0)
_, p_late = run_astar(m_late, 2.0)
d_soon, d_late = detour(p_soon), detour(p_late)
print(f"\nheat at crossing:  soon={h_soon:.3f}  late={h_late:.3f}")
print(f"detour (heat_weight=2):  soon={d_soon:.2f}  late={d_late:.2f}")
check("2: soon-crossing is hotter at the crossing than late-crossing",
      h_soon > h_late + 1e-3, f"soon={h_soon:.3f} late={h_late:.3f}")
check("2: soon-crossing forces a LARGER detour than late (timing affects routing)",
      d_soon > d_late + 0.3, f"soon={d_soon:.2f} late={d_late:.2f}")

# ---- summary ----
n_pass = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"SUMMARY: {n_pass} passed, {len(results) - n_pass} failed (of {len(results)})")
print("NOTE: global A* avoids a time-WEIGHTED static tube; true space-time")
print("      ('dodge it at the exact moment') is local_opt's job, not the guide's.")
print("=" * 60)
