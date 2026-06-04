"""Golden generator for avoid_config.py + cost.py C++ port."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.local.minco import MinjerkTraj
from sando_py.local.obstacles import SphereObstacle, AABBObstacle
from sando_py.local.avoid_config import default_config, resolve_mode
from sando_py.local.cost import penalty, obstacle_cost

rng = np.random.default_rng(2024)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


# --- a concrete trajectory ---
M = 3
wp = np.array([[0, 0, 2], [3, 1, 2], [6, -1, 2], [9, 0, 2]], dtype=float)
T = np.array([1.0, 1.1, 0.9])
mj = MinjerkTraj(wp, T)
mid = mj.eval(mj.t_end * 0.5)   # 摆在轨迹中点附近,确保软罚非零

# obstacles on the path: a moving human (hard) + a static wall box (soft)
sph_c0 = mid + np.array([0.0, 0.1, 0.0]); sph_r = 0.5
sph_v = np.array([0.2, -0.1, 0.0]); sph_a = np.array([0.0, 0.05, 0.0])
q1 = mj.eval(mj.t_end * 0.25)
aabb_lo = q1 - np.array([0.3, 0.3, 0.5]); aabb_hi = q1 + np.array([0.3, 0.3, 0.5])

obstacles = [SphereObstacle(centre0=sph_c0, radius=sph_r, vel=sph_v, accel=sph_a, class_name="human"),
             AABBObstacle(lo=aabb_lo, hi=aabb_hi, class_name="wall")]
cfg = default_config()
K = 60
cost = obstacle_cost(mj, obstacles, cfg, K=K)

lines = []
lines += ["CASE", f"M {M}", f"T {fmt(T)}", f"WP {fmt(wp)}",
          f"V0 {fmt(np.zeros(3))}", f"A0 {fmt(np.zeros(3))}",
          f"VF {fmt(np.zeros(3))}", f"AF {fmt(np.zeros(3))}",
          f"SPH {fmt(sph_c0)} {repr(float(sph_r))} {fmt(sph_v)} {fmt(sph_a)}",
          f"AABB {fmt(aabb_lo)} {fmt(aabb_hi)}",
          f"K {K}", f"COST {repr(float(cost))}", "END"]

# penalty samples
ps = []
for d, ds in [(0.1, 0.8), (0.9, 0.8), (0.0, 0.5), (0.5, 0.5), (-0.2, 0.4)]:
    ps += [d, ds, penalty(d, ds)]
lines += [f"PENALTY {fmt(ps)}"]

# resolve_mode samples: class, override -> mode (encode override "" as NONE)
rm = []
for cls, ov in [("human", ""), ("wall", ""), ("unknown", ""), ("wall", "hard"), ("human", "soft")]:
    rm.append(f"{cls}|{ov if ov else 'NONE'}|{resolve_mode(type('O',(),{'class_name':cls})(), cfg, None if not ov else ov)}")
lines += ["RESOLVE " + " ".join(rm)]

out = os.path.join(os.path.dirname(__file__), "cost_cases.txt")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}  cost={cost:.6g}")
