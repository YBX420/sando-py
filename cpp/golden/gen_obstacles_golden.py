"""Golden generator for obstacles.py (SphereObstacle / AABBObstacle) C++ port."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.local.obstacles import SphereObstacle, AABBObstacle

rng = np.random.default_rng(777)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []
# --- sphere cases ---
for _ in range(5):
    c0 = rng.uniform(-3, 3, 3); r = rng.uniform(0.2, 0.8)
    v = rng.normal(0, 1.0, 3); a = rng.normal(0, 0.5, 3)
    s = SphereObstacle(centre0=c0, radius=r, vel=v, accel=a, class_name="human")
    pts = rng.uniform(-5, 5, (8, 3)); ts = rng.uniform(0, 2.0, 8)
    pred = np.array([s.predict(t) for t in ts])
    velat = np.array([s.velocity_at(t) for t in ts])
    sd = np.array([s.signed_dist(pts[i], ts[i]) for i in range(8)])
    lines += ["SPHERE", f"C0 {fmt(c0)}", f"R {repr(float(r))}", f"V {fmt(v)}", f"A {fmt(a)}",
              f"N 8", f"PTS {fmt(pts)}", f"TS {fmt(ts)}",
              f"PRED {fmt(pred)}", f"VELAT {fmt(velat)}", f"SD {fmt(sd)}", "END"]

# --- aabb cases ---
for _ in range(5):
    c = rng.uniform(-3, 3, 3); half = rng.uniform(0.2, 1.5, 3)
    lo = c - half; hi = c + half
    b = AABBObstacle(lo=lo, hi=hi, class_name="wall")
    pts = rng.uniform(-5, 5, (8, 3))
    sd = np.array([b.signed_dist(pts[i], 0.0) for i in range(8)])
    lines += ["AABB", f"LO {fmt(lo)}", f"HI {fmt(hi)}",
              f"N 8", f"PTS {fmt(pts)}", f"SD {fmt(sd)}", "END"]

out = os.path.join(os.path.dirname(__file__), "obstacles_cases.txt")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}")
