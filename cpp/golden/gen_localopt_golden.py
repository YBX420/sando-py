"""Golden generator for local_opt.py geometric core: _signed_dist_and_grad + _seg_vander."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, ROOT)
from sando_py.local.obstacles import SphereObstacle, AABBObstacle
from sando_py.local.local_opt import _signed_dist_and_grad, _seg_vander

rng = np.random.default_rng(99)


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []

# --- sphere (moving) signed_dist_and_grad ---
for _ in range(4):
    c0 = rng.uniform(-3, 3, 3); r = rng.uniform(0.2, 0.8)
    v = rng.normal(0, 1.0, 3); a = rng.normal(0, 0.4, 3)
    s = SphereObstacle(centre0=c0, radius=r, vel=v, accel=a, class_name="human")
    for _ in range(6):
        # keep off the kink: ensure ||p - c(t)|| not tiny
        t = rng.uniform(0, 1.5)
        c = s.predict(t)
        p = c + rng.normal(0, 1.5, 3)
        if np.linalg.norm(p - c) < 0.3:
            p = c + np.array([1.0, 0.0, 0.0])
        d, dd_dp, dd_dt = _signed_dist_and_grad(s, p, t)
        lines += ["SDG_SPH", f"C0 {fmt(c0)}", f"R {repr(float(r))}", f"V {fmt(v)}", f"A {fmt(a)}",
                  f"P {fmt(p)}", f"T {repr(float(t))}",
                  f"D {repr(float(d))}", f"DDP {fmt(dd_dp)}", f"DDT {repr(float(dd_dt))}", "END"]

# --- aabb signed_dist_and_grad (clearly outside, and clearly inside) ---
for _ in range(4):
    c = rng.uniform(-3, 3, 3); half = rng.uniform(0.3, 1.2, 3)
    lo = c - half; hi = c + half
    b = AABBObstacle(lo=lo, hi=hi, class_name="wall")
    # outside point: push out along a random axis well beyond a face
    for _ in range(3):
        p = c + (half + rng.uniform(0.4, 2.0)) * np.sign(rng.normal(size=3))
        d, dd_dp, dd_dt = _signed_dist_and_grad(b, p, 0.0)
        lines += ["SDG_BOX", f"LO {fmt(lo)}", f"HI {fmt(hi)}", f"P {fmt(p)}", f"T 0.0",
                  f"D {repr(float(d))}", f"DDP {fmt(dd_dp)}", f"DDT 0.0", "END"]
    # inside point: near center, but make one axis the clear max-penetration (avoid ties)
    for _ in range(2):
        off = rng.uniform(-0.1, 0.1, 3); off[0] += 0.05  # bias axis-0 to break ties
        p = c + off * half
        d, dd_dp, dd_dt = _signed_dist_and_grad(b, p, 0.0)
        lines += ["SDG_BOX", f"LO {fmt(lo)}", f"HI {fmt(hi)}", f"P {fmt(p)}", f"T 0.0",
                  f"D {repr(float(d))}", f"DDP {fmt(dd_dp)}", f"DDT 0.0", "END"]

# --- seg_vander ---
for kappa in [4, 8, 16]:
    taus = rng.uniform(0.0, 1.2, kappa)
    B0, B1, B2, B3 = _seg_vander(taus)
    lines += ["VANDER", f"K {kappa}", f"TAUS {fmt(taus)}",
              f"B0 {fmt(B0)}", f"B1 {fmt(B1)}", f"B2 {fmt(B2)}", f"B3 {fmt(B3)}", "END"]

out = os.path.join(os.path.dirname(__file__), "localopt_cases.txt")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}")
