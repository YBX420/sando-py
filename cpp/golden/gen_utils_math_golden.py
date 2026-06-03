"""Golden generator for utils.py numeric helpers C++ port."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from sando_py import utils as U

rng = np.random.default_rng(2027)
def fmt(a): return " ".join(repr(float(x)) for x in np.asarray(a, float).reshape(-1))

lines = []
# angle_wrap / angle_diff
aw = []
for a in list(rng.uniform(-20, 20, 12)) + [np.pi, -np.pi, 0.0, 3*np.pi]:
    aw += [a, U.angle_wrap(float(a))]
lines += ["ANGLEWRAP " + fmt(aw)]
ad = []
for _ in range(8):
    t, c = rng.uniform(-10, 10, 2); ad += [t, c, U.angle_diff(float(t), float(c))]
lines += ["ANGLEDIFF " + fmt(ad)]
# yaw<->quat
yq = []
for _ in range(8):
    y = rng.uniform(-np.pi, np.pi); q = U.quat_from_yaw(float(y)); y2 = U.yaw_from_quat(*q)
    yq += [y, q[0], q[1], q[2], q[3], y2]
lines += ["YAWQUAT " + fmt(yq)]
# saturate
for _ in range(6):
    v = rng.normal(0, 2, 3); vmax = float(rng.uniform(0.3, 3))
    lines += ["SAT", f"V {fmt(v)}", f"VMAX {repr(vmax)}", f"OUT {fmt(U.saturate(v, vmax))}", "END"]
# project_point_to_sphere
for _ in range(6):
    P1 = rng.uniform(-3,3,3); P2 = rng.uniform(-6,6,3); r = float(rng.uniform(0.5,3))
    lines += ["PSPH", f"P1 {fmt(P1)}", f"P2 {fmt(P2)}", f"R {repr(r)}", f"OUT {fmt(U.project_point_to_sphere(P1,P2,r))}", "END"]
# project_point_to_box
for _ in range(6):
    P1 = rng.uniform(-3,3,3); P2 = rng.uniform(-6,6,3); w = rng.uniform(0.5,3,3)
    lines += ["PBOX", f"P1 {fmt(P1)}", f"P2 {fmt(P2)}", f"W {fmt(w)}",
              f"OUT {fmt(U.project_point_to_box(P1,P2,float(w[0]),float(w[1]),float(w[2])))}", "END"]
# min_time_double_integrator 1d/3d
mt = []
for _ in range(10):
    p0,v0,pf,vf = rng.uniform(-5,5,4); vmax=float(rng.uniform(1,4)); amax=float(rng.uniform(1,6))
    mt += [p0,v0,pf,vf,vmax,amax, U.min_time_double_integrator_1d(float(p0),float(v0),float(pf),float(vf),vmax,amax)]
lines += ["MT1D " + fmt(mt)]
for _ in range(5):
    p0=rng.uniform(-5,5,3);v0=rng.uniform(-2,2,3);pf=rng.uniform(-5,5,3);vf=rng.uniform(-2,2,3)
    vm=rng.uniform(1,4,3);am=rng.uniform(1,6,3)
    lines += ["MT3D", f"P0 {fmt(p0)}", f"V0 {fmt(v0)}", f"PF {fmt(pf)}", f"VF {fmt(vf)}",
              f"VM {fmt(vm)}", f"AM {fmt(am)}", f"T {repr(float(U.min_time_double_integrator_3d(p0,v0,pf,vf,vm,am)))}", "END"]
# create_more_vertexes
for _ in range(4):
    npt = int(rng.integers(2,6)); path=[rng.uniform(-5,5,3) for _ in range(npt)]; d=float(rng.uniform(0.3,2))
    out = U.create_more_vertexes(path, d)
    lines += ["CMV", f"N {npt}", f"PATH {fmt(np.array(path))}", f"D {repr(d)}",
              f"NOUT {len(out)}", f"OUT {fmt(np.array(out))}", "END"]
# transform_inverse_se3 (build a random SE3)
for _ in range(4):
    y=rng.uniform(-np.pi,np.pi); qx,qy,qz,qw=U.quat_from_yaw(float(y)); t=rng.uniform(-3,3,3)
    M=U.transform_stamped_to_matrix(type("T",(),{"transform":type("Tr",(),{
        "translation":type("Tt",(),{"x":t[0],"y":t[1],"z":t[2]})(),
        "rotation":type("Tr2",(),{"x":qx,"y":qy,"z":qz,"w":qw})()})()})())
    Minv=U.transform_inverse_se3(M)
    lines += ["SE3", f"M {fmt(M)}", f"MINV {fmt(Minv)}", "END"]

out = os.path.join(os.path.dirname(__file__), "utils_math_cases.txt")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}")
