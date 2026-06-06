"""Golden generator for obstacle_tracker_kernels.hpp.

Pulls the REAL kernel functions out of sando_py/nodes/obstacle_tracker.py by AST
(so the golden runs the actual source, not a retype) and execs them with numpy/scipy.
sklearn is absent here, so _euclidean_cluster takes its cKDTree+union-find fallback —
exactly what obstacle_tracker_kernels.hpp::connected_components reproduces.
"""
import os, sys, ast
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
SRC = os.path.join(ROOT, "sando_py", "nodes", "obstacle_tracker.py")

# --- extract the named defs verbatim from the source and exec them ---
src = open(SRC, "r", encoding="utf-8").read()
src_lines = src.splitlines()
tree = ast.parse(src)
want = {"EKFState", "_ekf_predict", "_aekf_update", "_associate_cluster",
        "_polyfit_descending", "_calculate_variance", "_euclidean_cluster", "_voxel_downsample"}
segs = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
        # include decorators (get_source_segment starts at the def/class line, dropping @dataclass)
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        segs.append("\n".join(src_lines[start - 1:node.end_lineno]))
ns = {}
from dataclasses import dataclass, field
import typing
exec("import numpy as np\nfrom dataclasses import dataclass, field\n"
     "from typing import List, Optional, Tuple\n" + "\n\n".join(segs), ns)

EKFState = ns["EKFState"]
_ekf_predict = ns["_ekf_predict"]
_aekf_update = ns["_aekf_update"]
_polyfit = ns["_polyfit_descending"]
_variance = ns["_calculate_variance"]
_cluster = ns["_euclidean_cluster"]
_voxel = ns["_voxel_downsample"]


def fmt(a):
    return " ".join(repr(float(x)) for x in np.asarray(a, dtype=float).reshape(-1))


lines = []
rng = np.random.default_rng(7)

# --- 1. voxel downsample ---
pts = np.array([[0.05, 0.05, 0.05], [0.06, 0.04, 0.05], [0.30, 0.10, 0.10],
                [0.31, 0.11, 0.09], [0.30, 0.10, 0.45], [-0.1, -0.1, 0.2],
                [-0.05, -0.12, 0.21], [1.0, 1.0, 1.0]], dtype=float)
leaf = 0.2
vd = _voxel(pts, leaf)
lines += [f"VOXEL_IN_N {pts.shape[0]}", f"VOXEL_IN {fmt(pts)}", f"VOXEL_LEAF {repr(leaf)}",
          f"VOXEL_OUT_N {vd.shape[0]}", f"VOXEL_OUT {fmt(vd)}"]

# --- 2. clustering (fallback union-find path) ---
blob1 = np.array([0.0, 0.0, 1.0]) + 0.1 * rng.standard_normal((12, 3))
blob2 = np.array([5.0, 5.0, 1.0]) + 0.1 * rng.standard_normal((15, 3))
noise = np.array([[2.5, 2.5, 3.0]])  # isolated -> own component, below min_size
cpts = np.vstack([blob1, blob2, noise])
eps, min_s, max_s = 2.0, 10, 2000
clusters = _cluster(cpts, eps, min_s, max_s)
# describe each cluster by centroid+bbox+size, sorted by centroid for order-independence
desc = []
for c in clusters:
    pmin = c.min(axis=0); pmax = c.max(axis=0)
    cen = (pmin + pmax) * 0.5; box = pmax - pmin
    desc.append((cen, box, c.shape[0]))
desc.sort(key=lambda d: (round(d[0][0], 6), round(d[0][1], 6), round(d[0][2], 6)))
lines += [f"CLUST_IN_N {cpts.shape[0]}", f"CLUST_IN {fmt(cpts)}",
          f"CLUST_PARAMS {repr(eps)} {min_s} {max_s}", f"CLUST_NCLUST {len(desc)}"]
for i, (cen, box, sz) in enumerate(desc):
    lines += [f"CLUST_{i} {fmt(cen)} {fmt(box)} {sz}"]

# --- 3. EKF predict + adaptive update ---
s = EKFState()
s.x = np.array([1.0, 2.0, 1.5, 0.3, -0.2, 0.1, 0.05, 0.0, -0.05])
s.P = np.diag([0.2, 0.3, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02]).astype(float)
s.Q = np.eye(9) * 0.01
s.R = np.eye(3) * 0.01
lines += [f"EKF_IN_X {fmt(s.x)}", f"EKF_IN_P {fmt(s.P)}", f"EKF_IN_Q {fmt(s.Q)}", f"EKF_IN_R {fmt(s.R)}"]
dt = 0.1
_ekf_predict(s, dt)
lines += [f"EKF_DT {repr(dt)}", f"EKF_PRED_X {fmt(s.x)}", f"EKF_PRED_P {fmt(s.P)}"]

z = np.array([1.05, 2.02, 1.48])
bbox = np.array([0.6, 0.6, 1.8])
alpha = 0.98
_aekf_update(s, z, alpha, 12.3, bbox, True)
lines += [f"AEKF_Z {fmt(z)}", f"AEKF_BBOX {fmt(bbox)}", f"AEKF_ALPHA {repr(alpha)}",
          f"AEKF_X {fmt(s.x)}", f"AEKF_P {fmt(s.P)}", f"AEKF_R {fmt(s.R)}",
          f"AEKF_Q {fmt(s.Q)}", f"AEKF_BBOX_OUT {fmt(s.bbox)}"]

# --- 4. polyfit + variance (cubic and quintic) ---
t = np.linspace(0.0, 2.0, 21)
y = 0.5 + 0.8 * t - 0.3 * t * t + 0.05 * t ** 3 + 0.02 * np.sin(3 * t)
b3 = _polyfit(t, y, 3); v3 = _variance(t, y, b3, 3)
b5 = _polyfit(t, y, 5); v5 = _variance(t, y, b5, 5)
lines += [f"POLY_T {fmt(t)}", f"POLY_Y {fmt(y)}",
          f"POLY_B3 {fmt(b3)}", f"POLY_V3 {repr(float(v3))}",
          f"POLY_B5 {fmt(b5)}", f"POLY_V5 {repr(float(v5))}"]

# --- 5. predict_samples (clipped const-accel extrapolation, verbatim loop) ---
state = np.array([1.0, 2.0, 1.5, 0.4, -0.3, 0.0, 1.0, -1.0, 0.0])
num_steps, pdt = 20, 0.1
pos = state[:3].copy(); vel = state[3:6].copy(); acc = np.clip(state[6:9].copy(), -2.0, 2.0)
tv = [0.0]; xv = [pos[0]]; yv = [pos[1]]; zv = [pos[2]]
for step in range(num_steps):
    vel = np.clip(vel, -0.5, 0.5)
    acc = np.clip(acc, -0.8, 0.8)
    tt = step * pdt
    future = pos + vel * tt + 0.5 * acc * tt * tt
    tv.append(tt); xv.append(future[0]); yv.append(future[1]); zv.append(future[2])
    pos = future
    vel = vel + acc * pdt
lines += [f"PRED_STATE {fmt(state)}", f"PRED_STEPS {num_steps}", f"PRED_DT {repr(pdt)}",
          f"PRED_T {fmt(tv)}", f"PRED_X {fmt(xv)}", f"PRED_Y {fmt(yv)}", f"PRED_Z {fmt(zv)}"]

out = os.path.join(os.path.dirname(__file__), "obstacle_tracker_cases.txt")
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}  ({len(desc)} clusters, voxel {pts.shape[0]}->{vd.shape[0]})")
