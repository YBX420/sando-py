"""Generate a SANDO-style 'static hard + dynamic hard' obstacle scene for isaac_sando.yaml.
Replicates SANDO scripts/run_sim.py:_trefoil_expr + _generate_obstacle_json (dynamic 0.8 cubes on
trefoil-knot paths + static pillars/bars), in the 60 m corridor with a start/goal exclusion zone.
Both static and dynamic are HARD (class 'human' -> hard). Prints a YAML obstacle block to paste."""
import math, random

# —— SANDO 参数(scripts/run_sim.py 原样)——
scale_range = [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]
offset_range = [0.0, 3.0]
slower_range = [4.0, 6.0]
bbox_dynamic = [0.8, 0.8, 0.8]
bbox_static_vert = [0.4, 0.4, 4.0]
percentage_vert = 0.5
NUM, DYN_RATIO, SEED = 18, 0.65, 7
# 60m 走廊;起终点附近留出 exclusion(±2m)
X0, X1, Y0, Y1, Z0, Z1 = 6.0, 54.0, -6.0, 6.0, 1.5, 2.5
EXCL = (-3.0, 3.0, -2.0, 2.0)  # near start; goal handled by x<54


def trefoil(x0, y0, z0, sx, sy, sz, off, slower):
    tt = f"t/{slower:.3f}+{off:.3f}"
    return (f"{sx/6:.3f}*(sin({tt})+2*sin(2*({tt})))+{x0:.2f}",
            f"{sy/5:.3f}*(cos({tt})-2*cos(2*({tt})))+{y0:.2f}",
            f"{z0:.2f}",  # 钉住 z(走廊里飞 z=2,不让障碍上下窜出平面)
            f"{sx/6:.3f}*(1/{slower:.3f})*(cos({tt})+4*cos(2*({tt})))",
            f"{sy/5:.3f}*(1/{slower:.3f})*(-sin({tt})+4*sin(2*({tt})))",
            "0.0")


rng = random.Random(SEED)
nd = int(NUM * DYN_RATIO); ns = NUM - nd
lines = []
for i in range(NUM):
    dyn = i < nd
    while True:
        x = X0 + (X1 - X0) * rng.random(); y = Y0 + (Y1 - Y0) * rng.random(); z = 2.0
        if not (EXCL[0] <= x <= EXCL[1] and EXCL[2] <= y <= EXCL[3]):
            break
    if dyn:
        sx = rng.uniform(*scale_range[0]); sy = rng.uniform(*scale_range[1]); sz = rng.uniform(*scale_range[2])
        off = rng.uniform(*offset_range); sl = rng.uniform(*slower_range)
        tx, ty, tz, vx, vy, vz = trefoil(x, y, z, sx, sy, sz, off, sl)
        b = bbox_dynamic
        lines.append(f'    - {{class: human, size: [{b[0]}, {b[1]}, {b[2]}], '
                     f'traj: ["{tx}", "{ty}", "{tz}"], vel: ["{vx}", "{vy}", "{vz}"]}}  # dyn-hard')
    else:
        b = bbox_static_vert  # 竖柱(硬);z 钉在 2
        lines.append(f'    - {{class: human, size: [{b[0]}, {b[1]}, 3.0], center: [{x:.2f}, {y:.2f}, 2.0]}}  # static-hard')
print(f"# SANDO-style: {nd} dynamic-hard (trefoil 0.8 cubes) + {ns} static-hard pillars, corridor 0->60m")
print("\n".join(lines))
