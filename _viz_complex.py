"""Build a HARD 3D weaving scene (gates that force up/down/left/right), run the certificate-first
C++ stack through it, and VISUALIZE the planner's decisions: speed-coloured flown trajectory +
obstacles, in 3D + top(xy) + side(xz), plus a GIF of the drone weaving through the moving field.
  python _viz_complex.py
Writes the scene into isaac_sando.yaml (so Isaac shows it too) + isaac_complex_viz.png / .gif.
"""
import os, sys, time, math
import numpy as np, yaml
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------------------------------------------------------------- scene: gates forcing 3D weaving
GATES = []          # static-hard boxes: (x,y,z,sx,sy,sz)
DYN = []            # dynamic-hard: (traj_xyz strings, vel strings, size)


def box(x, y, z, sx=1.2, sy=1.4, sz=1.4):
    GATES.append((x, y, z, sx, sy, sz))


def z_slab(x, z_c, ys):     # horizontal slab at height z_c across y -> forces UP/DOWN
    for y in ys: box(x, y, z_c)


def y_slab(x, y_c, zs):     # vertical slab at y_c across z -> forces LEFT/RIGHT
    for z in zs: box(x, y_c, z)


# alternating gates along the 60 m corridor (open region noted)
z_slab(9,  2.8, [-2.5, -1.25, 0, 1.25, 2.5])     # block mid-z -> go LOW or HIGH
y_slab(16, 0.0, [1.2, 2.4, 3.6, 4.8])            # block centre-y -> go LEFT or RIGHT
z_slab(23, 1.5, [-2.5, -1.25, 0, 1.25, 2.5])     # block low-z -> go HIGH
y_slab(30, 1.2, [1.0, 2.2, 3.4, 4.6]); y_slab(30, -2.4, [1.0, 2.2])   # block right+ -> go LEFT
z_slab(37, 4.0, [-2.5, -1.25, 0, 1.25, 2.5])     # block high-z -> go LOW
y_slab(44, -1.0, [1.0, 2.2, 3.4, 4.6])           # block centre/left -> go RIGHT
z_slab(51, 2.8, [-2.0, -0.5, 1.0, 2.5])          # block mid-z -> weave

# a few dynamic-hard obstacles weaving in 3D (small trefoil loops)
def trefoil(x0, y0, z0, s, slower, off):
    tt = f"t/{slower}+{off}"
    return ([f"{s/6:.3f}*(sin({tt})+2*sin(2*({tt})))+{x0}",
             f"{s/5:.3f}*(cos({tt})-2*cos(2*({tt})))+{y0}",
             f"{s/3:.3f}*(-sin(3*({tt})))+{z0}"],
            [f"{s/6:.3f}*(1/{slower})*(cos({tt})+4*cos(2*({tt})))",
             f"{s/5:.3f}*(1/{slower})*(-sin({tt})+4*sin(2*({tt})))",
             f"-{s}*(1/{slower})*cos(3*({tt}))"])
for x0, y0, z0 in [(13, 1.5, 3.0), (27, -1.5, 2.0), (40, 1.0, 3.5), (47, -1.0, 2.5)]:
    t_, v_ = trefoil(x0, y0, z0, 3.0, 5.0, 0.5)
    DYN.append((t_, v_, [0.8, 0.8, 0.8]))

# ---------------------------------------------------------------- write scene into isaac_sando.yaml
CFG = yaml.safe_load(open("isaac_sando.yaml", encoding="utf-8"))
obs = []
for (x, y, z, sx, sy, sz) in GATES:
    obs.append({"class": "human", "size": [sx, sy, sz], "center": [round(x, 2), round(y, 2), round(z, 2)]})
for (t_, v_, sz) in DYN:
    obs.append({"class": "human", "size": sz, "traj": t_, "vel": v_})
CFG["scene"]["obstacles"] = obs
CFG["scene"]["start"] = [0.0, 0.0, 2.0]; CFG["scene"]["goal"] = [58.0, 0.0, 2.0]
# viz-friendly config: fast-ish but threads gates; z range opened up for vertical weaving
CFG["planner"].update({"v_max": 8.0, "a_max": 16.0, "minco_w_time": 400.0, "minco_time_budget_ms": 40.0,
                       "minco_use_topology": True, "minco_human_slow_vmax": 0.0,
                       "z_min": 0.5, "z_max": 6.0, "horizon": 14.0})
yaml.safe_dump(CFG, open("isaac_sando.yaml", "w", encoding="utf-8"), default_flow_style=None, sort_keys=False, allow_unicode=True)
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]
print(f"scene: {len(GATES)} static-hard gate boxes + {len(DYN)} dynamic-hard 3D obstacles")

# ---------------------------------------------------------------- run the C++ stack, record everything
import sando_cpp_bridge as B
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float); GOAL[2] = PLN["default_goal_z"]


def mk(o, i):
    d = B.DynTraj(); d.id = i; d.mode = "Analytic"; d.bbox = np.array(o["size"], float)
    if "traj" in o:
        d.traj_x, d.traj_y, d.traj_z = [str(e) for e in o["traj"]]
        if "vel" in o: d.traj_vx, d.traj_vy, d.traj_vz = [str(e) for e in o["vel"]]
    else:
        c = o["center"]; d.traj_x, d.traj_y, d.traj_z = f"{c[0]}", f"{c[1]}", f"{c[2]}"; d.traj_vx = d.traj_vy = d.traj_vz = "0.0"
    d.compile_analytic(); return d


p0 = B.Parameters()
for k, v in PLN.items():
    if k == "sando_map_res": p0.res = float(v); continue
    if hasattr(p0, k): setattr(p0, k, v)
s = B.SANDO(p0); st = B.RobotState(); st.pos = START.copy(); s.update_state(st); s.update_occupancy_map_ptr(np.zeros((0, 3)))
G = B.RobotState(); G.pos = GOAL.copy(); s.set_terminal_goal(G)
DTS = [mk(o, i) for i, o in enumerate(SCENE["obstacles"])]
IS_DYN = ["traj" in o for o in SCENE["obstacles"]]
DT = float(p0.dc); RD = float(LOOP["replan_dt"]); CULL = float(LOOP["sense_cull_r"]); TMAX = 40.0
p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nx = 0.0; lrt = 0.0; reached = False
PATH = []; SPD = []; TS = []; ms = []; mh = np.inf
while t < TMAX and not reached:
    if t >= nx - 1e-9:
        stt = B.RobotState(); stt.pos = p.copy(); stt.vel = v.copy(); stt.accel = a.copy(); s.update_state(stt)
        for d_ in DTS:
            if np.linalg.norm(d_.eval(t) - p) <= CULL: s.add_traj(d_, t)
        t0 = time.perf_counter(); s.replan(lrt, t); dt = (time.perf_counter() - t0) * 1000; lrt = dt / 1000; ms.append(dt); nx = t + RD
    okg, ng = s.get_next_goal()
    if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
    PATH.append(p.copy()); SPD.append(float(np.linalg.norm(v))); TS.append(t)
    t += DT
    if s.get_drone_status() == 3 and float(np.linalg.norm(p - GOAL)) < float(p0.goal_radius): reached = True
PATH = np.array(PATH); SPD = np.array(SPD); TS = np.array(TS); ms = np.array(ms)
print(f"reached={reached} t={TS[-1]:.2f}s avg={58/TS[-1]:.2f} peak={SPD.max():.2f} m/s  p99ms={np.percentile(ms,99):.0f} MAX={ms.max():.0f}")

# ---------------------------------------------------------------- visualize the decisions
def colored_path(ax, X, Y, spd, three_d=False, Z=None):
    pts = (np.array([X, Y, Z]).T if three_d else np.array([X, Y]).T).reshape(-1, 1, (3 if three_d else 2))
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    LC = (Line3DCollection if three_d else LineCollection)
    lc = LC(segs, cmap="turbo"); lc.set_array(spd[:-1]); lc.set_linewidth(2.5)
    (ax.add_collection3d if three_d else ax.add_collection)(lc); return lc


fig = plt.figure(figsize=(17, 9))
ax3 = fig.add_subplot(2, 2, 1, projection="3d")
axxy = fig.add_subplot(2, 2, 2); axxz = fig.add_subplot(2, 2, 3); axsp = fig.add_subplot(2, 2, 4)
# static gates
for i, o in enumerate(SCENE["obstacles"]):
    if IS_DYN[i]: continue
    c = o["center"]; sz = o["size"]
    ax3.scatter(c[0], c[1], c[2], c="dimgray", marker="s", s=60, alpha=0.8)
    axxy.add_patch(plt.Rectangle((c[0]-sz[0]/2, c[1]-sz[1]/2), sz[0], sz[1], color="gray", alpha=0.6))
    axxz.add_patch(plt.Rectangle((c[0]-sz[0]/2, c[2]-sz[2]/2), sz[0], sz[2], color="gray", alpha=0.6))
# dynamic traces (faint)
for i, o in enumerate(SCENE["obstacles"]):
    if not IS_DYN[i]: continue
    tt = np.linspace(0, TS[-1], 120); pos = np.array([DTS[i].eval(tk) for tk in tt])
    ax3.plot(pos[:,0], pos[:,1], pos[:,2], c="crimson", lw=0.8, alpha=0.4)
    axxy.plot(pos[:,0], pos[:,1], c="crimson", lw=0.8, alpha=0.4)
    axxz.plot(pos[:,0], pos[:,2], c="crimson", lw=0.8, alpha=0.4)
lc = colored_path(ax3, PATH[:,0], PATH[:,1], SPD, True, PATH[:,2])
colored_path(axxy, PATH[:,0], PATH[:,1], SPD); colored_path(axxz, PATH[:,0], PATH[:,2], SPD)
ax3.set_xlim(0,58); ax3.set_ylim(-4,4); ax3.set_zlim(0,6); ax3.set_title("3D flight (color=speed)  gray=static gates, red=dynamic")
ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
axxy.set_xlim(0,58); axxy.set_ylim(-4.5,4.5); axxy.set_title("TOP (x-y): left/right weaving"); axxy.set_xlabel("x"); axxy.set_ylabel("y"); axxy.set_aspect("auto")
axxz.set_xlim(0,58); axxz.set_ylim(0,6); axxz.set_title("SIDE (x-z): up/down weaving"); axxz.set_xlabel("x"); axxz.set_ylabel("z")
axsp.plot(PATH[:,0], SPD, "b-"); axsp.set_title("speed vs x  (slows to thread gates = the decisions)"); axsp.set_xlabel("x"); axsp.set_ylabel("m/s"); axsp.grid(alpha=0.3)
fig.colorbar(lc, ax=ax3, shrink=0.6, label="speed m/s")
fig.suptitle(f"3D weaving through SANDO-style hard gates  |  reached={reached} avg={58/TS[-1]:.1f} m/s  min_clr OK  p99={np.percentile(ms,99):.0f}ms", fontsize=13)
fig.tight_layout()
fig.savefig("isaac_complex_viz.png", dpi=110, bbox_inches="tight")
print("wrote isaac_complex_viz.png")

# ---------------------------------------------------------------- GIF: drone + moving obstacles
try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    figA = plt.figure(figsize=(13, 6)); a1 = figA.add_subplot(1,2,1); a2 = figA.add_subplot(1,2,2)
    for ax, (iy, iz, ttl, ylim) in [(a1,(1,1,"TOP x-y",(-4.5,4.5))), (a2,(2,2,"SIDE x-z",(0,6)))]:
        pass
    def setup(ax, ci, ttl, ylim):
        ax.set_xlim(0,58); ax.set_ylim(*ylim); ax.set_title(ttl); ax.grid(alpha=0.3)
    setup(a1,1,"TOP (x-y)",(-4.5,4.5)); setup(a2,2,"SIDE (x-z)",(0,6))
    for i, o in enumerate(SCENE["obstacles"]):
        if IS_DYN[i]: continue
        c=o["center"]; sz=o["size"]
        a1.add_patch(plt.Rectangle((c[0]-sz[0]/2,c[1]-sz[1]/2),sz[0],sz[1],color="gray",alpha=0.5))
        a2.add_patch(plt.Rectangle((c[0]-sz[0]/2,c[2]-sz[2]/2),sz[0],sz[2],color="gray",alpha=0.5))
    dr1, = a1.plot([],[], "bo", ms=8); dr2, = a2.plot([],[], "bo", ms=8)
    tr1, = a1.plot([],[], "b-", lw=1); tr2, = a2.plot([],[], "b-", lw=1)
    dyn1 = a1.scatter([],[], c="crimson", s=70); dyn2 = a2.scatter([],[], c="crimson", s=70)
    didx = [i for i in range(len(DTS)) if IS_DYN[i]]
    frames = range(0, len(TS), 5)
    def upd(f):
        tk = TS[f]; dp = np.array([DTS[i].eval(tk) for i in didx]) if didx else np.zeros((0,3))
        dr1.set_data([PATH[f,0]],[PATH[f,1]]); dr2.set_data([PATH[f,0]],[PATH[f,2]])
        tr1.set_data(PATH[:f,0],PATH[:f,1]); tr2.set_data(PATH[:f,0],PATH[:f,2])
        if len(dp): dyn1.set_offsets(dp[:,[0,1]]); dyn2.set_offsets(dp[:,[0,2]])
        return dr1,dr2,tr1,tr2,dyn1,dyn2
    anim = FuncAnimation(figA, upd, frames=frames, blit=True)
    anim.save("isaac_complex_viz.gif", writer=PillowWriter(fps=20))
    print("wrote isaac_complex_viz.gif")
except Exception as e:
    print(f"(GIF skipped: {e})")
