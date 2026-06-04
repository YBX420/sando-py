"""HARD 'penetrate-to-centroid' scene: a 3D porous obstacle CLOUD with the GOAL at its CENTRE OF
MASS — the drone must thread INWARD into the cluster and hold at the buried centre (no fly-around).
Runs the certificate-first C++ stack and visualizes the inward decision (3D + top + side + speed).
  python _viz_cloud.py   -> writes isaac_sando.yaml + isaac_cloud_viz.png / .gif
"""
import os, sys, time, math, random
import numpy as np, yaml
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------------------------------------------------------------- build a porous 3D obstacle cloud
C = np.array([30.0, 0.0, 2.5])     # cloud centre
R, POCKET, MIN_SEP = 7.0, 1.8, 2.7 # sphere radius, clear pocket at centre, min obstacle spacing
rng = random.Random(5)
pts = []
tries = 0
while len(pts) < 42 and tries < 12000:
    tries += 1
    # uniform-ish point in a ball around C
    d = np.array([rng.gauss(0, 1) for _ in range(3)]); d /= (np.linalg.norm(d) + 1e-9)
    q = C + d * (R * rng.random() ** (1 / 3.0))
    q[2] = min(5.3, max(1.0, q[2]))
    if np.linalg.norm(q - C) < POCKET: continue            # keep the centre (goal pocket) clear
    if any(np.linalg.norm(q - np.array(p)) < MIN_SEP for p in pts): continue
    pts.append([round(float(q[0]), 2), round(float(q[1]), 2), round(float(q[2]), 2)])

n_dyn = int(len(pts) * 0.3)


def trefoil(x0, y0, z0, s=2.2, slower=5.0, off=0.5):
    tt = f"t/{slower}+{off}"
    return ([f"{s/6:.3f}*(sin({tt})+2*sin(2*({tt})))+{x0}",
             f"{s/5:.3f}*(cos({tt})-2*cos(2*({tt})))+{y0}",
             f"{s/4:.3f}*(-sin(3*({tt})))+{z0}"],
            [f"{s/6:.3f}*(1/{slower})*(cos({tt})+4*cos(2*({tt})))",
             f"{s/5:.3f}*(1/{slower})*(-sin({tt})+4*sin(2*({tt})))",
             f"-{3*s/4:.3f}*(1/{slower})*cos(3*({tt}))"])


obs = []
for i, q in enumerate(pts):
    if i < n_dyn:
        t_, v_ = trefoil(*q)
        obs.append({"class": "human", "size": [0.8, 0.8, 0.8], "traj": t_, "vel": v_})
    else:
        obs.append({"class": "human", "size": [0.8, 0.8, 0.8], "center": q})

centroid = np.mean(np.array(pts), axis=0)            # GOAL = centre of mass of the obstacle cloud
GOALP = [round(float(centroid[0]), 2), round(float(centroid[1]), 2), round(float(centroid[2]), 2)]
print(f"cloud: {len(pts)} obstacles ({n_dyn} dynamic) in R={R} ball; GOAL at centroid {GOALP}")

# ---------------------------------------------------------------- write into isaac_sando.yaml
CFG = yaml.safe_load(open("isaac_sando.yaml", encoding="utf-8"))
CFG["scene"]["obstacles"] = obs
CFG["scene"]["start"] = [0.0, 0.0, 2.5]
CFG["scene"]["goal"] = GOALP
CFG["planner"].update({"v_max": 6.0, "a_max": 12.0, "minco_w_time": 200.0, "minco_time_budget_ms": 40.0,
                       "minco_use_topology": True, "minco_human_slow_vmax": 0.0, "force_goal_z": False,
                       "z_min": 0.5, "z_max": 6.0, "horizon": 14.0, "goal_radius": 0.8})
yaml.safe_dump(CFG, open("isaac_sando.yaml", "w", encoding="utf-8"), default_flow_style=None, sort_keys=False, allow_unicode=True)
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]

# ---------------------------------------------------------------- run the stack INTO the cloud
import sando_cpp_bridge as B
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)


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
DT = float(p0.dc); RD = float(LOOP["replan_dt"]); CULL = float(LOOP["sense_cull_r"])
p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nx = 0.0; lrt = 0.0; reached = False
PATH = []; SPD = []; TS = []; ms = []; mh = np.inf
while t < 45.0 and not reached:
    if t >= nx - 1e-9:
        stt = B.RobotState(); stt.pos = p.copy(); stt.vel = v.copy(); stt.accel = a.copy(); s.update_state(stt)
        for d_ in DTS:
            if np.linalg.norm(d_.eval(t) - p) <= CULL: s.add_traj(d_, t)
        t0 = time.perf_counter(); s.replan(lrt, t); dt = (time.perf_counter() - t0) * 1000; lrt = dt / 1000; ms.append(dt); nx = t + RD
    okg, ng = s.get_next_goal()
    if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
    PATH.append(p.copy()); SPD.append(float(np.linalg.norm(v))); TS.append(t)
    for i in range(len(DTS)):
        c = DTS[i].eval(t); sz = np.array(SCENE["obstacles"][i]["size"], float)
        lo = c - sz / 2; hi = c + sz / 2; out = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
        dd = float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))
        mh = min(mh, dd)
    t += DT
    if np.linalg.norm(p - GOAL) < float(p0.goal_radius): reached = True
PATH = np.array(PATH); SPD = np.array(SPD); TS = np.array(TS); ms = np.array(ms)
dist_to_goal = float(np.linalg.norm(PATH[-1] - GOAL))
print(f"reached={reached} t={TS[-1]:.2f}s final_dist_to_centroid={dist_to_goal:.2f}m min_clr={mh:+.2f} p99ms={np.percentile(ms,99):.0f} MAX={ms.max():.0f}")

# ---------------------------------------------------------------- visualize the INWARD decision
def cpath(ax, X, Y, spd, td=False, Z=None):
    pts_ = (np.array([X, Y, Z]).T if td else np.array([X, Y]).T).reshape(-1, 1, (3 if td else 2))
    segs = np.concatenate([pts_[:-1], pts_[1:]], axis=1)
    lc = (Line3DCollection if td else LineCollection)(segs, cmap="turbo"); lc.set_array(spd[:-1]); lc.set_linewidth(2.5)
    (ax.add_collection3d if td else ax.add_collection)(lc); return lc


fig = plt.figure(figsize=(17, 9))
ax3 = fig.add_subplot(2, 2, 1, projection="3d"); axxy = fig.add_subplot(2, 2, 2)
axxz = fig.add_subplot(2, 2, 3); axsp = fig.add_subplot(2, 2, 4)
for i, o in enumerate(SCENE["obstacles"]):
    c = (DTS[i].eval(0.0) if IS_DYN[i] else np.array(o["center"], float)); col = "crimson" if IS_DYN[i] else "dimgray"
    ax3.scatter(*c, c=col, marker="o", s=80, alpha=0.55)
    axxy.add_patch(plt.Circle((c[0], c[1]), 0.5, color=col, alpha=0.45))
    axxz.add_patch(plt.Circle((c[0], c[2]), 0.5, color=col, alpha=0.45))
    if IS_DYN[i]:
        tr = np.array([DTS[i].eval(tk) for tk in np.linspace(0, TS[-1], 80)])
        ax3.plot(tr[:,0], tr[:,1], tr[:,2], "crimson", lw=0.6, alpha=0.3)
        axxy.plot(tr[:,0], tr[:,1], "crimson", lw=0.6, alpha=0.3); axxz.plot(tr[:,0], tr[:,2], "crimson", lw=0.6, alpha=0.3)
lc = cpath(ax3, PATH[:,0], PATH[:,1], SPD, True, PATH[:,2])
cpath(axxy, PATH[:,0], PATH[:,1], SPD); cpath(axxz, PATH[:,0], PATH[:,2], SPD)
for ax, gi in [(axxy, (0,1)), (axxz, (0,2))]:
    ax.scatter(GOAL[gi[0]], GOAL[gi[1]], c="lime", marker="*", s=420, edgecolor="k", zorder=5, label="GOAL=centroid")
    ax.scatter(START[gi[0]], START[gi[1]], c="blue", marker="o", s=90, zorder=5, label="start")
ax3.scatter(*GOAL, c="lime", marker="*", s=420, edgecolor="k"); ax3.scatter(*START, c="blue", marker="o", s=90)
ax3.set_title("3D: drone threads INWARD to the cloud centroid (★)"); ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
ax3.set_xlim(0,40); ax3.set_ylim(-9,9); ax3.set_zlim(0,6)
axxy.set_title("TOP (x-y): penetrate to ★ centroid"); axxy.set_xlabel("x"); axxy.set_ylabel("y"); axxy.legend(loc="upper left", fontsize=8); axxy.set_xlim(-2,40); axxy.set_ylim(-9,9)
axxz.set_title("SIDE (x-z): inward + vertical"); axxz.set_xlabel("x"); axxz.set_ylabel("z"); axxz.set_xlim(-2,40); axxz.set_ylim(0,6)
axsp.plot(np.linalg.norm(PATH - GOAL, axis=1), SPD, "b-"); axsp.set_title("speed vs distance-to-centroid (slows as it enters)"); axsp.set_xlabel("dist to goal (m)"); axsp.set_ylabel("m/s"); axsp.grid(alpha=0.3); axsp.invert_xaxis()
fig.colorbar(lc, ax=ax3, shrink=0.6, label="speed m/s")
fig.suptitle(f"Penetrate to centroid of a 3D hard-obstacle cloud  |  reached={reached}  final_dist={dist_to_goal:.2f}m  min_clr={mh:+.2f}m  p99={np.percentile(ms,99):.0f}ms", fontsize=13)
fig.tight_layout(); fig.savefig("isaac_cloud_viz.png", dpi=110, bbox_inches="tight"); print("wrote isaac_cloud_viz.png")

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    figA = plt.figure(figsize=(13, 6)); a1 = figA.add_subplot(1,2,1); a2 = figA.add_subplot(1,2,2)
    for ax, ttl, yl in [(a1,"TOP (x-y)",(-9,9)), (a2,"SIDE (x-z)",(0,6))]:
        ax.set_xlim(-2,40); ax.set_ylim(*yl); ax.set_title(ttl); ax.grid(alpha=0.3)
    a1.scatter(GOAL[0],GOAL[1],c="lime",marker="*",s=380,edgecolor="k",zorder=5); a2.scatter(GOAL[0],GOAL[2],c="lime",marker="*",s=380,edgecolor="k",zorder=5)
    stat1=[];
    for i,o in enumerate(SCENE["obstacles"]):
        if not IS_DYN[i]:
            c=o["center"]; a1.add_patch(plt.Circle((c[0],c[1]),0.5,color="gray",alpha=0.4)); a2.add_patch(plt.Circle((c[0],c[2]),0.5,color="gray",alpha=0.4))
    dr1,=a1.plot([],[],"bo",ms=8); dr2,=a2.plot([],[],"bo",ms=8); tr1,=a1.plot([],[],"b-",lw=1); tr2,=a2.plot([],[],"b-",lw=1)
    dy1=a1.scatter([],[],c="crimson",s=70); dy2=a2.scatter([],[],c="crimson",s=70)
    didx=[i for i in range(len(DTS)) if IS_DYN[i]]
    def upd(f):
        tk=TS[f]; dp=np.array([DTS[i].eval(tk) for i in didx]) if didx else np.zeros((0,3))
        dr1.set_data([PATH[f,0]],[PATH[f,1]]); dr2.set_data([PATH[f,0]],[PATH[f,2]])
        tr1.set_data(PATH[:f,0],PATH[:f,1]); tr2.set_data(PATH[:f,0],PATH[:f,2])
        if len(dp): dy1.set_offsets(dp[:,[0,1]]); dy2.set_offsets(dp[:,[0,2]])
        return dr1,dr2,tr1,tr2,dy1,dy2
    FuncAnimation(figA, upd, frames=range(0,len(TS),5), blit=True).save("isaac_cloud_viz.gif", writer=PillowWriter(fps=20))
    print("wrote isaac_cloud_viz.gif")
except Exception as e:
    print(f"(GIF skipped: {e})")
