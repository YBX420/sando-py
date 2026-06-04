"""Directly COPY SANDO's scene: _trefoil_expr + _generate_obstacle_json are pasted VERBATIM from
sando-main/scripts/run_sim.py, called with SANDO's exact defaults (50 obstacles, 0.65 dynamic,
x in [5,100], y in [-6,6], z in [0.5,4.5], start (0,0,3)). All obstacles HARD. Goal at the field
end so the drone must fly THROUGH the whole obstacle field. Runs the C++ stack + visualizes.
"""
import os, sys, time, math, random as _rng_mod
import numpy as np, yaml
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ===== VERBATIM from sando-main/scripts/run_sim.py =========================================
def _trefoil_expr(x0, y0, z0, sx, sy, sz, offset, slower):
    """Trefoil knot position + velocity expression strings (matches dyn_obstacles.launch.py)."""
    tt = f"t/{slower}+{offset}"
    x_str = f"{sx / 6.0}*(sin({tt})+2*sin(2*{tt}))+{x0}"
    y_str = f"{sy / 5.0}*(cos({tt})-2*cos(2*{tt}))+{y0}"
    z_str = f"{sz / 2.0}*(-sin(3*{tt}))+{z0}"
    inv_s = f"(1/{slower})"
    vx_str = f"{sx / 6.0}*{inv_s}*(cos({tt})+4*cos(2*{tt}))"
    vy_str = f"{sy / 5.0}*{inv_s}*(-sin({tt})+4*sin(2*{tt}))"
    vz_str = f"-{3 * sz / 2.0}*{inv_s}*cos(3*{tt})"
    return x_str, y_str, z_str, vx_str, vy_str, vz_str


def _generate_obstacle_json(num_obstacles, seed, x_min, x_max, y_min, y_max, z_min, z_max,
                            dynamic_ratio=1.0, exclusion_zone=None):
    import random as _rng
    _rng.seed(seed)
    scale_range = [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]
    offset_range = [0.0, 3.0]
    slower_range = [4.0, 6.0]
    bbox_dynamic = [0.8, 0.8, 0.8]
    bbox_static_vert = [0.4, 0.4, 4.0]
    bbox_static_horiz = [0.4, 4.0, 0.4]
    percentage_vert = 0.35
    num_dynamic = int(num_obstacles * dynamic_ratio)
    num_static = num_obstacles - num_dynamic
    obstacles = []
    for i in range(num_obstacles):
        is_dynamic = i < num_dynamic
        while True:
            x = x_min + (x_max - x_min) * _rng.random()
            y = y_min + (y_max - y_min) * _rng.random()
            z = z_min + (z_max - z_min) * _rng.random()
            if exclusion_zone is None:
                break
            ex_min, ex_max, ey_min, ey_max = exclusion_zone
            if not (ex_min <= x <= ex_max and ey_min <= y <= ey_max):
                break
        if is_dynamic:
            sx = _rng.uniform(*scale_range[0]); sy = _rng.uniform(*scale_range[1]); sz = _rng.uniform(*scale_range[2])
            offset = _rng.uniform(*offset_range); slower = _rng.uniform(*slower_range)
            x_str, y_str, z_str, vx_str, vy_str, vz_str = _trefoil_expr(x, y, z, sx, sy, sz, offset, slower)
            bbox = bbox_dynamic
        else:
            static_idx = i - num_dynamic
            is_vertical = static_idx < (num_static * percentage_vert)
            if is_vertical:
                z = bbox_static_vert[2] / 2.0
                bbox = bbox_static_vert
            else:
                num_horiz = num_static - int(num_static * percentage_vert)
                horiz_idx = static_idx - int(num_static * percentage_vert)
                if horiz_idx < num_horiz / 2:
                    bbox = bbox_static_horiz
                else:
                    bbox = [4.0, 0.4, 0.4]
            x_str = str(x); y_str = str(y); z_str = str(z)
            vx_str = "0.0"; vy_str = "0.0"; vz_str = "0.0"
        obstacles.append({"traj_x": x_str, "traj_y": y_str, "traj_z": z_str,
                          "traj_vx": vx_str, "traj_vy": vy_str, "traj_vz": vz_str,
                          "size": list(bbox), "is_dynamic": is_dynamic})
    return obstacles
# ===========================================================================================

# SANDO generator (verbatim above); density/field env-tunable. Dense+short => constant avoidance.
_iv = lambda k, d: int(os.environ.get(k, d))
_fv = lambda k, d: float(os.environ.get(k, d))
NOBS = _iv("NOBS", 50); XEND = _fv("XEND", 100.0); YH = _fv("YH", 6.0); ZHI = _fv("ZHI", 4.5); DIAG = _iv("DIAG", 0)
RAW = _generate_obstacle_json(NOBS, seed=1, x_min=5.0, x_max=XEND, y_min=-YH, y_max=YH,
                              z_min=0.5, z_max=ZHI, dynamic_ratio=0.65, exclusion_zone=(-3, 8, -YH/2, YH/2))
CLS = os.environ.get("CLASS", "human")  # human=all hard 0.8 | wall=all soft 0.4 | mixed=dyn human + static wall


def _cls(o):
    if CLS == "mixed":
        return "human" if o["is_dynamic"] else "wall"
    return CLS


obs = [{"class": _cls(o), "size": o["size"],
        "traj": [o["traj_x"], o["traj_y"], o["traj_z"]],
        "vel": [o["traj_vx"], o["traj_vy"], o["traj_vz"]]} for o in RAW]
n_dyn = sum(o["is_dynamic"] for o in RAW)
dens = NOBS / ((XEND - 5.0) * (2 * YH) * (ZHI - 0.5))
print(f"SANDO scene (verbatim generator): {NOBS} obstacles = {n_dyn} dynamic + {NOBS-n_dyn} static | field x[5,{XEND}] y[+-{YH}] z[0.5,{ZHI}] density={dens:.3f}/m3")

CFG = yaml.safe_load(open("isaac_sando.yaml", encoding="utf-8"))
CFG["scene"]["obstacles"] = obs
CFG["scene"]["start"] = [0.0, 0.0, 3.0]; CFG["scene"]["goal"] = [XEND + 5.0, 0.0, 3.0]
EV = lambda k, d: float(os.environ.get(k, d))
TAG = os.environ.get("TAG", "")
CFG["planner"].update({"v_max": EV("VMAX", 6.0), "a_max": EV("AMAX", 12.0), "minco_w_time": EV("WTIME", 200.0),
                       "minco_time_budget_ms": EV("BUDGET", 40.0), "minco_use_topology": True,
                       "minco_human_slow_vmax": 0.0, "force_goal_z": True, "default_goal_z": 3.0,
                       "z_min": 0.5, "z_max": 4.5, "horizon": EV("HORIZON", 14.0),
                       "minco_sfc_radius": EV("SFC", 0.0), "minco_w_corridor": EV("WCORR", 0.0)})
CFG["loop"]["t_max"] = EV("TMAX", 60.0)
yaml.safe_dump(CFG, open("isaac_sando.yaml", "w", encoding="utf-8"), default_flow_style=None, sort_keys=False, allow_unicode=True)
PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]

# ---------------------------------------------------------------- run the C++ stack through the field
import sando_cpp_bridge as B
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float); GOAL[2] = PLN["default_goal_z"]


def mk(o, i):
    d = B.DynTraj(); d.id = i; d.mode = "Analytic"; d.bbox = np.array(o["size"], float)
    d.traj_x, d.traj_y, d.traj_z = [str(e) for e in o["traj"]]
    d.traj_vx, d.traj_vy, d.traj_vz = [str(e) for e in o["vel"]]
    d.compile_analytic(); return d


p0 = B.Parameters()
for k, v in PLN.items():
    if k == "sando_map_res": p0.res = float(v); continue
    if hasattr(p0, k): setattr(p0, k, v)
s = B.SANDO(p0); st = B.RobotState(); st.pos = START.copy(); s.update_state(st); s.update_occupancy_map_ptr(np.zeros((0, 3)))
G = B.RobotState(); G.pos = GOAL.copy(); s.set_terminal_goal(G)
DTS = [mk(o, i) for i, o in enumerate(SCENE["obstacles"])]
IS_DYN = [bool(RAW[i]["is_dynamic"]) for i in range(len(RAW))]
DT = float(p0.dc); RD = float(LOOP["replan_dt"]); CULL = float(LOOP["sense_cull_r"])
p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nx = 0.0; lrt = 0.0; reached = False
PATH = []; SPD = []; TS = []; ms = []; mh = np.inf; CLRD = []; CLRS = []; GP = []
while t < float(LOOP["t_max"]) and not reached:
    if t >= nx - 1e-9:
        stt = B.RobotState(); stt.pos = p.copy(); stt.vel = v.copy(); stt.accel = a.copy(); s.update_state(stt)
        for d_ in DTS:
            if np.linalg.norm(d_.eval(t) - p) <= CULL: s.add_traj(d_, t)
        t0 = time.perf_counter(); s.replan(lrt, t); dt = (time.perf_counter() - t0) * 1000; lrt = dt / 1000; ms.append(dt); nx = t + RD
        if DIAG:
            gp = s.get_global_path(); gpx = max((q[0] for q in gp), default=-1.0)
            GP.append((float(p[0]), float(gpx), len(gp), int(s.get_drone_status())))
    okg, ng = s.get_next_goal()
    if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
    PATH.append(p.copy()); SPD.append(float(np.linalg.norm(v))); TS.append(t)
    md_dyn = np.inf; md_stat = np.inf
    for i in range(len(DTS)):
        c = DTS[i].eval(t); sz = np.array(SCENE["obstacles"][i]["size"], float)
        lo = c - sz / 2; hi = c + sz / 2; out = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
        dd = float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))
        mh = min(mh, dd)
        if IS_DYN[i]: md_dyn = min(md_dyn, dd)
        else: md_stat = min(md_stat, dd)
    CLRD.append(md_dyn); CLRS.append(md_stat)
    if s.get_drone_status() == 3 and np.linalg.norm(p - GOAL) < float(p0.goal_radius): reached = True
    t += DT
CLRD = np.array(CLRD); CLRS = np.array(CLRS)
PATH = np.array(PATH); SPD = np.array(SPD); TS = np.array(TS); ms = np.array(ms)
below2 = float(np.sum(SPD < 2.0)) * DT
backtrack = float(np.sum(np.maximum(0.0, -np.diff(PATH[:, 0]))))
over50 = int(np.sum(ms > 50))
print(f"[{TAG}] reached={reached} t={TS[-1]:.2f}s final_x={PATH[-1,0]:.1f} avg={PATH[-1,0]/TS[-1]:.2f} peak={SPD.max():.2f} "
      f"min_clr={mh:+.2f} p99ms={np.percentile(ms,99):.0f} MAX={ms.max():.0f} #>50={over50} | time<2m/s={below2:.1f}s x_backtrack={backtrack:.1f}m")
if DIAG and GP:
    # during the STUCK phase (drone x within 2m of its final x), where does the GLOBAL heat-A* path reach?
    xf = PATH[-1, 0]
    stuck = [(dx, gpx, n, stt) for (dx, gpx, n, stt) in GP if dx > xf - 2.0]
    gpx_stuck = [g[1] for g in stuck]
    print(f"[DIAG] goal_x={GOAL[0]:.1f}  drone_stuck_x~{xf:.1f}  | during stuck: GLOBAL path reach: "
          f"min={min(gpx_stuck):.1f} max={max(gpx_stuck):.1f} median={float(np.median(gpx_stuck)):.1f}  "
          f"(npts median={int(np.median([g[2] for g in stuck]))}, status set={sorted(set(g[3] for g in stuck))})")
    print(f"[DIAG] global reaches goal? {'YES (global path OK, LOCAL solver traps)' if max(gpx_stuck) >= GOAL[0]-2 else 'NO (GLOBAL planner itself stops short -> global is the bottleneck)'}")

# ---------------------------------------------------------------- visualize
def cpath(ax, X, Y, spd, td=False, Z=None):
    pts_ = (np.array([X, Y, Z]).T if td else np.array([X, Y]).T).reshape(-1, 1, (3 if td else 2))
    segs = np.concatenate([pts_[:-1], pts_[1:]], axis=1)
    lc = (Line3DCollection if td else LineCollection)(segs, cmap="turbo"); lc.set_array(spd[:-1]); lc.set_linewidth(2.0)
    (ax.add_collection3d if td else ax.add_collection)(lc); return lc


XL = XEND + 8.0
fig = plt.figure(figsize=(18, 12))
axxy = fig.add_subplot(4, 1, 1); axxz = fig.add_subplot(4, 1, 2); axsp = fig.add_subplot(4, 1, 3); axcl = fig.add_subplot(4, 1, 4)
for i, o in enumerate(SCENE["obstacles"]):
    c = DTS[i].eval(0.0); sz = o["size"]; col = "crimson" if IS_DYN[i] else "dimgray"
    axxy.add_patch(plt.Rectangle((c[0]-sz[0]/2, c[1]-sz[1]/2), sz[0], sz[1], color=col, alpha=0.45))
    axxz.add_patch(plt.Rectangle((c[0]-sz[0]/2, c[2]-sz[2]/2), sz[0], sz[2], color=col, alpha=0.45))
    if IS_DYN[i]:
        tr = np.array([DTS[i].eval(tk) for tk in np.linspace(0, TS[-1], 60)])
        axxy.plot(tr[:,0], tr[:,1], "crimson", lw=0.5, alpha=0.25); axxz.plot(tr[:,0], tr[:,2], "crimson", lw=0.5, alpha=0.25)
lc = cpath(axxy, PATH[:,0], PATH[:,1], SPD); cpath(axxz, PATH[:,0], PATH[:,2], SPD)
axxy.scatter(*START[[0,1]], c="blue", s=80, zorder=5); axxy.scatter(*GOAL[[0,1]], c="lime", marker="*", s=300, edgecolor="k", zorder=5)
axxz.scatter(*START[[0,2]], c="blue", s=80, zorder=5); axxz.scatter(*GOAL[[0,2]], c="lime", marker="*", s=300, edgecolor="k", zorder=5)
axxy.set_xlim(-2, XL); axxy.set_ylim(-7,7); axxy.set_title(f"TOP (x-y): fly THROUGH SANDO's 50-obstacle field  [CLASS={CLS}]  (gray=static, red=dynamic)"); axxy.set_ylabel("y")
axxz.set_xlim(-2, XL); axxz.set_ylim(0,5); axxz.set_title("SIDE (x-z): vertical weaving"); axxz.set_ylabel("z")
axsp.plot(PATH[:,0], SPD, "b-"); axsp.set_xlim(-2, XL); axsp.set_title("speed vs x (color above = same speed)"); axsp.set_ylabel("m/s"); axsp.grid(alpha=0.3); axsp.axhline(PLN["v_max"], ls=":", c="gray")
# per-class DECISION: clearance to nearest dynamic(hard) vs static(soft) along the path
axcl.plot(PATH[:,0], np.clip(CLRD, -1, 6), "crimson", lw=1.4, label="clearance to nearest DYNAMIC (hard)")
axcl.plot(PATH[:,0], np.clip(CLRS, -1, 6), "dimgray", lw=1.4, label="clearance to nearest STATIC (soft)")
axcl.axhline(0.0, ls="--", c="k", lw=0.8); axcl.set_xlim(-2, XL); axcl.set_ylim(-0.5,4)
axcl.set_title("PER-CLASS DECISION: keeps a bigger berth around dynamic-hard than static-soft"); axcl.set_xlabel("x"); axcl.set_ylabel("clearance m"); axcl.legend(loc="upper right", fontsize=9); axcl.grid(alpha=0.3)
fig.colorbar(lc, ax=[axxy, axxz], shrink=0.7, label="speed m/s")
fig.suptitle(f"SANDO scene (50 obstacles, generator copied verbatim)  |  CLASS={CLS}  reached={reached} avg={PATH[-1,0]/TS[-1]:.1f} m/s peak={SPD.max():.1f}  min_clr={mh:+.2f}m  p99={np.percentile(ms,99):.0f}ms", fontsize=13)
_png = f"isaac_sando_scene_viz{('_'+TAG) if TAG else ''}.png"
fig.savefig(_png, dpi=100, bbox_inches="tight"); print(f"wrote {_png}")

# ---- animation: drone + moving dynamic obstacles (best view of the decisions over time) ----
try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    figA = plt.figure(figsize=(16, 6)); g1 = figA.add_subplot(2, 1, 1); g2 = figA.add_subplot(2, 1, 2)
    for ax, ttl, yl in [(g1, "TOP (x-y)", (-7, 7)), (g2, "SIDE (x-z)", (0, 5))]:
        ax.set_xlim(-2, XL); ax.set_ylim(*yl); ax.set_title(ttl); ax.grid(alpha=0.3)
    for i, o in enumerate(SCENE["obstacles"]):
        if not IS_DYN[i]:
            c = DTS[i].eval(0.0); sz = o["size"]
            g1.add_patch(plt.Rectangle((c[0]-sz[0]/2, c[1]-sz[1]/2), sz[0], sz[1], color="gray", alpha=0.4))
            g2.add_patch(plt.Rectangle((c[0]-sz[0]/2, c[2]-sz[2]/2), sz[0], sz[2], color="gray", alpha=0.4))
    g1.scatter(GOAL[0], GOAL[1], c="lime", marker="*", s=260, edgecolor="k", zorder=6); g2.scatter(GOAL[0], GOAL[2], c="lime", marker="*", s=260, edgecolor="k", zorder=6)
    d1, = g1.plot([], [], "bo", ms=8); d2, = g2.plot([], [], "bo", ms=8); l1, = g1.plot([], [], "b-", lw=1.3); l2, = g2.plot([], [], "b-", lw=1.3)
    s1 = g1.scatter([], [], c="crimson", s=55); s2 = g2.scatter([], [], c="crimson", s=55)
    didx = [i for i in range(len(DTS)) if IS_DYN[i]]
    def upd(f):
        tk = TS[f]; dp = np.array([DTS[i].eval(tk) for i in didx])
        d1.set_data([PATH[f,0]], [PATH[f,1]]); d2.set_data([PATH[f,0]], [PATH[f,2]])
        l1.set_data(PATH[:f,0], PATH[:f,1]); l2.set_data(PATH[:f,0], PATH[:f,2])
        s1.set_offsets(dp[:, [0,1]]); s2.set_offsets(dp[:, [0,2]]); return d1, d2, l1, l2, s1, s2
    _gif = f"isaac_sando_scene_viz{('_'+TAG) if TAG else ''}.gif"
    figA.suptitle(f"SANDO scene fly-through  [CLASS={CLS}]  avg={PATH[-1,0]/TS[-1]:.1f} m/s", fontsize=12)
    FuncAnimation(figA, upd, frames=range(0, len(TS), 6), blit=True).save(_gif, writer=PillowWriter(fps=20)); print(f"wrote {_gif}")
except Exception as e:
    print(f"(GIF skipped: {e})")
