"""20 (or N) slightly-SIMPLIFIED random scenes -> one 3D gif each (media/_rand_NN.gif).
Each scene: a short corridor (x 0->GOAL_X) with 2-4 random STATIC walls (pillars/bars/blocks, class=wall
-> soft) + 1-2 crossing PEOPLE (class=human -> hard continuous-time cert), start->goal down the middle.
Closed-loop rollout through the real C++ engine (sando_cpp_bridge), drone trail in 3D, red dot on body
collision. Production-default planner (use_st_graph off); inflation_hgp left at yaml 0.45.
Run:  python _rand_scenes.py [N=20] [workers=5]
"""
import os, sys
# use yaml defaults -> clear the env knobs _speed_bench reads (avoid the INFL->inflation_hgp trap)
for k in ("INFL", "DYNINFL", "STG", "STC", "STDSD", "VMAX", "BUDGET", "WCOR", "SFC", "WTIME"):
    os.environ.pop(k, None)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)

HARD = os.environ.get("HARD", "0") == "1"   # complex scenes: long dense corridor, fast walls & people
PHYS = os.environ.get("PHYS", "1") == "1"   # model the drone's physical accel/brake (momentum), not teleport
GOAL_X = float(os.environ.get("GOALX", "200.0" if HARD else "24.0"))   # HARD: long tunnel (x->200)
VMAX = float(os.environ.get("VM", "10.0" if HARD else "6.0"))          # HARD: high vmax to chase avg ~7 m/s
T_MAX = (GOAL_X / 3.0 + 12.0) if HARD else 18.0                        # enough for ~3 m/s worst-case threading
_OMEGA = float(os.environ.get("OMEGA", "8.0"))   # position-loop bandwidth (rad/s); tighter = less overshoot
KP, KV = _OMEGA * _OMEGA, 2.0 * _OMEGA           # critically damped
# --- Intel RealSense D435i depth camera (FIXED sensor model -- do NOT tune for convenience) ---
# Depth FOV 87 deg (H) x 58 deg (V); usable obstacle range ~0.28 m (min-Z) to ~6 m (reliable depth).
CAM_RANGE_MIN = 0.28
CAM_RANGE = 6.0
CAM_HFOV_COS = float(np.cos(np.radians(43.5)))      # 87 deg horizontal half-angle
CAM_VFOV_TAN = float(np.tan(np.radians(29.0)))      # 58 deg vertical half-angle (elevation limit)
FRAMESTEP = int(os.environ.get("FRAMESTEP", "6"))   # gif: sample every Nth sim step (smaller = smoother)
FPS = int(os.environ.get("FPS", "14"))


def visible(p, heading, c, sz):
    """RealSense D435i-like depth perception: an obstacle is SEEN iff its nearest point is within
    [0.28, 6] m AND inside the depth FOV (87 deg horizontal, 58 deg vertical). heading = drone's HORIZONTAL
    forward (body-forward camera). NO full-map / global obstacle list -- only what the depth camera sees."""
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    near = np.minimum(np.maximum(p, lo), hi)
    d = near - p; dist = float(np.linalg.norm(d))
    if dist < CAM_RANGE_MIN or dist > CAM_RANGE: return False
    dh = np.array([d[0], d[1], 0.0]); dhn = float(np.linalg.norm(dh))
    if dhn < 1e-6: return True                                    # directly above/below, within range
    if float(dh @ heading) / dhn < CAM_HFOV_COS: return False     # horizontal FOV (87 deg)
    if abs(d[2]) / dhn > CAM_VFOV_TAN: return False               # vertical FOV (58 deg)
    return True


def step_drone(p, v, ng, amax, vmax, dt):
    """Physical 2nd-order tracking: the drone CANNOT teleport to the planner's setpoint -- it accelerates
    toward it with finite force (feedforward planned accel + PD feedback), clamped to a_max (momentum), and
    its speed is capped at ~v_max. Integrate -> the executed path lags/overshoots the plan like a real quad.
    Returns (p_new, v_new, a_cmd)."""
    sp_p = np.asarray(ng.pos, float); sp_v = np.asarray(ng.vel, float); sp_a = np.asarray(ng.accel, float)
    a_cmd = sp_a + KP * (sp_p - p) + KV * (sp_v - v)
    n = float(np.linalg.norm(a_cmd))
    if n > amax: a_cmd = a_cmd * (amax / n)               # physical force limit
    v = v + a_cmd * dt
    sv = float(np.linalg.norm(v))
    if sv > vmax * 1.05: v = v * (vmax * 1.05 / sv)        # soft speed cap (small headroom)
    p = p + v * dt
    return p, v, a_cmd


def box_faces(c, s):
    h = 0.5 * np.asarray(s, float); c = np.asarray(c, float)
    x0, y0, z0 = c - h; x1, y1, z1 = c + h
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[V[i] for i in f] for f in idx]


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def make_scene(seed):
    rng = np.random.default_rng(1000 + seed)
    obs = []
    n_wall = int(rng.integers(round(0.13 * GOAL_X), round(0.22 * GOAL_X))) if HARD else int(rng.integers(2, 5))
    xs = np.sort(rng.uniform(4.0, GOAL_X - 4.0, size=n_wall))
    for cx in xs:
        kind = rng.integers(0, 3)
        cy = float(rng.uniform(-1.6, 1.6))
        if kind == 0:      # vertical pillar
            size = [0.4, 0.4, float(rng.uniform(3.0, 4.5))]; cz = 2.5
        elif kind == 1:    # horizontal bar across y
            size = [0.4, float(rng.uniform(2.0, 3.2)), 0.4]; cz = float(rng.uniform(2.2, 3.4))
        else:              # block
            s = float(rng.uniform(0.6, 1.1)); size = [s, s, s]; cz = float(rng.uniform(2.2, 3.4))
        obs.append({"class": "wall", "size": size, "center": [float(cx), cy, cz]})
    n_hum = int(rng.integers(round(0.04 * GOAL_X), round(0.07 * GOAL_X))) if HARD else int(rng.integers(1, 3))
    vlo, vhi = (0.7, 2.0) if HARD else (0.6, 1.4)
    ystart = 3.0
    for _ in range(n_hum):
        cx = float(rng.uniform(6.0, GOAL_X - 5.0))
        y0 = float(rng.choice([-ystart, ystart]))
        vy = float(rng.uniform(vlo, vhi)) * (-1.0 if y0 > 0 else 1.0)   # walk toward the centre
        # LOOSE timing (not adversarial): cross within a wide window around the drone's ~arrival
        t0 = max(0.0, cx / 5.0 + float(rng.uniform(-5.0, 10.0))) if HARD else 0.0
        cz = 2.8
        obs.append({"class": "human", "size": [0.5, 0.5, 1.6],
                    "traj": [f"{cx}", f"{y0}+{vy}*(t-{t0})", f"{cz}"], "vel": ["0.0", f"{vy}", "0.0"]})
    return obs


def record(seed):
    import _speed_bench as B
    from sando_cpp_bridge import RobotState, SANDO
    par, _, _, _, LOOP = B.build()
    par.v_max = VMAX
    par.use_st_graph = False; par.use_spacetime_corridor = False
    par.force_goal_z = False
    START = np.array([0.0, 0.0, 3.0]); GOAL = np.array([GOAL_X, 0.0, 3.0])
    OBS = make_scene(seed)
    DTS = [B.make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    DR = float(par.drone_radius)
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); cull = float(LOOP.get("sense_cull_r", 40.0))
    AMAX = float(par.a_max)
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); pp = START.copy(); t = 0.0; nr = 0.0; last_rt = 0.0
    reached = False; nstep = 0; frames = []; worst = np.inf
    while t < T_MAX and not reached and nstep < 6000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p.copy(); st.vel = v.copy(); st.accel = a.copy(); sando.update_state(st)
            vh = np.array([v[0], v[1], 0.0]); vhn = float(np.linalg.norm(vh))
            hd = (vh / vhn) if vhn > 0.5 else np.array([1.0, 0.0, 0.0])     # camera = horizontal body-forward
            for j, d in enumerate(DTS):
                if visible(p, hd, d.eval(t), SZ[j]): sando.add_traj(d, t)   # D435i depth FOV only, not the full map
            del vh, vhn
            sando.replan(last_rt, t); nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg:
            if PHYS: p, v, a = step_drone(p, v, ng, AMAX, VMAX, DT)            # physical momentum tracking
            else: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        # densified clearance over the executed sub-step (catch grazes the physical path makes between steps)
        cmin = min(signed_box(pp + s * (p - pp), DTS[j].eval(t), SZ[j]) - DR
                   for j in range(len(OBS)) for s in (0.0, 0.5, 1.0))
        pp = p.copy()
        worst = min(worst, cmin)
        if nstep % FRAMESTEP == 0:
            boxes = [(DTS[j].eval(t).copy(), SZ[j], DTS[j].id) for j in range(len(OBS))]
            frames.append((t, p.copy(), boxes, cmin))
        t += DT; nstep += 1
        if float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    return frames, START, GOAL, reached, t, worst


def render_one(seed):
    frames, START, GOAL, reached, t_end, worst = record(seed)
    tx = [f[1][0] for f in frames]; ty = [f[1][1] for f in frames]; tz = [f[1][2] for f in frames]
    fig = plt.figure(figsize=(18, 4)); ax = fig.add_subplot(111, projection="3d")

    def upd(k):
        ax.cla()
        ax.set_xlim(0, GOAL_X + 2); ax.set_ylim(-4, 4); ax.set_zlim(0.5, 4.5)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        tt, p, boxes, cmin = frames[k]
        wp, hp = [], []
        for c, s, oid in boxes:
            faces = box_faces(c, s)
            (wp if 200 <= oid < 300 else hp).extend(faces)
        if wp: ax.add_collection3d(Poly3DCollection(wp, alpha=0.16, facecolor="0.5", edgecolor="0.35", linewidths=0.2))
        if hp: ax.add_collection3d(Poly3DCollection(hp, alpha=0.30, facecolor="tab:orange", edgecolor="darkorange", linewidths=0.3))
        ax.plot(tx[:k + 1], ty[:k + 1], tz[:k + 1], color="deepskyblue", lw=2)
        ax.scatter([p[0]], [p[1]], [p[2]], color=("red" if cmin < 0 else "yellow"), edgecolor="k", s=55)
        ax.scatter([GOAL[0]], [GOAL[1]], [GOAL[2]], color="red", marker="*", s=110)
        ax.view_init(elev=24, azim=-70)
        ax.set_title(f"scene {seed:02d}  t={tt:4.1f}s  x={p[0]:4.1f}  body_clr={cmin:+.2f}m"
                     f"  {'REACHED' if reached else ''}")
        return []

    ani = anim.FuncAnimation(fig, upd, frames=len(frames), interval=70, blit=False)
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    out = os.path.join(ROOT, "media", f"_rand_{seed:02d}.gif")
    ani.save(out, writer=anim.PillowWriter(fps=FPS)); plt.close(fig)
    return f"scene {seed:02d}: reached={int(reached)} collided={int(worst < 0)} worst_body={worst:+.3f} t={t_end:.1f}s frames={len(frames)}"


def _job(seed):
    try:
        return render_one(seed)
    except Exception as e:
        return f"scene {seed:02d}: ERROR {type(e).__name__}: {e}"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    from concurrent.futures import ProcessPoolExecutor
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for seed, line in zip(range(N), ex.map(_job, range(N))):
            results[seed] = line; print(line, flush=True)
    coll = sum("collided=1" in v for v in results.values())
    reach = sum("reached=1" in v for v in results.values())
    print(f"=== {N} scenes: reached {reach}/{N}, collided {coll}/{N} -> media/_rand_00..{N-1:02d}.gif ===")


if __name__ == "__main__":
    main()
