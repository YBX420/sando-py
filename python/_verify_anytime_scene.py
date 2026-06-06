"""Task #4 verification: run the Python planner closed-loop on the complex moving scene
(isaac_sando.yaml) with the deterministic certificate-first stack ON vs OFF, and measure:
  - replan ms distribution (p99 / max / #>50ms)  -> is compute bounded <50ms?
  - reached / collided / min clearance           -> is it still safe & does it arrive?
  - truncation-invariance: the per-replan minimum human clearance must stay >= 0 always.
"""
import os, sys, time
import numpy as np, yaml
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
CFG = yaml.safe_load(open("isaac_sando.yaml", encoding="utf-8")); PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]
from sando_py.types import Parameters, RobotState, DynTraj, DroneStatus
from sando_py.planner import SANDO
GR = int(DroneStatus.GOAL_REACHED)
START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)


def build_par(overrides):
    par = Parameters()
    for k, v in PLN.items():
        if k == "sando_map_res": par.res = float(v); continue
        if hasattr(par, k): setattr(par, k, v)
    for k, v in overrides.items():
        setattr(par, k, v)
    return par


def mk(o, i):
    d = DynTraj(); d.id = (200 + i) if o.get("class", "wall") == "wall" else i; d.mode = "Analytic"
    d.bbox = np.array(o["size"], float)
    if "traj" in o:
        d.traj_x, d.traj_y, d.traj_z = [str(e) for e in o["traj"]]
        if "vel" in o: d.traj_vx, d.traj_vy, d.traj_vz = [str(e) for e in o["vel"]]
    else:
        c = o["center"]; d.traj_x, d.traj_y, d.traj_z = f"{c[0]}", f"{c[1]}", f"{c[2]}"; d.traj_vx = d.traj_vy = d.traj_vz = "0.0"
    d.compile_analytic(); return d


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def run(label, overrides):
    par = build_par(overrides)
    if getattr(par, "force_goal_z", False): GOAL[2] = float(par.default_goal_z)
    s = SANDO(par)
    st = RobotState(); st.pos = START.copy(); s.update_state(st)
    s.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); s.set_terminal_goal(G)
    DTS = [mk(o, i) for i, o in enumerate(SCENE["obstacles"])]
    HUM = [o.get("class", "wall") != "wall" for o in SCENE["obstacles"]]
    DT = float(par.dc); RD = float(LOOP["replan_dt"]); TMAX = float(LOOP["t_max"]); CULL = float(LOOP["sense_cull_r"])
    p = START.copy(); v = np.zeros(3); a = np.zeros(3); t = 0.0; nx = 0.0; lrt = 0.0; reached = False
    msrec = []; min_all = min_hum = np.inf
    while t < TMAX and not reached:
        if t >= nx - 1e-9:
            stt = RobotState(); stt.pos = p.copy(); stt.vel = v.copy(); stt.accel = a.copy(); s.update_state(stt)
            for dt_ in DTS:
                if np.linalg.norm(dt_.eval(t) - p) <= CULL: s.add_traj(dt_, t)
            t0 = time.perf_counter(); s.replan(lrt, t); ms = (time.perf_counter() - t0) * 1000; lrt = ms / 1000
            msrec.append(ms); nx = t + RD
        okg, ng = s.get_next_goal()
        if okg: p = np.asarray(ng.pos, float); v = np.asarray(ng.vel, float); a = np.asarray(ng.accel, float)
        for i, o in enumerate(SCENE["obstacles"]):
            d = signed_box(p, DTS[i].eval(t), np.array(o["size"], float))
            min_all = min(min_all, d)
            if HUM[i]: min_hum = min(min_hum, d)
        t += DT
        if s.get_drone_status() == GR and float(np.linalg.norm(p - GOAL)) < float(par.goal_radius): reached = True
    ms = np.array(msrec)
    print(f"[{label}] reached={reached} min_clr_all={min_all:+.3f} min_clr_human={min_hum:+.3f} collided={min_all < 0}")
    print(f"    replan ms: median={np.median(ms):.1f} p90={np.percentile(ms,90):.1f} p99={np.percentile(ms,99):.1f} "
          f"MAX={ms.max():.1f}  #>50ms={(ms>50).sum()}/{len(ms)}")
    return ms, reached, min_hum


print("=== Task #4: deterministic certificate-first stack vs baseline (Python planner) ===")
run("BASELINE (off)", {"v_max": 5.0, "a_max": 8.0})
run("ANYTIME+TOPO 45ms", {"v_max": 5.0, "a_max": 8.0, "minco_time_budget_ms": 45.0, "minco_use_topology": True})
print("\nKEY: gatekeeper guarantees min_clr_human >= 0 (no collision) REGARDLESS of the deadline;")
print("the deadline only bounds replan ms. If #>50ms drops to ~0 and min_clr_human stays >=0, verified.")
