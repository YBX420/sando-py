"""持续测速 bench：真 C++ 管线（sando_cpp_bridge）驱动同一套 SANDO 编排器跑闭环，
障碍按 DynTraj 解析式运动（= Isaac 背后那套物理，只是不渲染）。每轮重读 isaac_sando.yaml
（可边跑边调参），算【真实平均速度 avg = 执行 setpoint 弧长 / 飞行时间】，记到 _speed_bench.log。
跑:  python _speed_bench.py          # 不停循环（后台）
     python _speed_bench.py 1        # 只跑 1 轮（取基线）
不改场景：只读 scene、只调 planner 参数。"""
import os, sys, time
import numpy as np, yaml

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from sando_cpp_bridge import (Parameters, RobotState, DynTraj, SANDO,
                              DroneStatus_GOAL_REACHED as GOAL_REACHED)


def build():
    CFG = yaml.safe_load(open(os.path.join(ROOT, "isaac_sando.yaml"), encoding="utf-8"))
    PLN, SCENE, LOOP = CFG["planner"], CFG["scene"], CFG["loop"]
    par = Parameters()
    for k, v in PLN.items():
        if k == "sando_map_res":
            par.res = float(v); continue
        if hasattr(par, k):
            setattr(par, k, v)
    # env overrides for fast sweeps (do NOT touch the scene): BUDGET/VMAX/SFC/WCOR/AMAX/WTIME
    _ov = {"minco_time_budget_ms": "BUDGET", "v_max": "VMAX", "minco_sfc_radius": "SFC",
           "minco_w_corridor": "WCOR", "a_max": "AMAX", "minco_w_time": "WTIME",
           "minco_w_vel": "WVEL", "minco_w_accel": "WACCEL",
           "inflation_hgp": "INFL", "dyn_base_inflation_m": "DYNINFL",
           "y_min": "YMIN", "y_max": "YMAX", "heat_weight": "HEATW"}
    for field, ev in _ov.items():
        if os.environ.get(ev):
            setattr(par, field, float(os.environ[ev]))
    if os.environ.get("RETIME"):
        setattr(par, "minco_retime_overshoot", os.environ["RETIME"] == "1")
    if os.environ.get("FREEGOAL"):
        setattr(par, "use_free_goal", os.environ["FREEGOAL"] == "1")
    START = np.array(SCENE["start"], float); GOAL = np.array(SCENE["goal"], float)
    if bool(getattr(par, "force_goal_z", False)):
        GOAL[2] = float(par.default_goal_z)
    return par, START, GOAL, SCENE["obstacles"], LOOP


def make_dt(i, o):
    dt = DynTraj(); dt.id = (200 + i) if o.get("class", "wall") == "wall" else i
    dt.mode = "Analytic"; dt.bbox = np.array(o["size"], float)
    if "traj" in o:
        dt.traj_x, dt.traj_y, dt.traj_z = [str(e) for e in o["traj"]]
        if "vel" in o:
            dt.traj_vx, dt.traj_vy, dt.traj_vz = [str(e) for e in o["vel"]]
    else:
        c = o["center"]
        dt.traj_x, dt.traj_y, dt.traj_z = f"{c[0]}", f"{c[1]}", f"{c[2]}"
        dt.traj_vx = dt.traj_vy = dt.traj_vz = "0.0"
    dt.compile_analytic(); return dt


def signed_box(p, c, sz):
    lo = c - 0.5 * sz; hi = c + 0.5 * sz
    out = np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0)
    return float(np.linalg.norm(out)) if np.any(out > 0) else float(np.max(np.maximum(lo - p, p - hi)))


def episode():
    par, START, GOAL, OBS, LOOP = build()
    DTS = [make_dt(i, o) for i, o in enumerate(OBS)]
    SZ = [np.array(o["size"], float) for o in OBS]
    sando = SANDO(par)
    st = RobotState(); st.pos = START.copy(); sando.update_state(st)
    sando.update_occupancy_map_ptr(np.zeros((0, 3)))
    G = RobotState(); G.pos = GOAL.copy(); sando.set_terminal_goal(G)

    DT = float(par.dc); RD = float(LOOP["replan_dt"]); T_MAX = float(LOOP["t_max"])
    cull = float(LOOP.get("sense_cull_r", 40.0))
    p_d = START.copy(); v_d = np.zeros(3); a_d = np.zeros(3)
    t = 0.0; last_rt = 0.0; nr = 0.0; reached = False; nstep = 0
    arc = 0.0; min_all = np.inf; collided = False; ms_list = []; t_reach = T_MAX
    stuck_x = None; stall_t = 0.0; ninv = 0
    cruise_arc = 0.0; cruise_t = 0.0
    DIAG = os.environ.get("DIAG", "0") == "1"
    while t < T_MAX and not reached and nstep < 8000:
        if t >= nr - 1e-9:
            st = RobotState(); st.pos = p_d.copy(); st.vel = v_d.copy(); st.accel = a_d.copy()
            sando.update_state(st)
            for j, dt in enumerate(DTS):
                if np.linalg.norm(dt.eval(t) - p_d) <= cull:
                    sando.add_traj(dt, t)
            a = time.perf_counter(); ret = sando.replan(last_rt, t); last_rt = time.perf_counter() - a
            ms_list.append(last_rt * 1000.0)
            okr = bool(ret[0] if isinstance(ret, tuple) else ret)
            if not okr:
                ninv += 1
            if DIAG and nstep % 25 < 5:
                print("    [replan x=%.1f ok=%d ms=%.0f]" % (p_d[0], int(okr), last_rt * 1000.0), flush=True)
            nr = t + RD
        okg, ng = sando.get_next_goal()
        if okg:
            npos = np.asarray(ng.pos, float)
            step = float(np.linalg.norm(npos - p_d))
            arc += step
            # sustained-cruise window: steady flight through the clutter, excluding the one-time
            # accel-from-rest (x<8) and the final decel-to-goal (x>56). This is the speed the drone
            # holds in continuous (SANDO-pattern) operation; the rest-to-rest total avg under-counts it.
            if 8.0 <= p_d[0] <= 56.0:
                cruise_arc += step
                cruise_t += DT
            p_d = npos; v_d = np.asarray(ng.vel, float); a_d = np.asarray(ng.accel, float)
        cmin = np.inf
        for j in range(len(OBS)):
            d = signed_box(p_d, DTS[j].eval(t), SZ[j])
            if d < cmin:
                cmin = d
        min_all = min(min_all, cmin)
        if cmin < 0.0:
            collided = True
        # track where it stalls (forward speed < 1 m/s for a while)
        if stuck_x is None or p_d[0] > stuck_x + 0.5:
            stuck_x = p_d[0]; stall_t = t
        if DIAG and nstep % 25 == 0:
            print("  t=%5.2f x=%5.1f |v|=%5.2f clr=%+5.2f ms=%4.0f"
                  % (t, p_d[0], float(np.linalg.norm(v_d)), cmin, (ms_list[-1] if ms_list else 0)), flush=True)
        t += DT; nstep += 1
        if sando.get_drone_status() == GOAL_REACHED and float(np.linalg.norm(p_d - GOAL)) < float(par.goal_radius):
            reached = True; t_reach = t
    flight_t = t_reach if reached else t
    avg = arc / max(flight_t, 1e-6)
    cruise_avg = cruise_arc / max(cruise_t, 1e-6)
    ms = np.array(ms_list) if ms_list else np.array([0.0])
    return dict(avg=avg, arc=arc, flight_t=flight_t, reached=reached, final_x=float(p_d[0]),
                min_clear=min_all, collided=collided, ms_mean=float(ms.mean()),
                ms_p99=float(np.percentile(ms, 99)), ms_max=float(ms.max()),
                vmax=float(par.v_max), sfc=float(getattr(par, "minco_sfc_radius", 0.0)),
                wcor=float(getattr(par, "minco_w_corridor", 0.0)),
                stuck_x=(stuck_x if not reached else None), stall_t=stall_t,
                ninv=ninv, nrep=len(ms_list), cruise_avg=cruise_avg)


def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = forever
    logp = os.path.join(ROOT, "_speed_bench.log")
    n = 0
    while True:
        n += 1
        try:
            r = episode()
        except Exception as e:
            line = f"ep={n} ERROR {type(e).__name__}: {e}"
        else:
            stuck = f" STUCK@x={r['stuck_x']:.1f}(t={r['stall_t']:.1f})" if r["stuck_x"] is not None else ""
            line = ("ep=%d AVG=%.2f CRUISE=%.2f m/s  arc=%.1f t=%.2fs reached=%d final_x=%.1f minclr=%+.2f coll=%d"
                    " | vmax=%.0f sfc=%.2f wcor=%.0f | ms mean=%.0f p99=%.0f max=%.0f%s"
                    % (n, r["avg"], r["cruise_avg"], r["arc"], r["flight_t"], r["reached"], r["final_x"], r["min_clear"],
                       r["collided"], r["vmax"], r["sfc"], r["wcor"], r["ms_mean"], r["ms_p99"], r["ms_max"], stuck)
                    + " inv=%d/%d" % (r["ninv"], r["nrep"]))
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if n_iter and n >= n_iter:
            break


if __name__ == "__main__":
    main()
