"""Stage 3 — per-class HARD(human)/SOFT(wall) behavioural gate (the headline).

Coverage:
  CT.*  CONTINUOUS-TIME guarantee: reproduce M2's soft-only breach (min human
        clearance < d_safe) and show the convex-hull+ALM path achieves DENSELY
        sampled (2000 pt/traj) min clearance >= d_safe with NO sample-gap leak.
  ALM.* mechanics: lambda monotone >= 0, raising rho drives max-violation -> 0,
        bounded outer iters.
  STOP.* a human blocking the only narrow gap -> explicit infeasible / STOP,
        never a false 'valid'.
  ABL.* per-class vs all-soft vs all-hard produce distinct machine-readable
        behaviour (read from the certificate dict).
  CERT.* interpretability certificate dict shape + content.

Run:  python3 test/stage3_minco_perclass.py
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams, resolve_mode          # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.local_opt import (                                     # noqa: E402
    plan_minco, hard_clearance, check_feasibility, _partition_obstacles, OptParams, DetourConfig)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


NO_DETOUR = DetourConfig(enabled=False)
D_SAFE = 0.8


def dense_min_clearance(tr, obs, K=2000):
    """Independent dense 2000-pt continuous-time min signed_dist to `obs`."""
    ts = np.linspace(tr.t_start, tr.t_end, K)
    pts = tr.eval(ts)
    return float(min(obs.signed_dist(p, float(t)) for p, t in zip(pts, ts)))


# ===========================================================================
# CT — the headline: M2 soft-only BREACH vs hard 0-violation, dense 2000 pts
# ===========================================================================
# Canonical M2-trap scene (reproduces the spec's ~0.094 m breach): a straight
# A* path with an off-axis human; the soft cubic penalty's boundary gradient
# vanishes at d_safe so it PLATEAUS just inside d_safe.
straight = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
mid = straight[len(straight) // 2].copy()
human = SphereObstacle(mid + np.array([0.0, 0.25, 0.0]), 0.5, class_name="human")

print("\n--- CT.0 soft-only (the M2 baseline) BREACHES the human ---")
cfg_soft_w = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e3),
              "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
opt_soft = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0, avoid_override="soft")
tr_s, info_s = plan_minco(straight, [human], cfg_soft_w, opt_params=opt_soft,
                          detour_cfg=NO_DETOUR)
clr_soft = dense_min_clearance(tr_s, human, K=2000)
breach_soft = D_SAFE - clr_soft
check("CT.0a soft-only min clearance is INSIDE d_safe (the breach)",
      clr_soft < D_SAFE - 0.02,
      f"dense min_clr={clr_soft:.4f} d_safe={D_SAFE} breach={breach_soft:.4f}")
check("CT.0b soft-only breach reported as clearance_violation (no false 'valid')",
      (not info_s["trajectory_valid"]) and info_s["failure_reason"] == "clearance_violation",
      f"valid={info_s['trajectory_valid']} reason={info_s['failure_reason']}")
check("CT.0c soft-only ALM is OFF (no hard constraints, empty certificate)",
      info_s["n_hard"] == 0 and len(info_s["hard_certificates"]) == 0)

print("\n--- CT.1 HARD (convex-hull + ALM) clears the human at >= d_safe ---")
cfg = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
opt_hard = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0)  # per-class -> human hard
tr_h, info_h = plan_minco(straight, [human], cfg, opt_params=opt_hard,
                          detour_cfg=NO_DETOUR)
clr_hard = dense_min_clearance(tr_h, human, K=2000)
check("CT.1a HARD dense 2000-pt min clearance >= d_safe (continuous-time guarantee)",
      clr_hard >= D_SAFE - 1e-6,
      f"dense min_clr={clr_hard:.4f} d_safe={D_SAFE} margin={clr_hard - D_SAFE:.4f}")
check("CT.1b HARD trajectory is valid (no clearance_violation)",
      info_h["trajectory_valid"] and info_h["failure_reason"] is None,
      f"valid={info_h['trajectory_valid']} reason={info_h['failure_reason']}")
check("CT.1c HARD beats soft-only clearance by a clear margin",
      clr_hard > clr_soft + 0.05, f"hard={clr_hard:.4f} vs soft={clr_soft:.4f}")

print("\n--- CT.2 NO sample-gap leak: analytic certificate margin agrees with dense ---")
# the analytic convex-hull certificate margin (min over ctrl-pts) must be >= 0,
# and the dense (2000-pt) clearance must NOT be below it minus a tiny tol -> proves
# the control polygon clearing implies the continuous curve clears (no gap leak).
cert_margin = info_h["certificate_margin"]
check("CT.2a analytic certificate margin >= 0 (control polygon clears)",
      cert_margin >= -1e-6, f"certificate_margin={cert_margin:.4f}")
check("CT.2b dense clearance >= d_safe AND consistent with certificate (no leak)",
      clr_hard >= D_SAFE - 1e-6 and clr_hard >= D_SAFE + cert_margin - 1e-3,
      f"dense={clr_hard:.4f} d_safe+margin={D_SAFE + cert_margin:.4f}")

# super-dense 5000-pt probe just to be paranoid about between-sample leaks
clr_5k = dense_min_clearance(tr_h, human, K=5000)
check("CT.2c 5000-pt super-dense probe still >= d_safe (paranoid gap check)",
      clr_5k >= D_SAFE - 1e-6, f"5000-pt min_clr={clr_5k:.4f}")


# ===========================================================================
# ALM — multiplier monotonicity, rho drives violation -> 0, bounded outer iters
# ===========================================================================
print("\n--- ALM.* augmented-Lagrangian mechanics ---")
alm = info_h["alm"]
check("ALM.0 lambda_max >= 0 (multipliers non-negative)", alm["lambda_max"] >= 0.0,
      f"lambda_max={alm['lambda_max']:.3f}")
# PHR multipliers are NOT globally monotone in their MAX (a constraint that goes
# from violated to satisfied has its own lambda DECREASE toward its KKT value);
# the true invariant is non-negativity (max(0,.)) at every outer step + bounded.
check("ALM.1 every entry of lambda_history is >= 0 (PHR max(0,.) projection)",
      all(v >= 0.0 for v in alm["lambda_history"]),
      f"lambda_history={['%.2f' % v for v in alm['lambda_history']]}")
check("ALM.2 final max constraint violation ~ 0 (rho drove it down)",
      alm["max_violation"] < 1e-2, f"max_violation={alm['max_violation']:.2e}")
check("ALM.3 outer iters bounded (<= cap)",
      0 < alm["outer_iters"] <= opt_hard.alm_outer_iters,
      f"outer_iters={alm['outer_iters']} cap={opt_hard.alm_outer_iters}")

# explicit rho-drives-violation sweep: smaller rho0 + fewer outers should leave a
# larger residual violation than the full solve (monotone in the penalty budget).
print("\n--- ALM.4 raising the penalty budget drives violation monotonically down ---")
viols = []
for rho0, outers in [(1.0, 1), (5.0, 2), (10.0, 4), (20.0, 8)]:
    o = OptParams(maxiter=300, kappa=16, vmax=3.0, amax=3.0,
                  alm_rho0=rho0, alm_outer_iters=outers)
    _, inf = plan_minco(straight, [human], cfg, opt_params=o, detour_cfg=NO_DETOUR)
    viols.append(inf["alm"]["max_violation"])
check("ALM.4 max_violation non-increasing as (rho0, outer_iters) grow",
      all(b <= a + 5e-3 for a, b in zip(viols, viols[1:])),
      f"violations={['%.2e' % v for v in viols]}")


# ===========================================================================
# STOP — a human blocking the only narrow gap -> explicit infeasible, no false valid
# ===========================================================================
print("\n--- STOP.* human blocking the only gap -> explicit clearance_violation ---")
# narrow corridor: two walls leave a gap of width ~1.2 m, a human (r=0.4, d_safe=0.8
# => needs 1.2 m berth) sits dead-centre. With endpoints pinned on the straight
# line through the gap, NO feasible MINCO can both clear the human by 0.8 AND stay
# in the corridor -> must STOP.
gap_path = np.column_stack([np.linspace(0, 6, 10), np.zeros(10), np.full(10, 1.0)])
block_human = SphereObstacle([3.0, 0.0, 1.0], 0.4, class_name="human")
# tight walls hemming both sides so the optimiser cannot bulge around the human
wallA = AABBObstacle([2.3, -3.0, 0.0], [3.7, -1.05, 2.0], class_name="wall")
wallB = AABBObstacle([2.3, 1.05, 0.0], [3.7, 3.0, 2.0], class_name="wall")
cfg_stop = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
            "wall":  AvoidParams("wall",  "hard", 0.1, 1.0e4)}   # walls HARD too
opt_stop = OptParams(maxiter=200, kappa=16, vmax=3.0, amax=3.0,
                     alm_outer_iters=8, alm_rho0=20.0)
tr_stop, info_stop = plan_minco(gap_path, [block_human, wallA, wallB], cfg_stop,
                                opt_params=opt_stop, detour_cfg=NO_DETOUR)
clr_stop = dense_min_clearance(tr_stop, block_human, K=2000)
check("STOP.0 human clearance is BELOW d_safe (gap truly blocked)",
      clr_stop < D_SAFE, f"dense min human clr={clr_stop:.4f} < d_safe={D_SAFE}")
check("STOP.1 plan_minco surfaces explicit clearance_violation (no false valid)",
      (not info_stop["trajectory_valid"]) and info_stop["failure_reason"] == "clearance_violation",
      f"valid={info_stop['trajectory_valid']} reason={info_stop['failure_reason']}")
check("STOP.2 the violated constraint is the HUMAN (hard_violation flag set)",
      info_stop["hard_violation"] and info_stop["hard_max_breach"] > opt_stop.clearance_tol,
      f"hard_violation={info_stop['hard_violation']} breach={info_stop['hard_max_breach']:.4f}")


# ===========================================================================
# ABL — per-class vs all-soft vs all-hard, distinct & machine-readable
# ===========================================================================
print("\n--- ABL.* ablation switch: per-class / all-soft / all-hard ---")
# scene with a human AND a wall, both near the path
abl_path = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
abl_mid = abl_path[len(abl_path) // 2].copy()
abl_human = SphereObstacle(abl_mid + np.array([0.0, 0.3, 0.0]), 0.4, class_name="human")
abl_wall = AABBObstacle(abl_mid + np.array([0.0, -0.7, -0.3]),
                        abl_mid + np.array([2.0, -0.3, 0.3]), class_name="wall")
abl_obs = [abl_human, abl_wall]
abl_cfg = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
           "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}

def run(override):
    o = OptParams(maxiter=250, kappa=16, vmax=3.0, amax=3.0, avoid_override=override)
    return plan_minco(abl_path, abl_obs, abl_cfg, opt_params=o, detour_cfg=NO_DETOUR)

tr_pc, info_pc = run(None)        # per-class: human hard, wall soft
tr_as, info_as = run("soft")      # all-soft: both soft, ALM off
tr_ah, info_ah = run("hard")      # all-hard: both in ALM (incl. the wall)

check("ABL.0 per-class: human -> hard, wall -> soft (from avoid_modes)",
      info_pc["avoid_modes"]["human"] == "hard" and info_pc["avoid_modes"]["wall"] == "soft",
      f"modes={info_pc['avoid_modes']}")
check("ABL.1 per-class: ALM has exactly the human as hard (n_hard=1, n_soft=1)",
      info_pc["n_hard"] == 1 and info_pc["n_soft"] == 1,
      f"n_hard={info_pc['n_hard']} n_soft={info_pc['n_soft']}")
check("ABL.2 all-soft: ALM OFF (n_hard=0, empty certificate)",
      info_as["n_hard"] == 0 and len(info_as["hard_certificates"]) == 0,
      f"n_hard={info_as['n_hard']} certs={len(info_as['hard_certificates'])}")
check("ABL.3 all-hard: BOTH obstacles in the ALM (n_hard=2, incl. the wall)",
      info_ah["n_hard"] == 2 and any(c["class"] == "wall" for c in info_ah["hard_certificates"]),
      f"n_hard={info_ah['n_hard']} classes={set(c['class'] for c in info_ah['hard_certificates'])}")
# distinct clearance behaviour: per-class & all-hard clear the human; all-soft may not
clr_pc = dense_min_clearance(tr_pc, abl_human, K=2000)
clr_as = dense_min_clearance(tr_as, abl_human, K=2000)
clr_ah = dense_min_clearance(tr_ah, abl_human, K=2000)
check("ABL.4 per-class clears the human >= d_safe (human is hard)",
      clr_pc >= D_SAFE - 1e-6, f"per-class human clr={clr_pc:.4f}")
check("ABL.5 all-hard also clears the human >= d_safe",
      clr_ah >= D_SAFE - 1e-6, f"all-hard human clr={clr_ah:.4f}")
check("ABL.6 the three arms produce DISTINCT human clearances (ablation visible)",
      not (abs(clr_pc - clr_as) < 1e-3 and abs(clr_pc - clr_ah) < 1e-3),
      f"per-class={clr_pc:.4f} all-soft={clr_as:.4f} all-hard={clr_ah:.4f}")
# FAIL-SAFE: unknown class -> HARD
check("ABL.7 fail-safe: unknown class resolves to HARD",
      resolve_mode(SphereObstacle([0, 0, 0], 0.3, class_name="alien"), abl_cfg, None) == "hard")


# ===========================================================================
# CERT — interpretability certificate dict shape + content
# ===========================================================================
print("\n--- CERT.* interpretability certificate records ---")
certs = info_pc["hard_certificates"]
check("CERT.0 one record per (human, segment, ctrl-pt): 6 ctrl-pts * M segs",
      len(certs) == 6 * tr_pc.M, f"n_certs={len(certs)} expected={6 * tr_pc.M}")
rec = certs[0]
need = {"human_idx", "class", "seg", "ctrl_pt", "P", "c_h", "R", "dist",
        "clearance", "g", "lambda", "rho", "active", "force"}
check("CERT.1 record has all drawable keys", not (need - set(rec.keys())),
      f"missing={need - set(rec.keys())}")
check("CERT.2 lambda (shadow price) >= 0 on every record",
      all(c["lambda"] >= 0.0 for c in certs))
check("CERT.3 force = lambda * unit(P-c) is finite", all(np.all(np.isfinite(c["force"])) for c in certs))
check("CERT.4 at least one active certificate on the constrained scene",
      any(c["active"] for c in certs), f"n_active={sum(c['active'] for c in certs)}")
check("CERT.5 continuous_min_clearance reported and >= d_safe (per-class)",
      info_pc["continuous_min_clearance"] >= D_SAFE - 1e-6,
      f"continuous_min_clearance={info_pc['continuous_min_clearance']:.4f}")


# ===========================================================================
# SOFT — a soft obstacle (wall) is MEANT to be grazed: grazing must NOT
# invalidate the trajectory; only HARD obstacles gate clearance validity.
# Regression for the verdict layer: check_feasibility judges every obstacle
# uniformly, so a soft graze trips its clearance_violation; the fixed verdict
# re-derives clearance from the hard set, then falls back to the kinematic checks.
# (The hard-breach direction is covered by STOP.* above.)
# ===========================================================================
print("\n--- SOFT.* soft graze does not invalidate; kinematic fallback intact ---")
soft_path = np.linspace([0.0, 0.0, 1.5], [8.0, 0.0, 1.5], 9)
# Wall face 0.3 m off the path while d_safe=0.4 -> the pinned endpoints already
# sit inside d_safe, so any trajectory grazes it (non-vacuous by construction).
soft_wall = AABBObstacle([-1.0, 0.3, 0.0], [9.0, 3.0, 3.0], class_name="wall")
far_human = SphereObstacle([4.0, -6.0, 1.5], 0.4, class_name="human")
soft_cfg = {"human": AvoidParams("human", "hard", D_SAFE, 1.0e4),
            "wall":  AvoidParams("wall", "soft", 0.4, 1.0e1)}

opt_soft = OptParams(vmax=10.0, amax=10.0)   # kinematics out of the way -> isolate clearance
tr_soft, info_soft = plan_minco(soft_path, [far_human, soft_wall], soft_cfg,
                                opt_params=opt_soft, detour_cfg=NO_DETOUR)
wall_clr = dense_min_clearance(tr_soft, soft_wall, K=2000)
human_clr = dense_min_clearance(tr_soft, far_human, K=2000)
check("SOFT.0 non-vacuous: the soft wall IS grazed (clr < d_safe - tol)",
      wall_clr < 0.4 - opt_soft.clearance_tol, f"min wall clr={wall_clr:.4f}")
check("SOFT.1 human clear (min human clr >= d_safe)",
      human_clr >= D_SAFE - 1e-6, f"min human clr={human_clr:.4f}")
naive_soft = check_feasibility(tr_soft, [far_human, soft_wall], soft_cfg, opt_soft)
check("SOFT.2 naive uniform feasibility WOULD flag the graze (this was the bug)",
      (not naive_soft["trajectory_valid"]) and naive_soft["failure_reason"] == "clearance_violation",
      f"naive valid={naive_soft['trajectory_valid']} reason={naive_soft['failure_reason']}")
check("SOFT.3 FIXED verdict: a soft graze does NOT invalidate (valid, reason None)",
      info_soft["trajectory_valid"] and info_soft["failure_reason"] is None,
      f"valid={info_soft['trajectory_valid']} reason={info_soft['failure_reason']}")
check("SOFT.4 no hard violation (human is the only hard obstacle, and it is clear)",
      info_soft["hard_violation"] is False, f"hard_violation={info_soft['hard_violation']}")

# kinematic fallback: a soft graze AND a genuine velocity violation -> the verdict
# must surface velocity_violation, not the (now-ignored) soft clearance.
opt_v = OptParams(vmax=0.1, amax=10.0, w_vel=0.0)   # vmax tiny + no speed penalty -> clearly exceeds
tr_v, info_v = plan_minco(soft_path, [far_human, soft_wall], soft_cfg,
                          opt_params=opt_v, detour_cfg=NO_DETOUR)
check("SOFT.5 non-vacuous: still grazing AND velocity genuinely exceeded",
      dense_min_clearance(tr_v, soft_wall, K=2000) < 0.4 - opt_v.clearance_tol
      and info_v["feasibility"]["max_v"] > opt_v.vmax * (1.0 + opt_v.vel_tol),
      f"max_v={info_v['feasibility']['max_v']:.3f}")
check("SOFT.6 fallback surfaces velocity_violation (soft graze correctly ignored)",
      (not info_v["trajectory_valid"]) and info_v["failure_reason"] == "velocity_violation",
      f"valid={info_v['trajectory_valid']} reason={info_v['failure_reason']}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
