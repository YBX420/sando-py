"""Stage 3 — per-class HARD/ALM performance gates (REAL numbers reported).

  P.0 single per-class replan (human hard) on a modest scene -> report wall time
  P.1 ALM outer iters are bounded (<= cap) on every scene
  P.2 all-hard ablation (walls in ALM too) still bounded + finite
  P.3 the ALM constraint build vectorizes (no per-ctrl-pt python blowup):
      time scales sub-quadratically with M
  P.4 a full detour multi-start per-class replan completes in a few seconds

Thresholds are generous; a regression (not a microbench miss) trips these.
"""
import os
import sys
import time
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams                        # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle          # noqa: E402
from sando_py.local.local_opt import plan_minco, OptParams, DetourConfig   # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


NO_DETOUR = DetourConfig(enabled=False)
CFG = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}

path = np.column_stack([np.linspace(0, 8, 12), np.zeros(12), np.full(12, 2.0)])
mid = path[len(path) // 2].copy()
human = SphereObstacle(mid + np.array([0.0, 0.3, 0.0]), 0.4, class_name="human")
wall = AABBObstacle(mid + np.array([0.0, -0.7, -0.3]),
                    mid + np.array([2.0, -0.3, 0.3]), class_name="wall")


# ---------------------------------------------------------------- P.0 single replan
print("\n--- P.0 single per-class replan (human hard) wall time ---")
opt = OptParams(maxiter=200, kappa=16, vmax=3.0, amax=3.0)
# warm up (import / numpy jit-ish)
plan_minco(path, [human], CFG, opt_params=opt, detour_cfg=NO_DETOUR)
ts = []
for _ in range(5):
    t0 = time.time()
    tr, info = plan_minco(path, [human, wall], CFG, opt_params=opt, detour_cfg=NO_DETOUR)
    ts.append(time.time() - t0)
t_med = float(np.median(ts))
check("P.0a single per-class replan < 2 s (research Python, modest scene)",
      t_med < 2.0,
      f"median={t_med * 1000:.1f}ms over {len(ts)} runs (min={min(ts)*1000:.1f}ms), "
      f"outer_iters={info['alm']['outer_iters']}")
check("P.0b returned finite trajectory + certificate", np.all(np.isfinite(tr.c)))


# ---------------------------------------------------------------- P.1 outer bound
print("\n--- P.1 ALM outer iters bounded on a batch ---")
rng = np.random.default_rng(7)
worst_outer = 0
bad = 0
for _ in range(8):
    off = float(rng.uniform(0.15, 0.6)); r = float(rng.uniform(0.3, 0.55))
    h = SphereObstacle(mid + np.array([0.0, off, 0.0]), r, class_name="human")
    _, inf = plan_minco(path, [h], CFG, opt_params=opt, detour_cfg=NO_DETOUR)
    worst_outer = max(worst_outer, inf["alm"]["outer_iters"])
    if inf["alm"]["outer_iters"] > opt.alm_outer_iters:
        bad += 1
check("P.1 every ALM solve respects the outer-iter cap", bad == 0,
      f"worst outer_iters={worst_outer}, cap={opt.alm_outer_iters}")


# ---------------------------------------------------------------- P.2 all-hard
print("\n--- P.2 all-hard ablation (walls in ALM) bounded + finite ---")
opt_ah = OptParams(maxiter=200, kappa=16, vmax=3.0, amax=3.0, avoid_override="hard")
t0 = time.time()
tr_ah, info_ah = plan_minco(path, [human, wall], CFG, opt_params=opt_ah, detour_cfg=NO_DETOUR)
t_ah = time.time() - t0
check("P.2a all-hard replan < 3 s", t_ah < 3.0,
      f"{t_ah * 1000:.1f}ms, n_hard={info_ah['n_hard']}, outer={info_ah['alm']['outer_iters']}")
check("P.2b all-hard trajectory finite", np.all(np.isfinite(tr_ah.c)))


# ---------------------------------------------------------------- P.3 vectorized scaling
print("\n--- P.3 constraint build vectorizes: time scales sub-quadratically with M ---")
times = {}
for M_cap in (4, 8, 12):
    o = OptParams(maxiter=80, kappa=12, vmax=3.0, amax=3.0, M_cap=M_cap)
    # denser seed path so the seed picks more segments
    dense_path = np.column_stack([np.linspace(0, 8, 2 * M_cap + 2),
                                  np.zeros(2 * M_cap + 2), np.full(2 * M_cap + 2, 2.0)])
    t0 = time.time()
    _, inf = plan_minco(dense_path, [human], CFG, opt_params=o, detour_cfg=NO_DETOUR)
    times[M_cap] = (time.time() - t0, inf["M"])
ratio = times[12][0] / max(times[4][0], 1e-6)
check("P.3 3x more segments costs < 9x time (sub-quadratic, vectorized C2B/D)",
      ratio < 9.0,
      f"t(M~4)={times[4][0]*1000:.0f}ms t(M~12)={times[12][0]*1000:.0f}ms "
      f"ratio={ratio:.1f}x (M={times[4][1]},{times[8][1]},{times[12][1]})")


# ---------------------------------------------------------------- P.4 detour multi-start
print("\n--- P.4 full detour multi-start per-class replan ---")
t0 = time.time()
tr_d, info_d = plan_minco(path, [human], CFG, opt_params=opt)   # detour ENABLED
t_d = time.time() - t0
check("P.4 detour multi-start per-class replan < 6 s", t_d < 6.0,
      f"{t_d * 1000:.1f}ms, seeds={info_d['n_seeds_tried']}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
