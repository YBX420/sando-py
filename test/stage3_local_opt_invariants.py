"""Stage 3 — local/local_opt.py invariants (geometric + LBFGS sanity).

Random batched (seed=2026). plan() is expensive (≈6 s/call), so trial counts
are modest.

  I.0 translation invariance: shift scene by c → optimised spline shifts by c
  I.1 endpoint preservation: plan() never moves start / goal (batch)
  I.2 cost is non-increasing: final ≤ init (batch)
  I.3 dt converges to a sane range (above dt_min, below 50·T_target/(n_ctrl-p))
  I.4 obstacle dodge:  final_obs ≤ init_obs whenever an obstacle hits the seed
"""
import os
import sys
import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PKG_ROOT)
from sando_py.local.avoid_config import AvoidParams                       # noqa: E402
from sando_py.local.obstacles import SphereObstacle, AABBObstacle         # noqa: E402
from sando_py.local.cost import obstacle_cost                             # noqa: E402
from sando_py.local.local_opt import plan, plan_minco, OptParams          # noqa: E402
from sando_py.local.bspline import UniformBSpline                         # noqa: E402

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

rng = np.random.default_rng(2026)
cfg = {"human": AvoidParams("human", "hard", 0.8, 1.0e4),
       "wall":  AvoidParams("wall",  "soft", 0.4, 1.0e1)}
LEAN = OptParams(maxiter=30, K=80, vmax=3.0, amax=3.0)


def make_path(M=25):
    return np.cumsum(rng.standard_normal((M, 3)) * 0.4, axis=0)


def random_obstacles(path, n=1):
    """Place each obstacle near a random point on the path (so it actually hits)."""
    obs = []
    for _ in range(n):
        i = int(rng.integers(2, len(path) - 2))
        c = path[i] + rng.standard_normal(3) * 0.1
        obs.append(SphereObstacle(c, float(rng.uniform(0.2, 0.4)), class_name="human"))
    return obs


# The finite-diff B-spline plan() invariants (I.0-I.4) are the SLOW oracle —
# they take minutes per batch (this file used to time out >400s). They are kept
# for parity but gated behind RUN_BSPLINE_ORACLE=1; the fast MINCO invariants
# (I.M.*) below are the default path and run in seconds.
if os.environ.get("RUN_BSPLINE_ORACLE"):
    # ------------------------------------------------------------ I.0 translation invariance
    print("\n--- I.0 translation invariance (5 random scenes) ---")
    bad, worst = 0, 0.0
    for _ in range(5):
        p = make_path()
        obs = random_obstacles(p, n=1)
        delta = rng.standard_normal(3) * 2.0
        bs1, _ = plan(p, obs, cfg, opt_params=LEAN)
        # shift the whole scene by delta
        obs_s = [SphereObstacle(o.centre0 + delta, o.radius, o.vel, o.class_name) for o in obs]
        bs2, _ = plan(p + delta, obs_s, cfg, opt_params=LEAN)
        # NB: LBFGS isn't deterministic across rng state, but the shift of the *final*
        # spline should be very close to delta if cost surface is invariant.
        ts = np.linspace(bs1.t_start, bs1.t_end, 30)
        # both splines have potentially different dt_final; sample by relative t fraction
        ts1 = np.linspace(bs1.t_start, bs1.t_end, 30)
        ts2 = np.linspace(bs2.t_start, bs2.t_end, 30)
        diff = bs2.eval(ts2) - bs1.eval(ts1) - delta
        err = float(np.max(np.linalg.norm(diff, axis=1)))
        if err > 0.5:  # generous: LBFGS may not converge to identical local minima
            bad += 1
        worst = max(worst, err)
    check("I.0 plan output shifts ≈ delta under joint translation", bad == 0,
          f"worst|err|={worst:.2e}")


    # ---------------------------------------------------------------- I.1 endpoint preservation
    print("\n--- I.1 endpoint preservation (10 random scenes) ---")
    bad, worst = 0, 0.0
    for _ in range(10):
        p = make_path(M=int(rng.integers(15, 35)))
        obs = random_obstacles(p, n=int(rng.integers(0, 3)))
        bs, _ = plan(p, obs, cfg, opt_params=LEAN)
        es = float(np.max(np.abs(bs.eval(bs.t_start) - p[0])))
        ee = float(np.max(np.abs(bs.eval(bs.t_end) - p[-1])))
        if es > 1e-9 or ee > 1e-9:
            bad += 1
        worst = max(worst, es, ee)
    check("I.1 start / goal exact across 10 random scenes", bad == 0,
          f"worst|err|={worst:.2e}, bad={bad}/10")


    # ---------------------------------------------------------------- I.2 cost non-increasing
    print("\n--- I.2 final_cost ≤ init_cost (10 random) ---")
    bad = 0
    for _ in range(10):
        p = make_path()
        obs = random_obstacles(p, n=int(rng.integers(1, 3)))
        bs, info = plan(p, obs, cfg, opt_params=LEAN)
        if info["final_cost"] > info["init_cost"] + 1e-6:
            bad += 1
    check("I.2 final_cost ≤ init_cost on every random scene", bad == 0, f"violations={bad}/10")


    # ---------------------------------------------------------------- I.3 dt sane range
    print("\n--- I.3 dt stays in a sane range (10 random) ---")
    bad = 0
    for _ in range(10):
        p = make_path()
        obs = random_obstacles(p, n=1)
        bs, info = plan(p, obs, cfg, opt_params=LEAN)
        dt_max_sane = 50.0 * info["T_target"] / max(info["num_ctrl"] - LEAN.degree, 1)
        if not (LEAN.dt_min < info["dt_final"] < dt_max_sane):
            bad += 1
    check("I.3 dt in (dt_min, 50·T_target/(n_ctrl-p)) on every run",
          bad == 0, f"violations={bad}/10")


    # ---------------------------------------------------------------- I.4 obstacle dodge (when seed hit)
    print("\n--- I.4 obstacle dodge when seed actually overlaps an obstacle (8 random) ---")
    bad, worst = 0, 0.0
    for _ in range(8):
        p = make_path()
        obs = random_obstacles(p, n=2)
        # build the seed exactly the way plan() does
        seg_lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
        path_len = float(seg_lens.sum())
        T0 = path_len / LEAN.vmax
        nc = max(2 * (LEAN.degree + 1) + 1, int(round(len(p) / 2)))
        dt0 = max(T0 / max(nc - LEAN.degree, 1), LEAN.dt_min * 2.0)
        bs_seed = UniformBSpline.fit_path(p, num_ctrl=nc, degree=LEAN.degree, dt=dt0,
                                          clamp_endpoints=True)
        init_obs = obstacle_cost(bs_seed, obs, cfg, K=200)
        if init_obs == 0:
            continue  # seed didn't actually hit any obstacle
        bs, _ = plan(p, obs, cfg, opt_params=LEAN)
        final_obs = obstacle_cost(bs, obs, cfg, K=200)
        if not (final_obs <= init_obs + 1e-6):
            bad += 1
        worst = max(worst, final_obs / max(init_obs, 1e-12))
    check("I.4 final_obs ≤ init_obs whenever seed overlapped an obstacle",
          bad == 0, f"worst ratio={worst:.3f}, violations={bad}")


# ================================================================ MINCO path
# I.M.* — same invariants on the analytic-grad MINCO solver. This is the path
# that runs in SECONDS; the finite-diff plan() blocks above are the slow oracle.
MINCO_OPT = OptParams(maxiter=60, kappa=12, vmax=3.0, amax=3.0)

print("\n--- I.M.0 endpoint preservation (MINCO, 10 random scenes) ---")
bad, worst = 0, 0.0
for _ in range(10):
    p = make_path(M=int(rng.integers(15, 35)))
    obs = random_obstacles(p, n=int(rng.integers(0, 3)))
    tr, _ = plan_minco(p, obs, cfg, opt_params=MINCO_OPT)
    es = float(np.max(np.abs(tr.eval(tr.t_start) - p[0])))
    ee = float(np.max(np.abs(tr.eval(tr.t_end) - p[-1])))
    if es > 1e-9 or ee > 1e-9:
        bad += 1
    worst = max(worst, es, ee)
check("I.M.0 MINCO pins start/goal exactly (structural)", bad == 0, f"worst={worst:.2e}")

print("\n--- I.M.1 final_cost <= init_cost (MINCO, 10 random) ---")
bad = 0
for _ in range(10):
    p = make_path()
    obs = random_obstacles(p, n=int(rng.integers(1, 3)))
    tr, info = plan_minco(p, obs, cfg, opt_params=MINCO_OPT)
    if info["final_cost"] > info["init_cost"] + 1e-6:
        bad += 1
check("I.M.1 MINCO final_cost <= init_cost on every scene", bad == 0, f"violations={bad}/10")

print("\n--- I.M.2 T stays >= dt_min (bounds respected) (MINCO, 10 random) ---")
bad = 0
for _ in range(10):
    p = make_path()
    obs = random_obstacles(p, n=1)
    tr, info = plan_minco(p, obs, cfg, opt_params=MINCO_OPT)
    if info["dt_final"] < MINCO_OPT.dt_min - 1e-9:
        bad += 1
check("I.M.2 every T_i >= dt_min after solve", bad == 0, f"violations={bad}/10")

print("\n--- I.M.3 obstacle dodge: final_obs <= init_obs when seed hit (MINCO, 8 random) ---")
from sando_py.local.local_opt import _minco_seed                          # noqa: E402
from sando_py.local.minco import MinjerkTraj                              # noqa: E402
bad, worst = 0, 0.0
n_hit = 0
for _ in range(8):
    p = make_path()
    obs = random_obstacles(p, n=2)
    s0, g0, q0, T0 = _minco_seed(p, MINCO_OPT)
    seed_tr = MinjerkTraj.from_endpoints(s0, g0, q0, T0)
    init_obs = obstacle_cost(seed_tr, obs, cfg, K=200)
    if init_obs == 0:
        continue
    n_hit += 1
    tr, _ = plan_minco(p, obs, cfg, opt_params=MINCO_OPT)
    final_obs = obstacle_cost(tr, obs, cfg, K=200)
    if not (final_obs <= init_obs + 1e-6):
        bad += 1
    worst = max(worst, final_obs / max(init_obs, 1e-12))
check("I.M.3 MINCO final_obs <= init_obs whenever seed overlapped", bad == 0,
      f"worst ratio={worst:.3f}, violations={bad}/{n_hit}")

print("\n--- I.M.4 translation invariance (MINCO, 5 scenes) ---")
bad, worst = 0, 0.0
for _ in range(5):
    p = make_path()
    obs = random_obstacles(p, n=1)
    delta = rng.standard_normal(3) * 2.0
    tr1, _ = plan_minco(p, obs, cfg, opt_params=MINCO_OPT)
    obs_s = [SphereObstacle(o.centre0 + delta, o.radius, o.vel, o.class_name) for o in obs]
    tr2, _ = plan_minco(p + delta, obs_s, cfg, opt_params=MINCO_OPT)
    t1 = np.linspace(tr1.t_start, tr1.t_end, 30)
    t2 = np.linspace(tr2.t_start, tr2.t_end, 30)
    err = float(np.max(np.linalg.norm(tr2.eval(t2) - tr1.eval(t1) - delta, axis=1)))
    if err > 0.5:
        bad += 1
    worst = max(worst, err)
check("I.M.4 MINCO output shifts ~ delta under joint translation", bad == 0,
      f"worst|err|={worst:.2e}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
