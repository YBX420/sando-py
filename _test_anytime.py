"""Smoke test for the anytime-feasible ALM: default (no budget) unchanged; with a
time budget the solve truncates, stays bounded, and still returns a CERTIFIED-FEASIBLE
trajectory (anytime_feasible=True) — the core 'computation-invariant safety' property."""
import numpy as np, time
from sando_py.local.obstacles import SphereObstacle
from sando_py.local.avoid_config import default_config
from sando_py.local.local_opt import OptParams, plan_minco, DetourConfig

cfg = default_config()
path = np.array([[0, 0, 2], [3, 0, 2], [6, 0, 2], [9, 0, 2]], float)
# a moving human (HARD) sitting right on the path -> forces the ALM to work
humans = [SphereObstacle(centre0=np.array([4.5, 0.3, 2.0]), radius=0.4,
                         vel=np.array([0.0, 1.0, 0.0]), class_name="human",
                         accel=np.zeros(3))]
opt = OptParams(vmax=5, amax=8)


def run(budget, dc=None):
    t0 = time.perf_counter()
    mj, info = plan_minco(path, humans, cfg, opt_params=opt, detour_cfg=dc, time_budget_ms=budget)
    ms = (time.perf_counter() - t0) * 1000
    return info, ms


def show(tag, info, ms):
    print(f"{tag:22s}: valid={info['trajectory_valid']} anytime_feas={info['anytime_feasible']} "
          f"trunc={info['truncated']} cert_margin={info['certificate_margin']:.3f} ms={ms:5.0f} seeds={info['n_seeds_tried']}")


print("=== anytime-feasible ALM + deterministic topology smoke ===")
show("no-budget (default)", *run(None))
for bud in [80, 40, 20, 8]:
    show(f"budget={bud}ms (detour)", *run(bud))
print("--- deterministic H-signature topology seed (use_topology=True) ---")
topo = DetourConfig(enabled=True, use_topology=True)
for bud in [50, 20, 8]:
    show(f"budget={bud}ms (topology)", *run(bud, topo))
print("\nKEY: whenever valid=True the returned trajectory is the latest CERTIFIED-FEASIBLE")
print("iterate (cert_margin>=0); a deadline only cuts quality/seeds, never the safety margin.")
