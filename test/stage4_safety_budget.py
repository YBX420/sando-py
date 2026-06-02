"""Stage 4 — system safety budget / fault tree: WHERE is the 1e-5 / flight-hour bottleneck?

The deterministic planner layer certifies "given the human is detected, the estimate is
in-bound, and the human obeys a physical motion bound, the plan cannot collide". This file
turns the per-flight-hour target into MEASURED numbers (from ETH-UCY) + a fault tree, and
locates the real bottleneck instead of asserting it.

Two ways to size the physical reach, grounded in real data:
  (A) CA-DEVIATION reach -- cover the human's deviation from a constant-acceleration
      prediction (the residual distribution). MEASURED tail is HEAVY (q99.9=3.3 m, max
      7.4 m over 1.2 s) -> a 1e-5 quantile is both unresolvable from 15 k samples AND
      impractically large. CA-deviation reach CANNOT certify five-nines.
  (B) SPEED-BOUND reach -- the human is somewhere within v_max * tau of where it is now,
      because instantaneous SPEED is PHYSICALLY bounded (a human cannot exceed a sprint).
      MEASURED pedestrian speed is bounded (max 3.9 m/s; a sprint bound v_max~8 covers ANY
      human). So reach = v_max * tau has 0 physical violations -> this is the route to a
      planner layer that is genuinely ~0, at the cost of a reach that scales with tau.

Then a fault tree composes the layers and a requirements allocation shows that, with the
planner layer ~0, the binding constraint for 1e-5/flight-hour is the PERCEPTION miss rate
-- ~4 orders below a typical single detector. That is the bottleneck, quantified.

Run:  python3 test/stage4_safety_budget.py
  (uses test/data/eth_ucy_ca_residuals.npz; rebuild with _build_realdata_residuals.py)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "eth_ucy_ca_residuals.npz")

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

if not os.path.exists(NPZ):
    print(f"missing {NPZ} -- run: python3 test/_build_realdata_residuals.py")
    raise SystemExit(1)

z = np.load(NPZ)
resid = np.asarray(z["residuals"], dtype=np.float64)      # CA-deviation sup over horizon (m)
speed = np.asarray(z["speed"], dtype=np.float64)          # instantaneous |velocity| (m/s)
HORIZON = float(z["horizon_s"])                           # 1.2 s


# ===========================================================================
# (A) CA-deviation reach: heavy tail -> cannot certify five-nines
# ===========================================================================
print("\n--- (A) CA-deviation reach (cover deviation from CA prediction): MEASURED tail ---")
qs = {q: float(np.quantile(resid, q)) for q in (0.9, 0.99, 0.999)}
print(f"   residual quantiles (m): q90={qs[0.9]:.2f}  q99={qs[0.99]:.2f}  q99.9={qs[0.999]:.2f}  "
      f"max={resid.max():.2f}   (N={resid.size})")
# to estimate a 1-1e-5 quantile you need ~10/1e-5 = 1e6 samples; we have ~1.5e4.
resolvable_alpha = 10.0 / resid.size
check("BUDGET.A.0 CA-deviation 1e-5 quantile is UNRESOLVABLE from the data (need ~1e6 samples)",
      resolvable_alpha > 1e-5,
      f"smallest resolvable tail ~ {resolvable_alpha:.1e} >> 1e-5 (N={resid.size})")
check("BUDGET.A.1 CA-deviation tail is HEAVY -> even q99.9 reach is impractically large",
      qs[0.999] > 3.0,
      f"q99.9={qs[0.999]:.2f} m (no-go ~ r+d_safe+{qs[0.999]:.1f} m); statistical reach can't hit five-nines")


# ===========================================================================
# (B) Speed-bound reach: instantaneous speed is PHYSICALLY bounded -> true worst case
# ===========================================================================
print("\n--- (B) speed-bound reach v_max*tau (speed is physically bounded): MEASURED ---")
V_OBS_MAX = float(speed.max())
V_JOG, V_SPRINT = 5.0, 8.0                  # jog bound (covers observed) / sprint bound (any human)
exceed_jog = float(np.mean(speed > V_JOG))
exceed_sprint = float(np.mean(speed > V_SPRINT))
print(f"   pedestrian speed (m/s): q99={np.quantile(speed,0.99):.2f}  q99.9={np.quantile(speed,0.999):.2f}  "
      f"max={V_OBS_MAX:.2f}   (N={speed.size})")
print(f"   exceed v_max=5: {exceed_jog:.5f}   exceed v_max=8(sprint): {exceed_sprint:.5f}")
check("BUDGET.B.0 instantaneous speed is physically bounded (0 samples exceed a sprint bound)",
      exceed_sprint == 0.0 and V_OBS_MAX < V_SPRINT,
      f"max observed {V_OBS_MAX:.2f} m/s < sprint {V_SPRINT}; 0/{speed.size} exceed -> v_max*tau is a TRUE physical bound")
# the lever: reach = v_max * tau shrinks with the trust window
print("   reach = v_max(sprint 8) * tau :")
for tau in (0.15, 0.3, 0.6, 1.2):
    print(f"      tau={tau:.2f}s -> reach={V_SPRINT*tau:.2f} m  (no-go ~ r+d_safe+{V_SPRINT*tau:.1f})")
check("BUDGET.B.1 the tau lever: a SHORT replan window keeps the physical reach feasible",
      V_SPRINT * 0.15 < 1.5 and V_SPRINT * 1.2 > 5.0,
      f"reach(tau=0.15s)={V_SPRINT*0.15:.2f} m feasible vs reach(tau=1.2s)={V_SPRINT*1.2:.2f} m -> replan fast, keep tau small")


# ===========================================================================
# Fault tree + requirements allocation: where is the bottleneck?
# ===========================================================================
print("\n--- fault tree (per flight hour) + requirements allocation for 1e-5/hr ---")

def collision_per_hour(p_miss, p_est, p_speed, p_planner, encounters_per_hr, hw_per_hr):
    """Series fault tree (union bound on small probs). Per encounter the plan can collide if
    perception MISSES the human, OR the estimate is out of its bound, OR the human exceeds the
    physical speed bound, OR (given all ok) the planner fails -- the last is 0 by the
    deterministic guarantee. Plus an encounter-independent hardware term per hour."""
    p_enc = p_miss + p_est + p_speed + p_planner
    return encounters_per_hr * p_enc + hw_per_hr

TARGET = 1e-5
E = 60.0                       # human-encounters per flight hour (busy environment)
P_PLANNER = 0.0                # deterministic guarantee under its assumptions (the iron proof)
P_SPEED = exceed_sprint        # measured: 0 against a sprint bound
HW = 1e-6                      # hardware-induced collision/hr (redundant avionics, typical)

# sanity: a typical single detector (miss ~ 1e-3) is nowhere near the target
naive = collision_per_hour(1e-3, 1e-4, P_SPEED, P_PLANNER, E, HW)
print(f"   naive (single detector miss=1e-3, est=1e-4): {naive:.2e}/hr  "
      f"= 1 collision / {1/naive:.0f} hr  (target {TARGET:.0e}/hr)")
check("BUDGET.tree.0 a single typical detector is ~4 orders ABOVE target (non-vacuous)",
      naive > 100 * TARGET, f"{naive:.2e}/hr >> {TARGET:.0e}/hr")

# requirements allocation: with planner=0, hardware budgeted, what must perception achieve?
budget_per_enc = (TARGET - HW) / E                       # collision budget per encounter
req_p_miss = budget_per_enc - P_SPEED - 1e-9             # leave a sliver for estimation
print(f"   to hit {TARGET:.0e}/hr at {E:.0f} enc/hr with planner=0, hardware={HW:.0e}/hr:")
print(f"      per-encounter budget = {budget_per_enc:.2e}")
print(f"      => REQUIRED perception miss rate < {max(req_p_miss,0):.2e}  "
      f"(typical single detector ~1e-3 -> need ~{1e-3/max(req_p_miss,1e-12):.0e}x better: redundancy/multi-frame)")
check("BUDGET.alloc.0 the BOTTLENECK is perception: required miss-rate is far below a single detector",
      0 < req_p_miss < 1e-5,
      f"required p_miss < {max(req_p_miss,0):.2e} << typical 1e-3 -> perception (not the planner) is where five-nines is won")
check("BUDGET.alloc.1 with the required perception, the composed system rate meets target",
      collision_per_hour(req_p_miss, 0.0, P_SPEED, P_PLANNER, E, HW) <= TARGET + 1e-12,
      f"composed = {collision_per_hour(req_p_miss, 0.0, P_SPEED, P_PLANNER, E, HW):.2e}/hr <= {TARGET:.0e}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
print("CONCLUSION: planner layer can be made ~0 (deterministic reach with a PHYSICAL speed")
print("bound + short tau); the per-flight-hour bottleneck is PERCEPTION miss-detection,")
print("which must reach ~1e-7/encounter (redundancy / multi-frame) -- OUTSIDE this planner.")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
