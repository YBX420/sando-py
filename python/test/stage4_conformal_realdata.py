"""Stage 4 — conformal Q_alpha recalibrated on REAL pedestrian data (ETH-UCY),
and the measurable cost of the distribution gap.

Closes the honesty boundary from the synthetic coverage probe: Q_alpha was calibrated
on a SYNTHETIC error model (e ~ t^3 jerk). Split conformal only guarantees coverage
RELATIVE to the calibration distribution, so the synthetic 0.90 says nothing about
real humans. Here we calibrate on REAL CA-prediction residuals (test/data/
eth_ucy_ca_residuals.npz, built by _build_realdata_residuals.py from ETH-UCY) and
MEASURE what the distribution gap costs.

The residual R_j = sup_{s in [0,H]} ||c_true(s) - c_pred(s)|| is the real over-horizon
deviation of a pedestrian from a causal constant-acceleration extrapolation -- exactly
what Q_alpha must cover.

Checks:
  REAL.0  split conformal calibrated AND tested on REAL data hits target coverage
          (the METHOD is sound on real residuals, not just synthetic).
  REAL.1  the real-data Q_alpha (grounded, in metres over the horizon) is reported --
          this is the number a deployed certificate should actually use.
  SHIFT.synth  calibrating Q on SYNTHETIC and deploying on REAL breaks coverage
          (under- or over-covers): the synthetic 0.90 does NOT transfer. MEASURED.
  SHIFT.scene  even REAL->REAL shift bites: calibrate on a calm scene (zara), deploy
          on a busy one (eth) -> coverage collapses. Exchangeability is load-bearing.

Run:  python3 test/stage4_conformal_realdata.py
  (needs test/data/eth_ucy_ca_residuals.npz; rebuild with _build_realdata_residuals.py)
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

ALPHA = 0.1
target = 1.0 - ALPHA
rng = np.random.default_rng(2026)

z = np.load(NPZ)
real = np.asarray(z["residuals"], dtype=np.float64)      # (Nreal,) metres, sup over horizon
H = int(z["H"]); horizon_s = float(z["horizon_s"])
print(f"\nreal residuals: N={real.size}  horizon={horizon_s:.1f}s ({H} steps)  "
      f"median={np.median(real):.3f}m  q90={np.quantile(real, 0.9):.3f}m  max={real.max():.3f}m")


def split_conformal_quantile(cal):
    """Finite-sample conservative (1-alpha) quantile index ceil((n+1)(1-a))/n."""
    n = cal.size
    lvl = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    return float(np.quantile(cal, lvl, method="higher"))


def avg_coverage(cal_pool, test_pool, n_cal, REPEAT=200):
    """Mean held-out coverage of test_pool under Q calibrated on a fresh n_cal draw
    of cal_pool, averaged over REPEAT splits. Returns (mean_cov, mean_Q)."""
    covs, Qs = [], []
    for _ in range(REPEAT):
        cal = rng.choice(cal_pool, size=n_cal, replace=False) if n_cal <= cal_pool.size \
            else rng.choice(cal_pool, size=n_cal, replace=True)
        Q = split_conformal_quantile(cal)
        covs.append(float(np.mean(test_pool <= Q)))
        Qs.append(Q)
    return float(np.mean(covs)), float(np.mean(Qs))


# --- REAL: calibrate AND test on real (disjoint draws from the real pool) ---------
# Split the real pool into two disjoint halves so cal and test never share a window.
perm = rng.permutation(real.size)
real_cal_pool = real[perm[: real.size // 2]]
real_test_pool = real[perm[real.size // 2:]]
n_cal = 1000
cov_real, Q_real = avg_coverage(real_cal_pool, real_test_pool, n_cal)
print(f"\nREAL  calibrate+test on real: held-out coverage = {cov_real:.4f}  Q_real = {Q_real:.3f} m")
check("REAL.0 conformal on REAL pedestrian residuals hits target coverage",
      cov_real >= target - 0.01, f"held-out {cov_real:.4f} >= {target:.2f} (cal+test both real, disjoint)")
check("REAL.1 grounded real-data Q_alpha reported (the number a deployed cert needs)",
      0.3 < Q_real < 3.0, f"Q_real = {Q_real:.3f} m over {horizon_s:.1f}s  (vs synthetic placeholder)")


# --- synthetic residuals over the SAME horizon (e ~ t^3 jerk, the old model) -------
def synth_residuals(n):
    ts = np.linspace(0.0, horizon_s, 41)
    j = rng.normal(0.0, 1.5, (n, 3))
    e = (1.0 / 6.0) * j[:, None, :] * (ts[None, :, None] ** 3) + rng.normal(0.0, 0.02, (n, 41, 3))
    return np.linalg.norm(e, axis=2).max(axis=1)
synth = synth_residuals(20000)
Q_synth = split_conformal_quantile(synth)
print(f"synthetic (e~t^3) over {horizon_s:.1f}s: Q_synth = {Q_synth:.3f} m   "
      f"(real Q = {Q_real:.3f} m -> synthetic {'UNDER' if Q_synth < Q_real else 'OVER'}-estimates by "
      f"{abs(Q_synth - Q_real):.3f} m)")


# --- SHIFT.synth: calibrate on synthetic, DEPLOY on real -> coverage breaks --------
cov_real_under_synth = float(np.mean(real_test_pool <= Q_synth))
print(f"\nSHIFT  Q calibrated on SYNTHETIC, deployed on REAL: coverage = {cov_real_under_synth:.4f} "
      f"(target {target:.2f})")
check("SHIFT.synth synthetic-calibrated Q does NOT hold on real (distribution gap is MEASURED)",
      abs(cov_real_under_synth - target) > 0.03,
      f"real coverage under synthetic Q = {cov_real_under_synth:.4f} != target {target:.2f} "
      f"({'UNDER-covers -> UNSAFE' if cov_real_under_synth < target else 'over-covers -> wasteful'})")


# --- SHIFT.scene: GENUINE real->real shift (calibrate on calm zara, deploy on busy eth) ---
# zara01 (parked-car promenade, calm, small CA error) vs eth (busy entrance, large error)
# are DIFFERENT scenes, not a circular magnitude split. Calibrating in one and flying in
# the other is the deployment reality, and it breaks the guarantee.
zara = np.asarray(z["res_zara01"], dtype=np.float64)
eth = np.asarray(z["res_eth"], dtype=np.float64)
Q_zara = split_conformal_quantile(zara)                      # calm-scene calibration
cov_eth_under_zara = float(np.mean(eth <= Q_zara))
print(f"\nSHIFT  Q from CALM scene (zara, q90={np.quantile(zara,0.9):.2f}m) deployed on BUSY scene "
      f"(eth, q90={np.quantile(eth,0.9):.2f}m): coverage = {cov_eth_under_zara:.4f}")
check("SHIFT.scene genuine real->real scene shift (zara->eth) collapses coverage",
      cov_eth_under_zara < target - 0.05,
      f"eth coverage under zara-calibrated Q={Q_zara:.3f}m is {cov_eth_under_zara:.4f} << {target:.2f}")


# --- ROBUST: the cheap, safe fix for the 0.53 collapse = calibrate on the WORST scene -
# Don't reach for context-conditioning Q(x) yet (gradient no longer free). The conservative
# constant fix -- same principle as a conservative constant epsilon_track -- is to calibrate
# on the MOST crowded scene (eth, q90=1.20m). That Q is conservative for ANY deployment no
# more extreme than eth, so coverage on the calmer scenes stays >= target (it over-covers).
hotel = np.asarray(z["res_hotel"], dtype=np.float64)
Q_eth = split_conformal_quantile(eth)                        # calibrate on the WORST scene
cov_zara_under_eth = float(np.mean(zara <= Q_eth))
cov_hotel_under_eth = float(np.mean(hotel <= Q_eth))
cov_pool_under_eth = float(np.mean(real <= Q_eth))
print(f"\nROBUST worst-scene calibration: Q_eth={Q_eth:.3f}m -> coverage  zara={cov_zara_under_eth:.4f}  "
      f"hotel={cov_hotel_under_eth:.4f}  pooled={cov_pool_under_eth:.4f}")
check("ROBUST.0 worst-scene (eth) calibration is conservative >= target on all calmer scenes",
      min(cov_zara_under_eth, cov_hotel_under_eth, cov_pool_under_eth) >= target,
      f"min coverage over {{zara,hotel,pooled}} = "
      f"{min(cov_zara_under_eth, cov_hotel_under_eth, cov_pool_under_eth):.4f} >= {target:.2f} "
      f"(vs the 0.53 collapse the OTHER way) -- bank 0.53 for the paper, ship worst-scene Q")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
