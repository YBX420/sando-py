"""Stage 4 — conformal continuous-time safety certificate: coverage experiment.

Framework validation behind the certificate novelty upgrade (R = r + d_safe + Q_alpha).
This is the SYNTHETIC evidence that lifting per-timestep coverage to OVER-INTERVAL
coverage is a real algorithmic reduction, not a relabelled inflation formula.

Theorem (proven separately): the relative-trajectory certificate guarantees
||P(s) - c_pred(s)|| >= R.  With prediction error e(s) = c_true(s) - c_pred(s),
the triangle inequality gives ||P(s) - c_true(s)|| >= R - ||e(s)||.  Conformal
calibration turns each calibration trajectory into ONE scalar nonconformity score
R_j = sup_s ||e_j(s)|| (the whole-horizon max error) and takes the (1-alpha)
quantile Q_alpha.  Then R = r + d_safe + Q_alpha => Pr(no collision over the whole
interval) >= 1 - alpha.

Coverage:
  COV.*   trajectory-conformal (sup-norm score) hits the target marginal coverage
          1-alpha, AVERAGED over many calibration splits (the guarantee is marginal
          over the calibration draw, so it fluctuates per split but the mean holds).
  TIGHT.* the sup-norm score is strictly TIGHTER than a naive per-timestep union
          bound (alpha/N each) -- same coverage for a much smaller safety margin Q.
          This is the "measurable consequence": fly-able margin, not话术.
  MARG.*  honest demonstration that a SINGLE split is marginal (dips below target),
          while the MEAN meets it -- so the guarantee must be read over the draw.

WHAT THIS NUMBER DOES AND DOES NOT PROVE (read before quoting 0.90):
  - It proves the conformal MACHINERY is correct and achieves the target coverage on
    its OWN distribution. Split conformal's guarantee is RELATIVE to the calibration
    distribution: held-out coverage >= 1-alpha holds because test and calibration are
    EXCHANGEABLE here (both drawn from the same synthetic error model). The METHOD is
    sound and strictly tighter than the union bound. That is the contribution validated.
  - It does NOT prove "safe around real humans". Q_alpha here is calibrated on a SYNTHETIC
    error model (CA predictor + unmodelled jerk, e ~ t^3 + noise). Real human motion is
    NOT exchangeable with this, so the guarantee does NOT transfer to deployment. Closing
    that gap requires recalibrating Q_alpha on REAL prediction-error residuals (mocap /
    a pedestrian dataset e.g. ETH-UCY, or a realistic sim human) -- this is method-sound,
    not yet deployment-safe. Tracked as the conformal line's next real boundary.

MC NOTE: a held-out coverage of ~0.901 at n_test=20000 sits at the NOMINAL level within
Monte-Carlo noise (per-split std ~ sqrt(0.9*0.1/n_test) ~ 0.21%). It is "exactly on target,
as expected", NOT a safety buffer above target -- do not read the 0.001 excess as margin.
The (1-alpha) quantile already uses the conservative finite-sample index ceil((n+1)(1-a))/n,
which makes the MARGINAL (over-the-draw) coverage >= 1-alpha; per-draw >= 1-alpha would need
a training-conditional (PAC / Beta-tail) bound, and MARG.0's ~44%-below-target is precisely
the marginal-not-per-draw signature.

Run:  python3 test/stage4_conformal_coverage.py
"""
import numpy as np

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# --- synthetic setup (seed fixed so 0.9008 / 1.60x reproduce) ---------------
rng = np.random.default_rng(2026)
T, N = 1.0, 41
ts = np.linspace(0.0, T, N)
ALPHA = 0.1
# n_test is a LARGE, INDEPENDENT held-out set (disjoint from calibration) so the
# reported coverage is genuine generalization, not the circular in-sample number.
n_cal, n_test, REPEAT = 500, 20000, 200


def sample(n):
    """Human true = CA + UNMODELLED jerk (predictor sees only CA); error ~ t^3 + noise."""
    j = rng.normal(0.0, 1.5, (n, 3))
    e = (1.0 / 6.0) * j[:, None, :] * (ts[None, :, None] ** 3) + rng.normal(0.0, 0.02, (n, N, 3))
    return np.linalg.norm(e, axis=2)                       # (n, N) ||e(t)||


# conformal levels (finite-sample corrected (1-alpha) quantile indices)
lvl_traj = min(np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal, 1.0)
lvl_union = min(np.ceil((n_cal + 1) * (1 - ALPHA / N)) / n_cal, 1.0)

cov_t, cov_u, ratio, cov_in = [], [], [], []
for _ in range(REPEAT):
    cal = sample(n_cal)                                    # (n_cal, N)  calibration
    test = sample(n_test)                                  # (n_test, N) DISJOINT held-out
    # trajectory-level conformal: ONE scalar per trajectory = sup_t ||e(t)||
    cal_sup = cal.max(axis=1)
    Q_traj = float(np.quantile(cal_sup, lvl_traj, method="higher"))
    # naive union bound: per-timestep conformal at alpha/N, then max over t
    Q_union = float(np.quantile(cal, lvl_union, axis=0, method="higher").max())
    sup_test = test.max(axis=1)
    cov_t.append(np.mean(sup_test <= Q_traj))              # HELD-OUT coverage (the real test)
    cov_in.append(np.mean(cal_sup <= Q_traj))              # IN-SAMPLE coverage (circular)
    cov_u.append(np.mean(sup_test <= Q_union))
    ratio.append(Q_union / Q_traj)

cov_t = np.asarray(cov_t)
cov_u = np.asarray(cov_u)
cov_in = np.asarray(cov_in)
mt, mu, mr = float(cov_t.mean()), float(cov_u.mean()), float(np.mean(ratio))
m_in = float(cov_in.mean())
target = 1.0 - ALPHA

print(f"\naveraged over {REPEAT} calibration splits (n_cal={n_cal}, HELD-OUT n_test={n_test}), "
      f"target >= {target:.2f}")
print(f"trajectory-conformal, HELD-OUT (the real test): mean coverage = {mt:.4f}")
print(f"trajectory-conformal, IN-SAMPLE on calibration : mean coverage = {m_in:.4f}  <- CIRCULAR, not validation")
print(f"naive per-timestep union bound, held-out       : mean coverage = {mu:.4f}")
print(f"union bound needs a {mr:.2f}x LARGER safety margin Q for no better coverage")
mc_std = float(np.sqrt(target * ALPHA / n_test))
print(f"MC note: per-split std ~ {mc_std*100:.2f}% at n_test={n_test} -> {mt:.4f} is AT nominal "
      f"{target:.2f} within noise, NOT a buffer above it\n")

# --- COV: trajectory-conformal hits the target coverage ON HELD-OUT DATA ------
# This is the only number that VALIDATES anything: sup_test is a fresh sample of
# n_test=20000 drawn INDEPENDENTLY of the calibration set used to pick Q_traj.
check("COV.0 trajectory-conformal HELD-OUT mean coverage >= target 1-alpha",
      mt >= target, f"held-out mean {mt:.4f} >= {target:.2f}  (n_test={n_test}, disjoint from cal)")
check("COV.1 union bound over-covers on held-out (conservative, as expected)",
      mu > target, f"mean {mu:.4f} > {target:.2f}")
# Guard against the classic self-deception: the IN-SAMPLE number is ~target BY
# CONSTRUCTION (ceil((n+1)(1-alpha))/n of cal points lie at/below their own
# (1-alpha) empirical quantile), so it can NEVER be the evidence. We show it lands
# near target too, but the HELD-OUT number (COV.0) on a disjoint 20k sample is the
# one that demonstrates generalization -- they agree because exchangeability holds.
check("COV.2 in-sample coverage is ~target by construction (circular, NOT the evidence)",
      abs(m_in - lvl_traj) < 0.01,
      f"in-sample {m_in:.4f} ~= quantile level {lvl_traj:.4f} (circular); held-out {mt:.4f} is the real check")

# --- TIGHT: sup-norm score strictly tighter than the union bound -------------
check("TIGHT.0 trajectory-conformal strictly tighter than union bound",
      mr > 1.05, f"{mr:.2f}x smaller Q for same/better coverage")

# --- MARG: the guarantee is marginal over the calibration draw ---------------
# A single split fluctuates around 1-alpha; some splits dip below it. This is
# honest: the conformal guarantee holds in expectation over the draw, not for
# every individual calibration set. The MEAN (COV.0) is what must clear target.
frac_below = float(np.mean(cov_t < target))
check("MARG.0 single splits do fluctuate below target (marginal, not per-split)",
      0.0 < frac_below < 0.5,
      f"{frac_below * 100:.0f}% of splits below target, std={cov_t.std():.4f}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
