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

HONEST SCOPE: synthetic error model (CA predictor + unmodelled jerk, e ~ t^3 + noise),
grid sup over the horizon, exchangeability assumed.  The theorem + the sup-norm-score
mechanism (avoiding the union bound) stand; this is framework validation, not the
final on-trajectory implementation (Bernstein-hull sup_s||e|| + noisy closed-loop
come next).

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
n_cal, n_test, REPEAT = 500, 2000, 200


def sample(n):
    """Human true = CA + UNMODELLED jerk (predictor sees only CA); error ~ t^3 + noise."""
    j = rng.normal(0.0, 1.5, (n, 3))
    e = (1.0 / 6.0) * j[:, None, :] * (ts[None, :, None] ** 3) + rng.normal(0.0, 0.02, (n, N, 3))
    return np.linalg.norm(e, axis=2)                       # (n, N) ||e(t)||


# conformal levels (finite-sample corrected (1-alpha) quantile indices)
lvl_traj = min(np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal, 1.0)
lvl_union = min(np.ceil((n_cal + 1) * (1 - ALPHA / N)) / n_cal, 1.0)

cov_t, cov_u, ratio = [], [], []
for _ in range(REPEAT):
    cal = sample(n_cal)                                    # (n_cal, N)
    test = sample(n_test)                                  # (n_test, N)
    # trajectory-level conformal: ONE scalar per trajectory = sup_t ||e(t)||
    Q_traj = float(np.quantile(cal.max(axis=1), lvl_traj, method="higher"))
    # naive union bound: per-timestep conformal at alpha/N, then max over t
    Q_union = float(np.quantile(cal, lvl_union, axis=0, method="higher").max())
    sup_test = test.max(axis=1)
    cov_t.append(np.mean(sup_test <= Q_traj))
    cov_u.append(np.mean(sup_test <= Q_union))
    ratio.append(Q_union / Q_traj)

cov_t = np.asarray(cov_t)
cov_u = np.asarray(cov_u)
mt, mu, mr = float(cov_t.mean()), float(cov_u.mean()), float(np.mean(ratio))
target = 1.0 - ALPHA

print(f"\naveraged over {REPEAT} calibration splits (n_cal={n_cal}, n_test={n_test}), "
      f"target >= {target:.2f}")
print(f"trajectory-conformal (sup-norm score):  mean coverage = {mt:.4f}")
print(f"naive per-timestep union bound:         mean coverage = {mu:.4f}")
print(f"union bound needs a {mr:.2f}x LARGER safety margin Q for no better coverage\n")

# --- COV: trajectory-conformal hits the target marginal coverage -------------
check("COV.0 trajectory-conformal mean coverage >= target 1-alpha",
      mt >= target, f"mean {mt:.4f} >= {target:.2f}")
check("COV.1 union bound over-covers (conservative, as expected)",
      mu > target, f"mean {mu:.4f} > {target:.2f}")

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
