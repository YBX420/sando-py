"""Stage 3 — local/minco.py FUNCTIONAL verification (MINCO min-jerk, s=3).

Standalone — loads minco.py directly, no ROS / package __init__.

Oracle: the closed-form rest-to-rest minimum-jerk quintic and a directly-built
quintic from arbitrary boundary conditions (Hermite). MINCO with M=1 and the
matching boundary state MUST reproduce these to ~1e-9.

Run:  python3 test/stage3_minco.py

Coverage:
  A. single segment, rest-to-rest, closed-form min-jerk  p(s)=p0+Δ(10s^3-15s^4+6s^5)
  B. single segment, arbitrary v0/a0/vf/af → solve the 6x6 Hermite system as oracle
  C. multi-segment: every interior waypoint EXACTLY interpolated
  D. coefficient round-trip: M(T) c = b holds (residual ~ machine eps)
  E. banded solve == dense solve (no silent corruption)
"""
import importlib.util
import os
import numpy as np

MOD_PATH = os.path.join(os.path.dirname(__file__), "..", "sando_py", "local", "minco.py")
spec = importlib.util.spec_from_file_location("minco_standalone", os.path.abspath(MOD_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MinjerkTraj = mod.MinjerkTraj

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def quintic_from_bc(p0, v0, a0, pT, vT, aT, T):
    """Solve the unique quintic with the 6 given boundary derivatives (per dim).

    Returns coeffs c[0..5] ascending so q(t)=sum c_j t^j. Acts as an independent
    oracle for the single-segment MINCO solve (different equations, same answer).
    """
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); a0 = np.asarray(a0, float)
    pT = np.asarray(pT, float); vT = np.asarray(vT, float); aT = np.asarray(aT, float)
    # rows: q(0),q'(0),q''(0),q(T),q'(T),q''(T)
    T2, T3, T4, T5 = T*T, T**3, T**4, T**5
    A = np.array([
        [1, 0, 0, 0,    0,     0],
        [0, 1, 0, 0,    0,     0],
        [0, 0, 2, 0,    0,     0],
        [1, T, T2, T3,  T4,    T5],
        [0, 1, 2*T, 3*T2, 4*T3, 5*T4],
        [0, 0, 2, 6*T, 12*T2, 20*T3],
    ], float)
    rhs = np.stack([p0, v0, a0, pT, vT, aT])  # (6, 3)
    return np.linalg.solve(A, rhs)  # (6, 3)


# ---------------------------------------------------------------- A. closed-form rest-to-rest
print("\n--- A. single segment rest-to-rest vs closed-form min-jerk ---")
rng = np.random.default_rng(0)
worstA = 0.0
for trial in range(8):
    p0 = rng.standard_normal(3) * 3
    pf = rng.standard_normal(3) * 3
    T = float(rng.uniform(0.3, 4.0))
    tr = MinjerkTraj(np.vstack([p0, pf]), [T])
    s = np.linspace(0, 1, 21)
    oracle = p0 + (pf - p0) * (10*s**3 - 15*s**4 + 6*s**5)[:, None]
    mine = tr.eval_deriv(s * T, 0)
    err = float(np.max(np.abs(mine - oracle)))
    worstA = max(worstA, err)
check("A.0 rest-to-rest matches 10s^3-15s^4+6s^5 (8 random)", worstA < 1e-9, f"worst={worstA:.2e}")
# also vel/accel zero at both ends
tr = MinjerkTraj(np.array([[0.,0,0],[1.,2,3]]), [2.0])
ev = max(np.max(np.abs(tr.eval_deriv(0.0,1))), np.max(np.abs(tr.eval_deriv(2.0,1))))
ea = max(np.max(np.abs(tr.eval_deriv(0.0,2))), np.max(np.abs(tr.eval_deriv(2.0,2))))
check("A.1 rest-to-rest endpoints v=0", ev < 1e-9, f"|v|={ev:.2e}")
check("A.2 rest-to-rest endpoints a=0", ea < 1e-9, f"|a|={ea:.2e}")


# ---------------------------------------------------------------- B. arbitrary BC vs Hermite oracle
print("\n--- B. single segment, arbitrary v0/a0/vf/af vs independent Hermite solve ---")
worstB = 0.0
worstBd = 0.0
for trial in range(10):
    p0 = rng.standard_normal(3); pf = rng.standard_normal(3)
    v0 = rng.standard_normal(3); a0 = rng.standard_normal(3)
    vf = rng.standard_normal(3); af = rng.standard_normal(3)
    T = float(rng.uniform(0.5, 3.0))
    tr = MinjerkTraj(np.vstack([p0, pf]), [T], v0=v0, a0=a0, vf=vf, af=af)
    coeff = quintic_from_bc(p0, v0, a0, pf, vf, af, T)  # (6,3) ascending
    ts = np.linspace(0, T, 25)
    oracle = np.array([np.array([t**j for j in range(6)]) @ coeff for t in ts])
    mine = tr.eval_deriv(ts, 0)
    worstB = max(worstB, float(np.max(np.abs(mine - oracle))))
    # derivative orders 1..4 vs analytic differentiation of the oracle poly
    for o in range(1, 5):
        dco = np.zeros_like(coeff)
        for j in range(o, 6):
            cf = 1.0
            for k in range(o):
                cf *= (j - k)
            dco[j-o] += cf  # placeholder, replaced below
        # build derivative oracle directly
        dor = np.array([
            np.array([ (np.prod([j-k for k in range(o)]) * (t**(j-o)) if j>=o else 0.0)
                       for j in range(6) ]) @ coeff
            for t in ts])
        mo = tr.eval_deriv(ts, o)
        worstBd = max(worstBd, float(np.max(np.abs(mo - dor))))
check("B.0 arbitrary-BC position matches Hermite quintic (10 random)", worstB < 1e-9, f"worst={worstB:.2e}")
check("B.1 arbitrary-BC derivatives (o=1..4) match", worstBd < 1e-8, f"worst={worstBd:.2e}")


# ---------------------------------------------------------------- C. multi-segment waypoint interpolation
print("\n--- C. multi-segment EXACT waypoint interpolation ---")
worstC = 0.0
for trial in range(8):
    M = int(rng.integers(2, 7))
    wp = rng.standard_normal((M+1, 3)) * 4
    T = rng.uniform(0.4, 2.5, size=M)
    tr = MinjerkTraj(wp, T)
    cum = np.concatenate([[0.0], np.cumsum(T)])
    # every junction (interior + endpoints) must hit the waypoint exactly
    for i in range(M+1):
        err = float(np.max(np.abs(tr.eval_deriv(cum[i], 0) - wp[i])))
        worstC = max(worstC, err)
check("C.0 all junctions interpolate waypoint (8 random, M=2..6)", worstC < 1e-9, f"worst={worstC:.2e}")


# ---------------------------------------------------------------- D. M(T)c=b residual
print("\n--- D. linear-system residual M(T) c = b ---")
worstD = 0.0
for trial in range(6):
    M = int(rng.integers(1, 8))
    wp = rng.standard_normal((M+1, 3)) * 3
    T = rng.uniform(0.3, 3.0, size=M)
    tr = MinjerkTraj(wp, T)
    A = tr.dense_M()
    res = float(np.max(np.abs(A @ tr.c - tr._b)))
    worstD = max(worstD, res)
check("D.0 residual |M c - b| ~ machine eps", worstD < 1e-9, f"worst={worstD:.2e}")


# ---------------------------------------------------------------- E. banded == dense
print("\n--- E. banded solve == dense solve ---")
worstE = 0.0
for trial in range(6):
    M = int(rng.integers(2, 9))
    wp = rng.standard_normal((M+1, 3)) * 5
    T = rng.uniform(0.2, 4.0, size=M)
    tr = MinjerkTraj(wp, T)
    c_dense = np.linalg.solve(tr.dense_M(), tr._b)
    worstE = max(worstE, float(np.max(np.abs(c_dense - tr.c))))
check("E.0 solve_banded == dense linalg.solve", worstE < 1e-9, f"worst={worstE:.2e}")


# ---------------------------------------------------------------- summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    print("Failures:")
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
