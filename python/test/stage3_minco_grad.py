"""Stage 3 — local/minco.py M1 ANALYTIC GRADIENT verification (MINCO min-jerk).

Standalone — loads minco.py directly, no ROS / package __init__.

GATE: central-difference check of the analytic gradients of the jerk-energy
cost J = integral ||jerk||^2 dt through the MINCO map  M(T) c = b(q):
  - dJ/dq  vs FD,  rel-err < 1e-6,  over >= 100 random (q,T,bcs) configs
  - dJ/dT  vs FD,  rel-err < 1e-6,  over the same configs
  - a single-T_i perturbation test (time-gradient only)
  - the generic backprop hook grad_from_dcost_dc on an EXTERNAL sampled cost
    (a smooth quadratic on trajectory samples) vs FD
  - the closed-form energy J vs an independent high-order quadrature
  - the energy partials dJ/dc and explicit dJ/dT vs sympy-free FD on c and T

All configs kept SMOOTH (jerk-energy is C-infinity in (q,T) for T>0, no kink),
so central differences converge cleanly.

Run:  python3 test/stage3_minco_grad.py
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


def rel_err(a, b):
    """Relative error with an absolute floor so near-zero comps don't blow up."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, 1e-8)
    return np.max(np.abs(a - b) / denom)


def fd_central_richardson(f, x, h0=1e-3):
    """Scalar-output central-difference d f/d x at scalar x with one Richardson
    step (Romberg O(h^4)).  f returns a float.  Far more accurate than a single
    central difference for the high-magnitude polynomial-in-T energy, so the
    1e-6 rel-err GATE reflects the ANALYTIC gradient, not FD truncation.

        D(h)   = (f(x+h)-f(x-h)) / (2h)            ~ g + a h^2 + ...
        D(h/2) = ...
        g ~= (4 D(h/2) - D(h)) / 3                 -> O(h^4)
    """
    def D(h):
        return (f(x + h) - f(x - h)) / (2.0 * h)
    d1 = D(h0)
    d2 = D(h0 / 2.0)
    return (4.0 * d2 - d1) / 3.0


def make_traj(rng, M, bc=True, scale=3.0):
    """Random smooth MINCO config (start, goal, q, T, optional bcs)."""
    wp = rng.standard_normal((M + 1, 3)) * scale
    T = rng.uniform(0.4, 2.0, size=M)
    kw = {}
    if bc:
        kw = dict(v0=rng.standard_normal(3) * 0.5,
                  a0=rng.standard_normal(3) * 0.5,
                  vf=rng.standard_normal(3) * 0.5,
                  af=rng.standard_normal(3) * 0.5)
    return wp, T, kw


# ============================================================ 0. energy vs quadrature
print("\n--- 0. closed-form energy J vs independent quadrature ---")
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(20):
    M = int(rng.integers(1, 6))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    J_analytic = tr.energy()
    # independent quadrature of ||jerk||^2 with fine Gauss-Legendre per segment
    xg, wg = np.polynomial.legendre.leggauss(40)
    Jq = 0.0
    cum = np.concatenate([[0.0], np.cumsum(T)])
    for i in range(M):
        a, b = cum[i], cum[i + 1]
        ts = 0.5 * (b - a) * xg + 0.5 * (a + b)
        jk = tr.eval_deriv(ts, 3)  # (K,3)
        integrand = np.sum(jk * jk, axis=1)
        Jq += 0.5 * (b - a) * np.sum(wg * integrand)
    worst = max(worst, abs(J_analytic - Jq) / max(abs(Jq), 1e-9))
check("0.0 energy J == quadrature (20 random)", worst < 1e-9, f"worst rel={worst:.2e}")


# ============================================================ 1. dJ/dc vs INDEPENDENT exact oracle
print("\n--- 1. energy partials (J, dJ/dc, explicit dJ/dT) vs independent sympy oracle ---")
# dJ/dc is the gradient of a closed-form QUADRATIC in c, so a single-h FD cannot
# reach 1e-6 on near-zero entries against an O(1e4) energy (cancellation). The
# correct oracle is the EXACT derivative: re-derive the per-segment jerk-energy
# integral symbolically (sympy), build dJ/dc3,c4,c5 and dJ/dT as Python lambdas,
# and compare to the code's hardcoded formulas at random (c,T). This pins down
# every numeric coefficient (72,144,240,384,720,1440 / 36,288,576,720,2880,3600).
import sympy as _sp
_t, _Tv = _sp.symbols("t Tv", positive=True)
_c3 = _sp.IndexedBase("c3"); _c4 = _sp.IndexedBase("c4"); _c5 = _sp.IndexedBase("c5")
_jerk = [6 * _c3[d] + 24 * _c4[d] * _t + 60 * _c5[d] * _t ** 2 for d in range(3)]
_Jsym = sum(_sp.integrate(_jerk[d] ** 2, (_t, 0, _Tv)) for d in range(3))
_syms = [_c3[d] for d in range(3)] + [_c4[d] for d in range(3)] + [_c5[d] for d in range(3)] + [_Tv]
_J_f = _sp.lambdify(_syms, _Jsym, "numpy")
_gc3_f = [_sp.lambdify(_syms, _sp.diff(_Jsym, _c3[d]), "numpy") for d in range(3)]
_gc4_f = [_sp.lambdify(_syms, _sp.diff(_Jsym, _c4[d]), "numpy") for d in range(3)]
_gc5_f = [_sp.lambdify(_syms, _sp.diff(_Jsym, _c5[d]), "numpy") for d in range(3)]
_gT_f = _sp.lambdify(_syms, _sp.diff(_Jsym, _Tv), "numpy")

worstJ = worstC = worstTexp = 0.0
for _ in range(40):
    M = int(rng.integers(1, 6))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    Jcode, gdC = tr._energy_and_gradc()
    gTexp = tr._energy_grad_time_explicit()
    Jor = 0.0
    for i in range(M):
        b = 6 * i
        a = list(tr.c[b + 3]) + list(tr.c[b + 4]) + list(tr.c[b + 5]) + [T[i]]
        Jor += float(_J_f(*a))
        for d in range(3):  # relative error: code vs sympy are the SAME formula
            worstC = max(worstC, rel_err(gdC[b + 3, d], _gc3_f[d](*a)))
            worstC = max(worstC, rel_err(gdC[b + 4, d], _gc4_f[d](*a)))
            worstC = max(worstC, rel_err(gdC[b + 5, d], _gc5_f[d](*a)))
        worstTexp = max(worstTexp, rel_err(gTexp[i], _gT_f(*a)))
        # head rows (0,1,2) of each segment must be exactly zero in dJ/dc
        worstC = max(worstC, float(np.max(np.abs(gdC[b:b + 3]))))
    worstJ = max(worstJ, abs(Jcode - Jor) / max(abs(Jor), 1e-9))
check("1.0 energy J == sympy oracle (40 random)", worstJ < 1e-12, f"worst rel={worstJ:.2e}")
check("1.1 dJ/dc == sympy exact derivative (40 random)", worstC < 1e-9, f"worst rel={worstC:.2e}")
check("1.2 explicit dJ/dT == sympy exact derivative (40 random)", worstTexp < 1e-9, f"worst rel={worstTexp:.2e}")


# ============================================================ 2. explicit dJ/dT vs FD
print("\n--- 2. explicit dJ/dT (c fixed) vs central FD ---")
worst = 0.0
for _ in range(15):
    M = int(rng.integers(1, 5))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    gExp = tr._energy_grad_time_explicit()
    c_fixed = tr.c.copy()
    T0 = tr.T.copy()
    def J_explicit_k(k):
        def f(val):
            Tv = T0.copy(); Tv[k] = val
            tr.T = Tv; tr.c = c_fixed   # energy(T) with c held fixed
            return tr._energy_and_gradc()[0]
        return f
    gfd = np.zeros(M)
    for k in range(M):
        gfd[k] = fd_central_richardson(J_explicit_k(k), T0[k])
    tr.T = T0; tr.c = c_fixed
    worst = max(worst, rel_err(gExp, gfd))
check("2.0 explicit dJ/dT == central FD (15 random)", worst < 1e-6, f"worst rel={worst:.2e}")


# ============================================================ 3. FULL dJ/dq vs FD (>=100 configs)
print("\n--- 3. FULL dJ/dq vs central FD through the MINCO map (>=100 configs) ---")
def J_total(wp, T, kw):
    return MinjerkTraj(wp, T, **kw).energy()

worst_q = 0.0
n_cfg = 0
for cfg in range(120):
    M = int(rng.integers(2, 7))      # need interior waypoints -> M>=2
    bc = (cfg % 2 == 0)
    wp, T, kw = make_traj(rng, M, bc=bc)
    tr = MinjerkTraj(wp, T, **kw)
    _, gq, _ = tr.energy_grad()
    # FD on interior waypoints q (rows 1..M-1 of wp), Richardson O(h^4)
    gfd = np.zeros((M - 1, 3))
    for i in range(M - 1):
        for d in range(3):
            def f(val, i=i, d=d):
                w = wp.copy(); w[i + 1, d] = val
                return J_total(w, T, kw)
            gfd[i, d] = fd_central_richardson(f, wp[i + 1, d])
    worst_q = max(worst_q, rel_err(gq, gfd))
    n_cfg += 1
check(f"3.0 dJ/dq == central FD ({n_cfg} configs)", worst_q < 1e-6, f"worst rel={worst_q:.2e}")


# ============================================================ 4. FULL dJ/dT vs FD (>=100 configs)
print("\n--- 4. FULL dJ/dT vs central FD through the MINCO map (>=100 configs) ---")
worst_T = 0.0
n_cfg = 0
for cfg in range(120):
    M = int(rng.integers(1, 7))
    bc = (cfg % 2 == 1)
    wp, T, kw = make_traj(rng, M, bc=bc)
    tr = MinjerkTraj(wp, T, **kw)
    _, _, gT = tr.energy_grad()
    gfd = np.zeros(M)
    for k in range(M):
        def f(val, k=k):
            Tv = T.copy(); Tv[k] = val
            return J_total(wp, Tv, kw)
        gfd[k] = fd_central_richardson(f, T[k])
    worst_T = max(worst_T, rel_err(gT, gfd))
    n_cfg += 1
check(f"4.0 dJ/dT == central FD ({n_cfg} configs)", worst_T < 1e-6, f"worst rel={worst_T:.2e}")


# ============================================================ 5. single-T_i perturb (time-grad only)
print("\n--- 5. single-T_i perturbation, time gradient only ---")
worst = 0.0
detail_pairs = []
for _ in range(30):
    M = int(rng.integers(2, 6))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    _, _, gT = tr.energy_grad()
    k = int(rng.integers(0, M))         # perturb exactly ONE T_k
    def f(val, k=k):
        Tv = T.copy(); Tv[k] = val
        return J_total(wp, Tv, kw)
    gfd_k = fd_central_richardson(f, T[k])
    e = abs(gT[k] - gfd_k) / max(abs(gfd_k), 1e-8)
    worst = max(worst, e)
    detail_pairs.append((gT[k], gfd_k))
a0, f0 = detail_pairs[0]
check("5.0 single-T_k grad matches FD (30 random)", worst < 1e-6,
      f"worst rel={worst:.2e}, e.g. analytic={a0:.4f} fd={f0:.4f}")


# ============================================================ 6. generic hook on EXTERNAL cost
print("\n--- 6. generic grad_from_dcost_dc on an external sampled cost vs FD ---")
# This is exactly how the per-class obstacle cost will use the hook. We sample
# the trajectory at FIXED per-segment fractions s in (0,1) (the GCOPTER way),
# so each sample sits at  tau = s * T_i  in segment i, and define
#     C(c,T) = sum over samples ||p_i(s*T_i) - target||^2.
# Because tau depends on T_i, the cost has BOTH:
#   * an implicit T-path (c = M(T)^{-1} b)            -> captured by the adjoint
#   * an explicit T-path (the sample point tau=s*T_i) -> supplied as
#       dC/dT_i|explicit = 2 diff . vel(tau) * s   (per sample in seg i)
# The hook must reproduce the TOTAL dC/dq and dC/dT under FD on (q,T) with the
# SAME fixed-fraction sampling.  This is the decisive end-to-end check of the
# generic backprop path with a nonzero explicit-time term.
FRACS = np.array([0.2, 0.5, 0.8])

def ext_cost_dc_dTexp(tr, target):
    """C, dC/dc (6M,3), dC/dT_explicit (M,) for the fixed-fraction sampled cost."""
    M = tr.M
    n = tr.NC * M
    dC = np.zeros((n, 3))
    dTexp = np.zeros(M)
    C = 0.0
    for i in range(M):
        for s in FRACS:
            tau = s * tr.T[i]
            row0 = MinjerkTraj._basis(tau, 0)   # position basis
            row1 = MinjerkTraj._basis(tau, 1)   # velocity basis
            ci = tr.c[6 * i:6 * i + 6]
            p = row0 @ ci
            vel = row1 @ ci
            diff = p - target
            C += float(diff.dot(diff))
            dC[6 * i:6 * i + 6] += 2.0 * np.outer(row0, diff)
            # explicit d/dT_i of ||p_i(s T_i)-target||^2  at fixed c:
            #   2 diff . (dp/dtau)(dtau/dT_i) = 2 diff . vel * s
            dTexp[i] += 2.0 * float(diff.dot(vel)) * s
    return C, dC, dTexp

def ext_cost_only(wp, T, kw, target):
    tr = MinjerkTraj(wp, T, **kw)
    C = 0.0
    for i in range(tr.M):
        for s in FRACS:
            p = tr.eval_deriv(s * T[i] + tr._cum[i], 0)
            diff = p - target
            C += float(diff.dot(diff))
    return C

worst_q = 0.0
worst_T = 0.0
for _ in range(30):
    M = int(rng.integers(2, 6))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    target = rng.standard_normal(3)
    C, dCdc, dTexp = ext_cost_dc_dTexp(tr, target)
    gq, gT = tr.grad_from_dcost_dc(dCdc, dcost_dT_explicit=dTexp)
    # FD dC/dq (Richardson O(h^4))
    gfd_q = np.zeros((M - 1, 3))
    for i in range(M - 1):
        for d in range(3):
            def fq(val, i=i, d=d):
                w = wp.copy(); w[i + 1, d] = val
                return ext_cost_only(w, T, kw, target)
            gfd_q[i, d] = fd_central_richardson(fq, wp[i + 1, d])
    worst_q = max(worst_q, rel_err(gq, gfd_q))
    # FD dC/dT  (fixed FRACTIONS -> sample tau scales with T, the explicit path)
    gfd_T = np.zeros(M)
    for k in range(M):
        def fT(val, k=k):
            Tv = T.copy(); Tv[k] = val
            return ext_cost_only(wp, Tv, kw, target)
        gfd_T[k] = fd_central_richardson(fT, T[k])
    worst_T = max(worst_T, rel_err(gT, gfd_T))
check("6.0 hook dC/dq == FD (external cost, 30 random)", worst_q < 1e-6, f"worst rel={worst_q:.2e}")
check("6.1 hook dC/dT == FD (external cost, 30 random)", worst_T < 1e-6, f"worst rel={worst_T:.2e}")


# ====================================================== 6b. hook with explicit=None, IMPLICIT-only T path
print("\n--- 6b. generic hook, T-sensitive cost, dcost_dT_explicit=None (implicit path only) ---")
# RED-TEAM addition: a cost whose ONLY T-dependence is the implicit c = M(T)^-1 b
# path (no explicit tau reparametrization), so the caller correctly passes
# dcost_dT_explicit=None. C = sum_i ||c5_i||^2 (squared leading coeff). c5
# depends strongly on T through the solve. Verified TRUNCATION-FREE via
# complex-step differentiation (Im(f(x+i h))/h, h=1e-30) of the whole map with an
# INDEPENDENT dense np.linalg.solve -- no FD cancellation, machine precision.
def _basis_cx(tau, order):
    row = np.zeros(6, dtype=np.complex128)
    for j in range(order, 6):
        co = 1.0
        for k in range(order):
            co *= (j - k)
        row[j] = co * (tau ** (j - order))
    return row

def _build_A_b_cx(M, T, ps, pg, q, v0, a0, vf, af):
    n = 6 * M
    A = np.zeros((n, n), np.complex128); b = np.zeros((n, 3), np.complex128)
    A[0, 0] = 1; A[1, 1] = 1; A[2, 2] = 2
    b[0] = ps; b[1] = v0; b[2] = a0
    for i in range(M - 1):
        base = 6 * i; Ti = T[i]
        A[base + 3, base:base + 6] = _basis_cx(Ti, 3); A[base + 3, base + 6:base + 12] -= _basis_cx(0, 3)
        A[base + 4, base:base + 6] = _basis_cx(Ti, 4); A[base + 4, base + 6:base + 12] -= _basis_cx(0, 4)
        A[base + 5, base:base + 6] = _basis_cx(Ti, 0); b[base + 5] = q[i]
        A[base + 6, base:base + 6] = _basis_cx(Ti, 0); A[base + 6, base + 6:base + 12] -= _basis_cx(0, 0)
        A[base + 7, base:base + 6] = _basis_cx(Ti, 1); A[base + 7, base + 6:base + 12] -= _basis_cx(0, 1)
        A[base + 8, base:base + 6] = _basis_cx(Ti, 2); A[base + 8, base + 6:base + 12] -= _basis_cx(0, 2)
    last = M - 1; base = 6 * last; TN = T[last]
    A[n - 3, base:base + 6] = _basis_cx(TN, 0); A[n - 2, base:base + 6] = _basis_cx(TN, 1); A[n - 1, base:base + 6] = _basis_cx(TN, 2)
    b[n - 3] = pg; b[n - 2] = vf; b[n - 1] = af
    return A, b

def _costC_cx(M, T, ps, pg, q, v0, a0, vf, af):
    A, b = _build_A_b_cx(M, T, ps, pg, q, v0, a0, vf, af)
    c = np.linalg.solve(A, b)
    C = 0. + 0.j
    for i in range(M):
        C += np.sum(c[6 * i + 5] * c[6 * i + 5])
    return C

HCS = 1e-30
worst_T = 0.0
for _ in range(30):
    M = int(rng.integers(2, 6))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    n = tr.NC * M
    dC = np.zeros((n, 3))
    for i in range(M):
        dC[6 * i + 5] += 2.0 * tr.c[6 * i + 5]
    _, gT = tr.grad_from_dcost_dc(dC, dcost_dT_explicit=None)
    v0 = kw.get("v0", np.zeros(3)); a0 = kw.get("a0", np.zeros(3))
    vf = kw.get("vf", np.zeros(3)); af = kw.get("af", np.zeros(3))
    gcs = np.zeros(M)
    for k in range(M):
        Tc = T.astype(np.complex128).copy(); Tc[k] += 1j * HCS
        gcs[k] = _costC_cx(M, Tc, wp[0], wp[-1], wp[1:-1].astype(np.complex128),
                           v0, a0, vf, af).imag / HCS
    worst_T = max(worst_T, rel_err(gT, gcs))
check("6b.0 hook dC/dT (explicit=None) == complex-step (30 random)", worst_T < 1e-6,
      f"worst rel={worst_T:.2e}")


# ============================================================ 7. adjoint solve is M^T (and banded)
print("\n--- 7. adjoint solve correctness: M^T x = rhs ---")
worst = 0.0
for _ in range(10):
    M = int(rng.integers(1, 8))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    rhs = rng.standard_normal((6 * M, 3))
    x = tr._solve_adjoint(rhs)
    A = tr.dense_M()
    res = float(np.max(np.abs(A.T @ x - rhs)))
    worst = max(worst, res)
check("7.0 _solve_adjoint solves M^T x = rhs (10 random)", worst < 1e-8, f"worst |M^T x - rhs|={worst:.2e}")


# ============================================================ 8. dM/dT_k . c vs FD
print("\n--- 8. (dM/dT_k) c term vs finite difference of M(T) c ---")
worst = 0.0
for _ in range(12):
    M = int(rng.integers(1, 7))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    c = tr.c
    for k in range(M):
        analytic = tr._dM_dT_times_c(k)
        # FD: d/dT_k of  M(T) c  with c held FIXED (Richardson, vector output)
        def f(val, k=k):
            Tv = T.copy(); Tv[k] = val
            return MinjerkTraj(wp, Tv, **kw).dense_M() @ c
        fd = fd_central_richardson(f, T[k])
        worst = max(worst, rel_err(analytic, fd))
check("8.0 (dM/dT_k) c == FD (12 random, all k)", worst < 1e-6, f"worst rel={worst:.2e}")


# ============================================================ 9. rest-to-rest analytic energy
print("\n--- 9. single-segment rest-to-rest: analytic energy vs known closed form ---")
# rest-to-rest min-jerk on [0,T], displacement d:  p(s)=d(10s^3-15s^4+6s^5),
# jerk integral = 720 d^2 / T^5 (classic result, per dim).
worst = 0.0
for _ in range(10):
    d = rng.standard_normal(3) * 2
    T = float(rng.uniform(0.3, 3.0))
    tr = MinjerkTraj(np.vstack([np.zeros(3), d]), [T])
    J = tr.energy()
    Jref = 720.0 * float(d.dot(d)) / T**5
    worst = max(worst, abs(J - Jref) / max(abs(Jref), 1e-9))
check("9.0 rest-to-rest energy == 720 d^2 / T^5 (10 random)", worst < 1e-9, f"worst rel={worst:.2e}")


# ============================================================ 10. gradient determinism / shapes
print("\n--- 10. shapes & determinism ---")
tr = MinjerkTraj(rng.standard_normal((5, 3)), rng.uniform(0.5, 1.5, 4))
J, gq, gT = tr.energy_grad()
shape_ok = (np.ndim(J) == 0 and gq.shape == (3, 3) and gT.shape == (4,))
check("10.0 energy_grad shapes (J scalar, gq (M-1,3), gT (M,))", shape_ok,
      f"gq{gq.shape} gT{gT.shape}")
J2, gq2, gT2 = tr.energy_grad()
det = (abs(J - J2) == 0 and np.array_equal(gq, gq2) and np.array_equal(gT, gT2))
check("10.1 energy_grad is deterministic", det, "bit-exact repeat")
# hook rejects wrong shape
try:
    tr.grad_from_dcost_dc(np.zeros((3, 3)))
    bad = False
except ValueError:
    bad = True
check("10.2 hook rejects wrong dcost_dc shape", bad)


# ============================================================ 11. boundary / adversarial gradients
print("\n--- 11. boundary & adversarial gradient checks ---")
# M=1 (no interior waypoints): dJ/dq is empty (0,3); dJ/dT_0 must still match FD.
wp = rng.standard_normal((2, 3)); T = np.array([1.3])
tr = MinjerkTraj(wp, T)
J, gq, gT = tr.energy_grad()
def f1(val):
    return MinjerkTraj(wp, np.array([val])).energy()
gfd = fd_central_richardson(f1, T[0])
check("11.0 M=1: empty dJ/dq, dJ/dT matches FD", gq.shape == (0, 3) and abs(gT[0] - gfd) / abs(gfd) < 1e-6,
      f"gq{gq.shape} rel={abs(gT[0]-gfd)/abs(gfd):.2e}")

# degenerate collinear waypoints (straight line) -> gradients still finite & FD-correct
M = 4
line = np.linspace(0, 1, M + 1)[:, None] * np.array([2.0, -1.0, 0.5])
T = rng.uniform(0.6, 1.4, M)
tr = MinjerkTraj(line, T)
_, gq, gT = tr.energy_grad()
finite = np.all(np.isfinite(gq)) and np.all(np.isfinite(gT))
gfd_q = np.zeros((M - 1, 3)); gfd_T = np.zeros(M)
for i in range(M - 1):
    for d in range(3):
        def fq(val, i=i, d=d):
            w = line.copy(); w[i + 1, d] = val
            return MinjerkTraj(w, T).energy()
        gfd_q[i, d] = fd_central_richardson(fq, line[i + 1, d])
for k in range(M):
    def fT(val, k=k):
        Tv = T.copy(); Tv[k] = val
        return MinjerkTraj(line, Tv).energy()
    gfd_T[k] = fd_central_richardson(fT, T[k])
check("11.1 collinear waypoints: gradients finite & FD-correct", finite and rel_err(gq, gfd_q) < 1e-6 and rel_err(gT, gfd_T) < 1e-6,
      f"q rel={rel_err(gq,gfd_q):.2e} T rel={rel_err(gT,gfd_T):.2e}")

# tiny + large duration mix: gradient still FD-correct (condition-aware tol)
M = 3
wp = rng.standard_normal((M + 1, 3)); T = np.array([0.05, 2.0, 0.1])
tr = MinjerkTraj(wp, T)
_, gq, gT = tr.energy_grad()
gfd_T = np.zeros(M)
for k in range(M):
    def fT(val, k=k):
        Tv = T.copy(); Tv[k] = val
        return MinjerkTraj(wp, Tv).energy()
    gfd_T[k] = fd_central_richardson(fT, T[k])
check("11.2 mixed tiny/large durations: dJ/dT FD-correct", rel_err(gT, gfd_T) < 1e-5, f"rel={rel_err(gT,gfd_T):.2e}")


# ============================================================ 12. O(M) backprop (banded, not dense)
print("\n--- 12. backprop is O(M) banded (doubling ratio ~linear, not O(M^3)) ---")
import time
sizes = [200, 400, 800, 1600]
times = []
for M in sizes:
    wp = rng.standard_normal((M + 1, 3)); Tv = rng.uniform(0.5, 1.5, M)
    tr = MinjerkTraj(wp, Tv)
    t0 = time.perf_counter()
    for _ in range(3):
        tr.energy_grad()
    times.append((time.perf_counter() - t0) / 3)
ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]
avg_ratio = float(np.mean(ratios))
# O(M^3) would give ~8x per doubling; banded/linear stays well under 4x.
check("12.0 doubling ratio ~linear (avg < 4, rules out O(M^3))", avg_ratio < 4.0,
      f"ratios={[f'{r:.2f}' for r in ratios]} avg={avg_ratio:.2f}")
# adjoint uses solve_banded (no dense inverse): confirm by source + bandwidth
import inspect
src = inspect.getsource(MinjerkTraj._solve_adjoint)
uses_banded = "solve_banded" in src and "inv" not in src and "np.linalg.solve" not in src
check("12.1 _solve_adjoint uses solve_banded, no dense inverse", uses_banded)


# ============================================================ 13. gradient drives energy DOWN (descent sanity)
print("\n--- 13. analytic gradient is a true descent direction on J ---")
# Taking a small step along -grad must DECREASE J (first-order correctness end-to-end).
worst_inc = -1e18
ok_all = True
for _ in range(20):
    M = int(rng.integers(2, 6))
    wp, T, kw = make_traj(rng, M)
    tr = MinjerkTraj(wp, T, **kw)
    J0, gq, gT = tr.energy_grad()
    gnorm2 = float(np.sum(gq * gq) + np.sum(gT * gT))
    if gnorm2 < 1e-12:
        continue
    step = 1e-6 / max(1.0, np.sqrt(gnorm2))
    q_new = tr.q - step * gq
    T_new = T - step * gT
    if np.any(T_new <= 0):
        continue
    wp_new = wp.copy(); wp_new[1:-1] = q_new
    J1 = MinjerkTraj(wp_new, T_new, **kw).energy()
    # predicted decrease ~ step * gnorm2; require actual decrease and first-order agreement
    pred = step * gnorm2
    actual = J0 - J1
    if actual <= 0:
        ok_all = False
    worst_inc = max(worst_inc, abs(actual - pred) / max(abs(pred), 1e-9))
check("13.0 -grad step decreases J (20 random)", ok_all, "all strictly decreased")
check("13.1 first-order decrease matches step*||g||^2", worst_inc < 1e-3, f"worst rel={worst_inc:.2e}")


# ============================================================ summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=========================  {passed}/{total} passed  =========================")
if passed != total:
    print("Failures:")
    for n, ok, d in results:
        if not ok:
            print(f"  - {n}  ({d})")
    raise SystemExit(1)
