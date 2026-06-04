"""Stage 3 — local/minco.py INVARIANTS (MINCO min-jerk, s=3).

Run:  python3 test/stage3_minco_invariants.py

Coverage:
  A. C^0..C^4 continuity across EVERY junction to <= 1e-9 (one-sided limits)
  B. EXACT waypoint interpolation at each junction time == q_i (MINCO property)
  C. boundary derivatives honored: v(0)=v0, a(0)=a0, v(T)=vf, a(T)=af
  D. derivatives vs central finite differences (orders 1..4) well inside segments
  E. cross-check vs bspline.py: same waypoints, MINCO samples vs UniformBSpline
     fit_path samples agree by CHORD distance (coarse geometric sanity, not exact)
  F. uniqueness: rebuilding with the same (q,T) gives identical coefficients
"""
import importlib.util
import os
import numpy as np

HERE = os.path.dirname(__file__)
def _load(name, rel):
    p = os.path.abspath(os.path.join(HERE, rel))
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

minco = _load("minco_standalone", "../sando_py/local/minco.py")
bspline = _load("bspline_standalone", "../sando_py/local/bspline.py")
MinjerkTraj = minco.MinjerkTraj
UniformBSpline = bspline.UniformBSpline

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- A. C^0..C^4 continuity (exact)
# Continuity is a PROPERTY OF THE TWO ADJACENT POLYNOMIALS, so it must be checked
# from the exact segment coefficients (end of segment i == start of segment i+1
# for every derivative order), NOT by a finite-h probe across the junction. A
# finite-h probe of a high-order derivative measures its genuine change over the
# probe interval (O(h * next_deriv)) and would falsely "fail" jerk/snap even
# though the trajectory is perfectly C^4.
rng = np.random.default_rng(11)
print("\n--- A. continuity via exact polynomial coefficients (strict 1e-9) ---")
def deriv_at(c6, tau, order):
    row = np.zeros(6)
    for j in range(order, 6):
        cf = 1.0
        for k in range(order):
            cf *= (j - k)
        row[j] = cf * (tau ** (j - order))
    return row @ c6
worstAp = {o: 0.0 for o in range(5)}
for trial in range(10):
    M = int(rng.integers(2, 8))
    wp = rng.standard_normal((M+1, 3)) * 4
    T = rng.uniform(0.3, 2.5, size=M)
    tr = MinjerkTraj(wp, T)
    for i in range(M-1):
        cL = tr.c[6*i:6*i+6]
        cR = tr.c[6*(i+1):6*(i+1)+6]
        for o in range(5):
            endL = deriv_at(cL, T[i], o)
            startR = deriv_at(cR, 0.0, o)
            worstAp[o] = max(worstAp[o], float(np.max(np.abs(endL - startR))))
for o in range(5):
    nm = {0:"C0 pos",1:"C1 vel",2:"C2 acc",3:"C3 jerk",4:"C4 snap"}[o]
    check(f"A.{o} {nm} exact junction match <=1e-9", worstAp[o] < 1e-9, f"worst={worstAp[o]:.2e}")


# ---------------------------------------------------------------- B. waypoint interpolation
print("\n--- B. exact waypoint interpolation ---")
worstB = 0.0
for trial in range(10):
    M = int(rng.integers(2, 8))
    wp = rng.standard_normal((M+1, 3)) * 4
    T = rng.uniform(0.3, 2.5, size=M)
    tr = MinjerkTraj(wp, T)
    cum = np.concatenate([[0.0], np.cumsum(T)])
    for i in range(1, M):
        worstB = max(worstB, float(np.max(np.abs(tr.eval_deriv(cum[i], 0) - wp[i]))))
check("B.0 interior waypoints interpolated exactly <=1e-9", worstB < 1e-9, f"worst={worstB:.2e}")


# ---------------------------------------------------------------- C. boundary conditions
print("\n--- C. boundary derivative conditions ---")
worstC = 0.0
for trial in range(8):
    M = int(rng.integers(1, 6))
    wp = rng.standard_normal((M+1, 3)) * 3
    T = rng.uniform(0.4, 2.0, size=M)
    v0 = rng.standard_normal(3); a0 = rng.standard_normal(3)
    vf = rng.standard_normal(3); af = rng.standard_normal(3)
    tr = MinjerkTraj(wp, T, v0=v0, a0=a0, vf=vf, af=af)
    e = max(
        np.max(np.abs(tr.eval_deriv(0.0,0) - wp[0])),
        np.max(np.abs(tr.eval_deriv(tr.t_end,0) - wp[-1])),
        np.max(np.abs(tr.eval_deriv(0.0,1) - v0)),
        np.max(np.abs(tr.eval_deriv(0.0,2) - a0)),
        np.max(np.abs(tr.eval_deriv(tr.t_end,1) - vf)),
        np.max(np.abs(tr.eval_deriv(tr.t_end,2) - af)),
    )
    worstC = max(worstC, float(e))
check("C.0 head/tail pos+vel+acc honored <=1e-9", worstC < 1e-9, f"worst={worstC:.2e}")


# ---------------------------------------------------------------- D. derivatives vs central diff
print("\n--- D. eval_deriv vs central finite differences ---")
M = 5
wp = rng.standard_normal((M+1, 3)) * 3
T = rng.uniform(0.6, 1.8, size=M)
tr = MinjerkTraj(wp, T)
cum = np.concatenate([[0.0], np.cumsum(T)])
# sample strictly inside segments (avoid junctions where higher derivs jump)
samp = []
for i in range(M):
    samp.extend(np.linspace(cum[i]+0.15*T[i], cum[i+1]-0.15*T[i], 4))
samp = np.array(samp)
h = 1e-4
for order in (1, 2, 3, 4):
    fd = np.zeros((len(samp), 3))
    for k, t in enumerate(samp):
        if order == 1:
            fd[k] = (tr.eval_deriv(t+h,0) - tr.eval_deriv(t-h,0)) / (2*h)
        elif order == 2:
            fd[k] = (tr.eval_deriv(t+h,0) - 2*tr.eval_deriv(t,0) + tr.eval_deriv(t-h,0)) / h**2
        elif order == 3:
            fd[k] = (tr.eval_deriv(t+2*h,0) - 2*tr.eval_deriv(t+h,0)
                     + 2*tr.eval_deriv(t-h,0) - tr.eval_deriv(t-2*h,0)) / (2*h**3)
        elif order == 4:
            fd[k] = (tr.eval_deriv(t+2*h,0) - 4*tr.eval_deriv(t+h,0) + 6*tr.eval_deriv(t,0)
                     - 4*tr.eval_deriv(t-h,0) + tr.eval_deriv(t-2*h,0)) / h**4
    analytic = tr.eval_deriv(samp, order)
    err = float(np.max(np.abs(fd - analytic)))
    tol = {1:1e-5, 2:1e-3, 3:5e-1, 4:5e2}[order]
    check(f"D.{order} eval_deriv(order={order}) ~ central diff", err < tol, f"max|err|={err:.2e}, tol={tol:.0e}")


# ---------------------------------------------------------------- E. cross-check vs bspline geometry
print("\n--- E. cross-check vs bspline.py (chord-distance sanity) ---")
# same waypoint set; MINCO interpolates them, the B-spline LSQ-fits a polyline
# through them. They are different curves, but should track each other to within
# a fraction of the path scale (coarse geometric sanity, NOT exact).
wpE = np.array([
    [0.,0,0],[1.,0.5,0],[2.,0.0,0.5],[3.,1.0,0.5],[4.,0.5,1.0],[5.,0.0,1.0]
])
M = wpE.shape[0]-1
T = np.full(M, 1.0)
tr = MinjerkTraj(wpE, T)
bs = UniformBSpline.fit_path(wpE, num_ctrl=12, degree=5, dt=1.0, clamp_endpoints=True)
# sample both, normalized parameter
sm = tr.eval_deriv(np.linspace(tr.t_start, tr.t_end, 200), 0)
sb = bs.eval(np.linspace(bs.t_start, bs.t_end, 200))
# nearest-point (chord) distance from each MINCO sample to the bspline polyline
def chord_dist(P, Q):
    # min distance from each point in P to polyline through Q
    out = []
    for p in P:
        seg = Q[:-1]; nxt = Q[1:]
        d = nxt - seg
        L2 = np.sum(d*d, axis=1)
        L2[L2 == 0] = 1e-12
        u = np.clip(np.sum((p - seg)*d, axis=1)/L2, 0, 1)
        proj = seg + u[:,None]*d
        out.append(np.min(np.linalg.norm(proj - p, axis=1)))
    return np.array(out)
cd = chord_dist(sm, sb)
scale = float(np.linalg.norm(wpE[-1] - wpE[0]))
check("E.0 MINCO tracks bspline geometry (max chord < 0.3*scale)",
      float(np.max(cd)) < 0.3*scale, f"max chord={np.max(cd):.3f}, scale={scale:.2f}")
# both must pass through the shared endpoints
check("E.1 both share endpoints",
      np.allclose(tr.eval_deriv(tr.t_start,0), wpE[0], atol=1e-9) and
      np.allclose(bs.eval(bs.t_start), wpE[0], atol=1e-9))


# ---------------------------------------------------------------- F. uniqueness / determinism
print("\n--- F. uniqueness of the min-jerk solution ---")
wpF = rng.standard_normal((6, 3)) * 2
TF = rng.uniform(0.5, 2.0, size=5)
c1 = MinjerkTraj(wpF, TF).c
c2 = MinjerkTraj(wpF, TF).c
check("F.0 same (q,T) -> identical coefficients", np.array_equal(c1, c2))


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
