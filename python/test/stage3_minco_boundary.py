"""Stage 3 — local/minco.py BOUNDARY / edge cases (MINCO min-jerk, s=3).

Run:  python3 test/stage3_minco_boundary.py

Coverage:
  A. M=1 single segment (no interior waypoints) — solve + endpoints + min-jerk
  B. M=2 (one interior waypoint) — interpolation + continuity
  C. tiny durations (1e-3) — system still solvable, waypoints exact
  D. large durations (1e3) — same
  E. mixed tiny+large durations in one trajectory
  F. nonzero v0/a0 only at start (vf/af zero)
  G. degenerate collinear waypoints (straight line) — exact, no NaN
  H. coincident consecutive waypoints (zero-length geometric step) — still solves
  I. input validation: wrong shapes / nonpositive T / too few waypoints raise
  J. out-of-domain t clips to [t_start, t_end] (bspline parity)
  K. scalar vs vector eval agree; eval == eval_deriv(.,0)
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

def raises(fn, exc=Exception):
    try:
        fn(); return False
    except exc:
        return True


# ---------------------------------------------------------------- A. M=1
print("\n--- A. M=1 single segment ---")
p0 = np.array([0.,0,0]); pf = np.array([2.,-1,3])
tr = MinjerkTraj(np.vstack([p0, pf]), [1.5])
check("A.0 M=1 t_end == sum(T)", abs(tr.t_end - 1.5) < 1e-15)
check("A.1 M=1 start exact", np.allclose(tr.eval_deriv(0.,0), p0, atol=1e-12))
check("A.2 M=1 goal exact", np.allclose(tr.eval_deriv(1.5,0), pf, atol=1e-12))
s = np.linspace(0,1,11)
oracle = p0 + (pf-p0)*(10*s**3-15*s**4+6*s**5)[:,None]
check("A.3 M=1 == closed-form min-jerk", np.allclose(tr.eval_deriv(s*1.5,0), oracle, atol=1e-9))
check("A.4 M=1 coeff shape (6,3)", tr.c.shape == (6,3))
check("A.5 M=1 dense_M is 6x6", tr.dense_M().shape == (6,6))


# ---------------------------------------------------------------- B. M=2
print("\n--- B. M=2 one interior waypoint ---")
wp = np.array([[0.,0,0],[1.,1,1],[2.,0,2]])
tr = MinjerkTraj(wp, [1.0, 1.2])
check("B.0 interior waypoint exact", np.allclose(tr.eval_deriv(1.0,0), wp[1], atol=1e-9))
# continuity at the single junction, orders 0..4 (exact coeff compare)
def deriv_at(c6, tau, o):
    row = np.zeros(6)
    for j in range(o,6):
        cf=1.0
        for k in range(o): cf*=(j-k)
        row[j]=cf*(tau**(j-o))
    return row@c6
cL = tr.c[0:6]; cR = tr.c[6:12]
gap = max(np.max(np.abs(deriv_at(cL,1.0,o)-deriv_at(cR,0.0,o))) for o in range(5))
check("B.1 C0..C4 continuous at the junction", gap < 1e-9, f"gap={gap:.2e}")


# ---------------------------------------------------------------- C/D/E. duration scaling
print("\n--- C/D/E. tiny / large / mixed durations ---")
wp = np.array([[0.,0,0],[1.,2,0],[3.,1,1],[4.,0,2]])
# Tolerance scales with the conditioning of M(T): the position rows carry T^5,
# so a duration ratio r between segments makes cond(M)~r^5 and float64 keeps
# only ~ (16 - 5*log10 r) digits. Tiny/large uniform cases are well-conditioned
# (machine eps); the mixed 1e-3..1e3 case (r=1e6) legitimately loses ~12 digits.
for tag, T in [("C tiny", np.array([1e-3,1e-3,1e-3])),
               ("D large", np.array([1e3,1e3,1e3])),
               ("E mixed", np.array([1e-3,5.0,1e3]))]:
    tr = MinjerkTraj(wp, T)
    cum = np.concatenate([[0.0], np.cumsum(T)])
    finite = np.all(np.isfinite(tr.c))
    werr = max(np.max(np.abs(tr.eval_deriv(cum[i],0)-wp[i])) for i in range(len(wp)))
    cond = float(np.linalg.cond(tr.dense_M()))
    tol = max(1e-9, 10.0 * cond * np.finfo(float).eps)  # condition-aware
    check(f"{tag}: finite coeffs", finite)
    check(f"{tag}: waypoints interpolated (cond={cond:.1e})", werr < tol,
          f"werr={werr:.2e}, tol={tol:.2e}")


# ---------------------------------------------------------------- F. nonzero start derivs only
print("\n--- F. nonzero v0/a0 at start only ---")
wp = np.array([[0.,0,0],[2.,1,0],[4.,0,1]])
v0 = np.array([1.5,-0.5,0.2]); a0 = np.array([-0.3,0.1,0.4])
tr = MinjerkTraj(wp, [1.0,1.0], v0=v0, a0=a0)
check("F.0 v(0)=v0", np.allclose(tr.eval_deriv(0.,1), v0, atol=1e-9))
check("F.1 a(0)=a0", np.allclose(tr.eval_deriv(0.,2), a0, atol=1e-9))
check("F.2 v(T)=0 (default)", np.allclose(tr.eval_deriv(tr.t_end,1), 0., atol=1e-9))
check("F.3 a(T)=0 (default)", np.allclose(tr.eval_deriv(tr.t_end,2), 0., atol=1e-9))


# ---------------------------------------------------------------- G. collinear waypoints
print("\n--- G. degenerate collinear waypoints (straight line) ---")
t = np.linspace(0,1,6)[:,None]
wp = np.array([0.,0,0]) + t*np.array([5.,2,-3])  # exactly collinear
T = np.full(5, 0.8)
tr = MinjerkTraj(wp, T)
cum = np.concatenate([[0.0], np.cumsum(T)])
werr = max(np.max(np.abs(tr.eval_deriv(cum[i],0)-wp[i])) for i in range(len(wp)))
samp = tr.eval_deriv(np.linspace(0,tr.t_end,50),0)
# all samples must lie on the line through wp[0],wp[-1]
dirv = wp[-1]-wp[0]; dirv/=np.linalg.norm(dirv)
offaxis = samp - wp[0] - np.outer((samp-wp[0])@dirv, dirv)
check("G.0 collinear: no NaN", np.all(np.isfinite(tr.c)))
check("G.1 collinear: waypoints exact", werr < 1e-9, f"werr={werr:.2e}")
check("G.2 collinear: stays on the line", float(np.max(np.abs(offaxis))) < 1e-9,
      f"max offaxis={np.max(np.abs(offaxis)):.2e}")


# ---------------------------------------------------------------- H. coincident waypoints
print("\n--- H. coincident consecutive waypoints ---")
wp = np.array([[0.,0,0],[1.,1,1],[1.,1,1],[2.,2,0]])  # wp[1]==wp[2]
T = np.array([1.0,0.7,1.0])
tr = MinjerkTraj(wp, T)
cum = np.concatenate([[0.0], np.cumsum(T)])
werr = max(np.max(np.abs(tr.eval_deriv(cum[i],0)-wp[i])) for i in range(len(wp)))
check("H.0 coincident: solvable, finite", np.all(np.isfinite(tr.c)))
check("H.1 coincident: waypoints still exact", werr < 1e-9, f"werr={werr:.2e}")


# ---------------------------------------------------------------- I. input validation
print("\n--- I. input validation ---")
check("I.0 wrong waypoint shape raises",
      raises(lambda: MinjerkTraj(np.zeros((4,2)), [1,1,1])))
check("I.1 mismatched M raises",
      raises(lambda: MinjerkTraj(np.zeros((4,3)), [1,1])))   # 4 pts need 3 durations
check("I.2 nonpositive duration raises",
      raises(lambda: MinjerkTraj(np.zeros((3,3)), [1.0, -0.5])))
check("I.3 zero duration raises",
      raises(lambda: MinjerkTraj(np.zeros((2,3)), [0.0])))
check("I.4 zero segments raises",
      raises(lambda: MinjerkTraj(np.zeros((1,3)), [])))
check("I.5 negative order raises",
      raises(lambda: MinjerkTraj(np.zeros((2,3)), [1.0]).eval_deriv(0.5, -1)))


# ---------------------------------------------------------------- J. domain guard
# 契约变更:域外超 1e-6 浮点噪声 -> 主动 raise(暴露 A_time/段时长 misalignment),不再静默夹回。
# 1e-6 以内的越界(t_end+ε 这类良性浮点噪声)仍夹回端点,给正常采样留活路。
print("\n--- J. out-of-domain t raises; float-noise overshoot still clips ---")
wp = np.array([[0.,0,0],[1.,1,1],[2.,0,2]])
tr = MinjerkTraj(wp, [1.0,1.0])
check("J.0 t<<t_start raises", raises(lambda: tr.eval_deriv(-3.0,0), ValueError))
check("J.1 t>>t_end raises", raises(lambda: tr.eval_deriv(99.0,0), ValueError))
check("J.2 vectorised OOB raises", raises(lambda: tr.eval_deriv(np.array([0.5, 99.0]),0), ValueError))
check("J.3 t_end+1e-9 noise still clips", np.allclose(tr.eval_deriv(tr.t_end+1e-9,0), tr.eval_deriv(tr.t_end,0)))
check("J.4 t_start-1e-9 noise still clips", np.allclose(tr.eval_deriv(-1e-9,0), tr.eval_deriv(0.0,0)))
check("J.5 deriv(order>5)==0", np.allclose(tr.eval_deriv(np.linspace(0,2,5),6), 0.0))


# ---------------------------------------------------------------- K. scalar/vector parity
print("\n--- K. scalar/vector eval parity ---")
ts = np.linspace(0, tr.t_end, 13)
vec = tr.eval_deriv(ts, 0)
pt = np.array([tr.eval_deriv(float(t),0) for t in ts])
check("K.0 batched == pointwise", np.allclose(vec, pt))
check("K.1 scalar returns (3,)", tr.eval_deriv(0.5,0).shape == (3,))
check("K.2 vector returns (K,3)", vec.shape == (13,3))
check("K.3 eval == eval_deriv(.,0)", np.allclose(tr.eval(ts), tr.eval_deriv(ts,0)))


# ------------------------------------------------ L. ADVERSARIAL (red-team M0-VERIFY)
# These probe the headline indexing/tau-power risk with checks the original build
# did not have: (1) MINCO == TRUE min-jerk QP where C3/C4 are NOT imposed but must
# emerge from KKT optimality; (2) nonzero v0/a0 on a 3-segment ASYMMETRIC-duration
# trajectory; (3) C0..C4 continuity at 1e-9 on that asymmetric case.
print("\n--- L. ADVERSARIAL (red-team) ---")
from math import factorial

def _dbasis(Tv, o):
    r = np.zeros(6)
    for j in range(o, 6):
        r[j] = factorial(j) / factorial(j - o) * (Tv ** (j - o))
    return r

def _jerk_hessian(Tv):
    H = np.zeros((6, 6))
    terms = {3: (0, 6.0), 4: (1, 24.0), 5: (2, 60.0)}
    for a, (pa, ca) in terms.items():
        for b_, (pb, cb) in terms.items():
            p = pa + pb
            H[a, b_] = ca * cb * (Tv ** (p + 1)) / (p + 1)
    return H

def _true_minjerk_qp(wp, T, v0, a0, vf, af):
    """Min sum_i int (p_i''')^2 s.t. ONLY head/tail p,v,a + waypoint pins + C0/C1/C2.
    C3 (jerk) and C4 (snap) continuity are deliberately NOT constrained; if MINCO is
    genuinely minimum-jerk they emerge as KKT conditions and the QP coeffs == MINCO c."""
    M = T.size; n = 6 * M
    H = np.zeros((n, n))
    for i in range(M):
        H[6 * i:6 * i + 6, 6 * i:6 * i + 6] = _jerk_hessian(T[i])
    rows = []; d = []
    def addrow(blocks, rhs):
        r = np.zeros(n)
        for col, vec in blocks:
            r[col:col + 6] += vec
        rows.append(r); d.append(rhs)
    addrow([(0, _dbasis(0, 0))], wp[0]); addrow([(0, _dbasis(0, 1))], v0)
    addrow([(0, _dbasis(0, 2))], a0)
    for i in range(M - 1):
        bs = 6 * i
        addrow([(bs, _dbasis(T[i], 0))], wp[1 + i])
        addrow([(bs, _dbasis(T[i], 0)), (bs + 6, -_dbasis(0, 0))], np.zeros(3))
        addrow([(bs, _dbasis(T[i], 1)), (bs + 6, -_dbasis(0, 1))], np.zeros(3))
        addrow([(bs, _dbasis(T[i], 2)), (bs + 6, -_dbasis(0, 2))], np.zeros(3))
    lt = M - 1; bs = 6 * lt
    addrow([(bs, _dbasis(T[lt], 0))], wp[-1]); addrow([(bs, _dbasis(T[lt], 1))], vf)
    addrow([(bs, _dbasis(T[lt], 2))], af)
    C = np.array(rows); d = np.array(d); m = C.shape[0]
    KKT = np.block([[H, C.T], [C, np.zeros((m, m))]])
    sol = np.zeros((n, 3))
    for dim in range(3):
        rhs = np.concatenate([np.zeros(n), d[:, dim]])
        sol[:, dim] = np.linalg.solve(KKT, rhs)[:n]
    return sol

rng = np.random.default_rng(2024)
worst_qp = 0.0
for _ in range(15):
    M = int(rng.integers(2, 6))
    wp_ = rng.standard_normal((M + 1, 3)); T_ = rng.uniform(0.4, 2.0, M)
    v0_ = rng.standard_normal(3) * 0.3; a0_ = rng.standard_normal(3) * 0.3
    vf_ = rng.standard_normal(3) * 0.3; af_ = rng.standard_normal(3) * 0.3
    trq = MinjerkTraj(wp_, T_, v0=v0_, a0=a0_, vf=vf_, af=af_)
    c_qp = _true_minjerk_qp(wp_, T_, v0_, a0_, vf_, af_)
    worst_qp = max(worst_qp, np.max(np.abs(c_qp - trq.c)))
check("L.0 MINCO == TRUE min-jerk QP (C3/C4 emerge from KKT)", worst_qp < 1e-7,
      f"worst={worst_qp:.2e}")

# nonzero BCs on a strongly ASYMMETRIC 3-segment trajectory
M = 3
wp_a = np.array([[0., 0, 0], [1.5, 0.5, 0.2], [3.0, -1.0, 0.8], [5., -2., 1.]])
T_a = np.array([0.3, 1.7, 0.5])           # asymmetric
v0_a = np.array([0.7, -0.3, 0.1]); a0_a = np.array([-0.4, 0.2, 0.05])
tra = MinjerkTraj(wp_a, T_a, v0=v0_a, a0=a0_a)
e_head = max(np.max(np.abs(tra.eval_deriv(0., 0) - wp_a[0])),
             np.max(np.abs(tra.eval_deriv(0., 1) - v0_a)),
             np.max(np.abs(tra.eval_deriv(0., 2) - a0_a)))
check("L.1 asymmetric+nonzero v0/a0: head p/v/a exact", e_head < 1e-9, f"err={e_head:.2e}")
cum_a = np.concatenate([[0.0], np.cumsum(T_a)])
e_wp = max(np.max(np.abs(tra.eval_deriv(cum_a[k + 1], 0) - wp_a[1 + k])) for k in range(M - 1))
check("L.2 asymmetric: 3-seg exact waypoint interp", e_wp < 1e-9, f"err={e_wp:.2e}")
e_cont = 0.0
for k in range(M - 1):
    cL_ = tra.c[6 * k:6 * k + 6]; cR_ = tra.c[6 * (k + 1):6 * (k + 1) + 6]
    for o in range(5):
        e_cont = max(e_cont, np.max(np.abs(_dbasis(T_a[k], o) @ cL_ - _dbasis(0.0, o) @ cR_)))
check("L.3 asymmetric: C0..C4 continuity", e_cont < 1e-9, f"err={e_cont:.2e}")

# _locate must not be off-by-one around junctions or at t_end
trl = MinjerkTraj(np.random.default_rng(1).standard_normal((5, 3)),
                  np.array([0.5, 1.0, 0.7, 1.3]))
cum_l = np.concatenate([[0.0], np.cumsum(trl.T)])
ok_loc = (trl._locate(trl.t_end)[0] == trl.M - 1)
for k in range(1, trl.M):
    ok_loc = ok_loc and (trl._locate(cum_l[k] - 1e-9)[0] == k - 1) \
                    and (trl._locate(cum_l[k] + 1e-9)[0] == k)
check("L.4 _locate no off-by-one at junctions / t_end", ok_loc)


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
