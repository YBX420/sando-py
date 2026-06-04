"""A+B 快版求解器(生产化第一步):解析 banded 梯度 + deadline 截断 anytime-feasible。
- B: time-aware 拓扑种子(按抵达时刻选过障侧、偏离最小的清空 y)
- Phase-1: 解析可行性恢复(梯度下降总违反量,解析 ∇g)
- A: fast feasible-direction —— 解析 cost 梯度(_minco_cost_grad)+ 解析约束 g/法向(_alm_constraints)
     + 一次性 dP/dq(固定 T,只动 q),每迭代解 QP 保持人体 g<=0;到 deadline 立即停,停在哪都可行。
被 _exp_production.py(闭环 + 密集 + 防御曲线)复用。
"""
import time
import numpy as np
import scipy.optimize
from sando_py.local.local_opt import (OptParams, MinjerkTraj, _minco_cost_grad,
                                       _alm_constraints, hard_clearance)


def _tr(start, goal, q, T, v0, a0, M):
    qq = q.reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    return MinjerkTraj.from_endpoints(start, goal, qq, T, v0=v0, a0=a0)


def time_aware_seed(start, goal, humans, M, T0, d_safe, time_aware=True, ywin=3.2):
    cum = np.concatenate([[0.0], np.cumsum(T0)])
    yg = np.linspace(-ywin, ywin, 161)
    q = []; prev = 0.0
    for i in range(M - 1):
        xb = start + (i + 1) / M * (goal - start)
        ti = cum[i + 1] if time_aware else 0.0
        cs = np.array([min(h.signed_dist(np.array([xb[0], y, start[2]]), ti) for h in humans) for y in yg])
        ok = yg[cs >= d_safe + 0.15]
        by = ok[np.argmin(np.abs(ok - prev))] if ok.size else yg[int(np.argmax(cs))]
        q.append([xb[0], by, start[2]]); prev = by
    return np.array(q).reshape(-1)


def _clr_per_human(start, goal, q, T, v0, a0, M, humans, K=120):
    tr = _tr(start, goal, q, T, v0, a0, M)
    ts = np.linspace(tr.t_start, tr.t_end, K); P = tr.eval(ts)
    return np.array([min(h.signed_dist(P[j], float(ts[j])) for j in range(K)) for h in humans])


def _dPdq(start, goal, q, T, v0, a0, M):
    """一次性:控制点 P(M,6,3) 对内部路点 q 的雅可比(固定 T,P 对 q 线性)。形状 (M*6*3, nq)。"""
    base = _tr(start, goal, q, T, v0, a0, M).control_points().reshape(-1)
    nq = q.size; J = np.zeros((base.size, nq)); eps = 1e-6
    for j in range(nq):
        qp = q.copy(); qp[j] += eps
        P = _tr(start, goal, qp, T, v0, a0, M).control_points().reshape(-1)
        J[:, j] = (P - base) / eps
    return J


def _cons_and_jac(start, goal, q, T, v0, a0, M, humans, cfg, opt, dPdq):
    """解析:活跃约束 g 与 ∇g/∇q。g_m=R-a^T(P_ik-c);dg/dq=-a^T dP_ik/dq。"""
    tr = _tr(start, goal, q, T, v0, a0, M)
    cons, P = _alm_constraints(tr, humans, cfg, opt=opt)
    g = cons["g"]; a = cons["n"]            # a: (Nc,3) 外法向 = -dg/dP
    H = len(humans)
    # 控制点 (i,k) 的 flat P 索引基 = (i*6+k)*3
    return g, a, H, tr


def fast_feasible_direction(start, goal, q0, T, v0, a0, M, humans, cfg, opt,
                            d_safe, iters=60, alpha=0.5, deadline_s=None, corridor=None):
    """从可行 q0 出发,每迭代保持人体 g<=0;deadline 一到停(停在哪都可行)。返回 (q, hist, ms, trunc)。"""
    t_start = time.perf_counter()
    nq = q0.size
    dPdq = _dPdq(start, goal, q0, T, v0, a0, M)          # 一次性
    q = q0.copy()
    hist = [float(_clr_per_human(start, goal, q, T, v0, a0, M, humans).min())]
    trunc = False
    Hn = len(humans)
    for _ in range(iters):
        if deadline_s is not None and (time.perf_counter() - t_start) >= deadline_s:
            trunc = True; break
        # 解析 cost 梯度(只取 q 部分)+ 可选走廊
        x = np.concatenate([q, T])
        f, gx = _minco_cost_grad(x, start, goal, M, [], cfg, opt, float(np.sum(T)), v0=v0, a0=a0)
        gf = gx[:nq].copy()
        if corridor is not None:
            wc, R, ctr = corridor
            qm = q.reshape(M - 1, 3)
            for i in range(M - 1):
                dd = qm[i] - ctr[i]; dn = np.linalg.norm(dd)
                if dn > R:
                    gf[3 * i:3 * i + 3] += 2.0 * wc * (dn - R) / max(dn, 1e-9) * dd
                    f += wc * (dn - R) ** 2
        # 解析约束 g + 法向
        tr = _tr(start, goal, q, T, v0, a0, M)
        cons, _ = _alm_constraints(tr, humans, cfg, opt=opt)
        g = cons["g"]; a = cons["n"]
        active = np.where(g > -0.2)[0]
        Gm = []; bm = []
        for m in active:
            i = m // (6 * Hn); k = (m % (6 * Hn)) // Hn
            base3 = (i * 6 + k) * 3
            row = -(a[m, 0] * dPdq[base3] + a[m, 1] * dPdq[base3 + 1] + a[m, 2] * dPdq[base3 + 2])
            Gm.append(row); bm.append(-alpha * g[m])
        cons_qp = []
        if Gm:
            Gm = np.array(Gm); bm = np.array(bm)
            cons_qp = [{"type": "ineq", "fun": (lambda u, Gi=Gm[r], bi=bm[r]: bi - Gi @ u)}
                       for r in range(len(Gm))]
        u0 = -gf / (np.linalg.norm(gf) + 1e-9)
        res = scipy.optimize.minimize(lambda u: 0.5 * float(np.dot(u + gf, u + gf)), u0,
                                      jac=lambda u: u + gf, constraints=cons_qp, method="SLSQP",
                                      bounds=[(-4, 4)] * nq, options={"maxiter": 25, "ftol": 1e-6})
        u = res.x
        if np.linalg.norm(u) < 1e-7:
            break
        # 回溯:用【采样连续时间净空】判可行(对移动行人 sound;控制点 g 不 sound)。
        t = 1.0; moved = False
        for _bt in range(12):
            qn = q + t * u
            if _clr_per_human(start, goal, qn, T, v0, a0, M, humans, K=50).min() >= d_safe - 1e-9:
                xn = np.concatenate([qn, T])
                fn, _ = _minco_cost_grad(xn, start, goal, M, [], cfg, opt, float(np.sum(T)), v0=v0, a0=a0)
                if fn <= f + 1e-9:
                    q = qn; moved = True; break
            t *= 0.5
        hist.append(float(_clr_per_human(start, goal, q, T, v0, a0, M, humans, K=50).min()))
        if not moved:
            break
    ms = (time.perf_counter() - t_start) * 1000.0
    return q, np.array(hist), ms, trunc


def restore_feasible(start, goal, q0, T, v0, a0, M, humans, cfg, opt, d_safe, iters=200):
    """解析可行性恢复:梯度下降总违反量 V=Σmax(0,d_safe-clr_h)^2 直到 min_clr>=d_safe。"""
    def V(q):
        c = _clr_per_human(start, goal, q, T, v0, a0, M, humans)
        v = np.maximum(d_safe - c, 0.0); return float((v * v).sum())
    q = q0.copy(); nq = q.size
    for _ in range(iters):
        c = _clr_per_human(start, goal, q, T, v0, a0, M, humans)
        if c.min() >= d_safe:
            return q, True
        v0v = V(q); g = np.zeros(nq)
        for j in range(nq):
            qp = q.copy(); qp[j] += 1e-4; g[j] = (V(qp) - v0v) / 1e-4
        ng = np.linalg.norm(g)
        if ng < 1e-9:
            break
        d = -g / ng; t = 0.6; stp = False
        for _bt in range(34):
            qn = q + t * d
            if V(qn) < v0v - 1e-12:
                q = qn; stp = True; break
            t *= 0.5
        if not stp:
            break
    return q, _clr_per_human(start, goal, q, T, v0, a0, M, humans).min() >= d_safe
