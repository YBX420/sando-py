"""MINCO minimum-jerk (s=3, quintic) trajectory backbone.

Replacement for the B-spline backbone of the local optimizer. MINCO
(Wang et al., IEEE T-RO 2022, GCOPTER framework) parameterizes a piecewise
quintic by intermediate waypoints q and segment durations T; given (q, T) the
minimum-jerk trajectory is UNIQUE and is recovered by solving a sparse banded
linear system  M(T) c = b(q)  (one row per scalar constraint, 6M rows).

Conventions (mirrors GCOPTER MINCO_S3NU exactly):
  M segments, segment i is  p_i(t) = sum_{j=0..5} c[6i+j] * t^j,  t in [0, T_i]
  coefficients c are stored ASCENDING in power (c[6i+0] is the constant term).
  banded system has lower/upper bandwidth 6.

Constraint row layout (6M rows total):
  rows 0,1,2          : p(0)=p_start, v(0)=v0, a(0)=a0      (head block)
  per junction i in 1..M-1  (loop var i = 0..M-2):
    6i+3 : C^3 (jerk)  continuity   p_i'''(T_i)  = p_{i+1}'''(0)
    6i+4 : C^4 (snap)  continuity   p_i''''(T_i) = p_{i+1}''''(0)
    6i+5 : p_i(T_i) = q_i           (EXACT waypoint interpolation)
    6i+6 : C^0 (pos)   continuity   p_i(T_i)  = p_{i+1}(0)
    6i+7 : C^1 (vel)   continuity   p_i'(T_i) = p_{i+1}'(0)
    6i+8 : C^2 (acc)   continuity   p_i''(T_i)= p_{i+1}''(0)
  rows 6M-3,-2,-1     : p(T_M)=p_goal, v=vf, a=af           (tail block)

eval_deriv(ts, order) mirrors UniformBSpline.eval_deriv: order 0 = position.

================================ 中文说明 ================================
这个文件是新规划器真正的"轨迹底座" (取代了原来的 B 样条 bspline.py)。

它是什么 (大白话):
  MINCO (Wang 等, IEEE T-RO 2022) 是一种参数化轨迹的聪明办法。你只需要给两样东西:
    - 一串中间途经点 q (轨迹要穿过的点)
    - 每一段的时长 T (走完每段花多少时间)
  给定 (q, T),那条"加加速度 (jerk,即加速度的变化率) 平方积分最小"的五次多项式
  轨迹就被唯一确定了。换句话说,曲线的全部系数 c 不是自由变量,而是由 (q, T)
  通过解一个线性方程组 M(T) c = b(q) 自动算出来的。

为什么这样设计 (它在规划器里的角色):
  优化器只需要去调 q 和 T 这两组少量变量,曲线的平滑性自动保证。而且这个映射
  (q,T) -> c 的梯度可以解析地反传 (见文件后半段的 M1 伴随法),所以优化器能拿到
  精确梯度,跑得又快又准。人 (硬约束) 和墙 (软场) 的避障代价都加在这条轨迹上。

输入/输出:
  输入 = 途经点 q + 各段时长 T (+ 可选的起末速度/加速度边界)。
  输出 = 一条能 eval (求位置) / eval_deriv (求各阶导) 的连续轨迹;
         control_points() 还能给出 Bernstein 控制点,用于"整段曲线都在凸包内"
         这种连续时间的安全证明 (避障硬约束就靠它)。

下面几个英文段落讲的是数学细节 (线性方程组每一行对应什么约束、系数怎么排),
逐函数我再用中文补充关键点;术语我尽量当场解释。
=========================================================================
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# Power -> Bernstein conversion for a degree-5 polynomial on [0, 1].
# A degree-5 polynomial p(tau) = sum_{j=0..5} a_j tau^j on tau in [0,1] equals
# sum_{k=0..5} P_k Bernstein_k(tau), Bernstein_k = C(5,k) tau^k (1-tau)^(5-k).
# The control points are P = C2B @ a with the lower-triangular constant matrix
#   C2B[k, j] = C(k, j) / C(5, j)   for j <= k else 0
# (verified eval-equivalent to 3e-15; P_0 = a_0 = p(0), P_5 = sum a_j = p(1)).
# Bernstein control points satisfy the convex-hull property: the whole curve
# lies in conv{P_k}, which is the continuous-time clearance certificate basis.
# ---------------------------------------------------------------------------
# C2B:把"幂级数系数"换算成"Bernstein 控制点"的常数矩阵 (一段五次曲线用)。
# 关键好处 —— 凸包性质:整段曲线一定落在它这 6 个 Bernstein 控制点围成的凸包里。
# 所以只要让这些控制点都离障碍够远,整段连续曲线就保证安全,不用逐点采样去查。
# 这就是人 (硬约束) 那套"凸包+ALM"避障的几何基础。
C2B = (1.0 / 60.0) * np.array([
    [60.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [60.0, 12.0,  0.0,  0.0,  0.0,  0.0],
    [60.0, 24.0,  6.0,  0.0,  0.0,  0.0],
    [60.0, 36.0, 18.0,  6.0,  0.0,  0.0],
    [60.0, 48.0, 36.0, 24.0, 12.0,  0.0],
    [60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
], dtype=np.float64)


class MinjerkTraj:
    """Minimum-jerk MINCO trajectory, s=3 (quintic segments), 3D.

    Build from full waypoints (start, intermediates..., goal) and per-segment
    durations. Boundary derivatives (v0,a0 at start, vf,af at goal) default to
    zero; pass arrays to override.

    最小 jerk 的 MINCO 轨迹,3D,每段是五次多项式。
    用法:给"完整途经点 (起点, 中间点..., 终点)"和"每段时长",就建好了。
    起点/终点的速度、加速度默认是 0 (停稳起步、停稳到达),想要别的就传进来。
    """

    DEG = 5          # quintic  每段是五次 (degree 5) 多项式
    NC = 6           # coefficients per segment  五次多项式有 6 个系数

    def __init__(
        self,
        waypoints_full: np.ndarray,
        durations: np.ndarray,
        v0: Optional[np.ndarray] = None,
        a0: Optional[np.ndarray] = None,
        vf: Optional[np.ndarray] = None,
        af: Optional[np.ndarray] = None,
    ):
        wp = np.asarray(waypoints_full, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] != 3:
            raise ValueError(f"waypoints_full must be (M+1, 3), got {wp.shape}")
        T = np.asarray(durations, dtype=np.float64).reshape(-1)
        M = T.size
        if wp.shape[0] != M + 1:
            raise ValueError(
                f"need M+1 waypoints for M durations: got {wp.shape[0]} pts, {M} durations"
            )
        if M < 1:
            raise ValueError("need at least one segment")
        if np.any(T <= 0.0):
            raise ValueError("all durations T_i must be > 0")

        self.M = M                # 段数
        self.d = 3
        self.T = T                # 每段时长,长度 M
        self.waypoints = wp
        self.p_start = wp[0].copy()
        self.p_goal = wp[-1].copy()
        self.q = wp[1:-1].copy()  # (M-1, 3) interior waypoints  去掉起终点的中间途经点

        z = np.zeros(3)
        self.v0 = z.copy() if v0 is None else np.asarray(v0, dtype=np.float64).reshape(3)
        self.a0 = z.copy() if a0 is None else np.asarray(a0, dtype=np.float64).reshape(3)
        self.vf = z.copy() if vf is None else np.asarray(vf, dtype=np.float64).reshape(3)
        self.af = z.copy() if af is None else np.asarray(af, dtype=np.float64).reshape(3)

        self.t_start = 0.0
        self.t_end = float(np.sum(T))
        # cumulative segment-start times, length M+1; last = t_end
        # 每段起始的累计时间 (前缀和):_cum[i] 是第 i 段的起点时刻,最后一项 = 总时长
        self._cum = np.concatenate([[0.0], np.cumsum(T)])

        # 构造完就立刻解线性系统,把曲线系数 c 算出来
        self._solve()

    # ---- alternative constructor ----------------------------------------

    @classmethod
    def from_endpoints(
        cls,
        start: np.ndarray,
        goal: np.ndarray,
        q: np.ndarray,
        durations: np.ndarray,
        v0=None, a0=None, vf=None, af=None,
    ) -> "MinjerkTraj":
        """Build from explicit start, goal, interior waypoints q (M-1, 3).

        另一种构造方式:分开传起点、终点、中间途经点 q,内部拼成完整途经点再建。
        优化时手里通常正好是 q,用这个更顺手。
        """
        start = np.asarray(start, dtype=np.float64).reshape(1, 3)
        goal = np.asarray(goal, dtype=np.float64).reshape(1, 3)
        q = np.asarray(q, dtype=np.float64)
        if q.size == 0:
            wp = np.vstack([start, goal])
        else:
            wp = np.vstack([start, q.reshape(-1, 3), goal])
        return cls(wp, durations, v0=v0, a0=a0, vf=vf, af=af)

    # ---- banded M(T) construction + solve --------------------------------

    def _build_banded(self):
        """Build M(T) in scipy banded storage (l=u=6) plus dense version.

        Returns (ab, b) where ab is (13, 6M) banded form for solve_banded and
        b is the (6M, 3) RHS.  ab[u + i - j, j] = A[i, j].
        Also stash a sparse/dense A for testing introspection.

        组装那个线性方程组 M(T) c = b 的左边矩阵 M 和右边 b。
        M 是个"带状矩阵" (非零元素只挤在主对角线附近的一条带子里,带宽上下各 6),
        所以用 scipy 的带状存储格式存,解的时候是 O(M) 线性时间,非常快。
        每一行对应一个标量约束 (见文件开头列出的:起末边界、各段在拼接处的位置/速度/
        加速度/jerk/snap 连续、以及途经点必须被精确穿过)。
        b 是右边项:把途经点 q、起末位置/速度/加速度填进对应的行。
        额外存一份稀疏记录 (_coo) 只为测试时能重建出完整稠密矩阵来核对。
        """
        M, NC = self.M, self.NC
        n = NC * M
        l = u = 6
        ab = np.zeros((l + u + 1, n), dtype=np.float64)
        b = np.zeros((n, 3), dtype=np.float64)
        # keep a COO-style record for an exact dense reconstruction in tests
        rows, cols, vals = [], [], []

        # 往矩阵第 i 行第 j 列写值 v;同时按带状存储下标写进 ab,并记一份给测试用
        def setA(i, j, v):
            ab[u + i - j, j] = v
            rows.append(i); cols.append(j); vals.append(v)

        # 预先算好各段时长的各次幂 (T, T^2, ..., T^5),下面填多项式约束要反复用到
        T1 = self.T
        T2 = T1 * T1
        T3 = T2 * T1
        T4 = T2 * T2
        T5 = T4 * T1

        # head block: p(0)=p_start, v(0)=v0, a(0)=a0
        # 头部 3 行:第一段在 t=0 处的 位置/速度/加速度 必须等于给定的起点边界
        setA(0, 0, 1.0)
        setA(1, 1, 1.0)
        setA(2, 2, 2.0)
        b[0] = self.p_start
        b[1] = self.v0
        b[2] = self.a0

        # interior junctions
        # 中间每个接缝 (第 i 段末尾接第 i+1 段开头) 写 6 行约束:
        # jerk 连续、snap 连续、第 i 段末必须过途经点 q_i、再加 位置/速度/加速度 连续
        for i in range(M - 1):
            base = 6 * i
            setA(base + 3, base + 3, 6.0)
            setA(base + 3, base + 4, 24.0 * T1[i])
            setA(base + 3, base + 5, 60.0 * T2[i])
            setA(base + 3, base + 9, -6.0)

            setA(base + 4, base + 4, 24.0)
            setA(base + 4, base + 5, 120.0 * T1[i])
            setA(base + 4, base + 10, -24.0)

            setA(base + 5, base + 0, 1.0)
            setA(base + 5, base + 1, T1[i])
            setA(base + 5, base + 2, T2[i])
            setA(base + 5, base + 3, T3[i])
            setA(base + 5, base + 4, T4[i])
            setA(base + 5, base + 5, T5[i])

            setA(base + 6, base + 0, 1.0)
            setA(base + 6, base + 1, T1[i])
            setA(base + 6, base + 2, T2[i])
            setA(base + 6, base + 3, T3[i])
            setA(base + 6, base + 4, T4[i])
            setA(base + 6, base + 5, T5[i])
            setA(base + 6, base + 6, -1.0)

            setA(base + 7, base + 1, 1.0)
            setA(base + 7, base + 2, 2.0 * T1[i])
            setA(base + 7, base + 3, 3.0 * T2[i])
            setA(base + 7, base + 4, 4.0 * T3[i])
            setA(base + 7, base + 5, 5.0 * T4[i])
            setA(base + 7, base + 7, -1.0)

            setA(base + 8, base + 2, 2.0)
            setA(base + 8, base + 3, 6.0 * T1[i])
            setA(base + 8, base + 4, 12.0 * T2[i])
            setA(base + 8, base + 5, 20.0 * T3[i])
            setA(base + 8, base + 8, -2.0)

            b[base + 5] = self.q[i]  # 这一行的右边就是第 i 个途经点 (强制精确穿过)

        # tail block: p(T_M)=p_goal, v=vf, a=af  (segment M-1)
        # 尾部 3 行:最后一段在末端 (t=T_last) 处的 位置/速度/加速度 必须等于终点边界
        last = M - 1
        bN = 6 * M
        setA(bN - 3, bN - 6, 1.0)
        setA(bN - 3, bN - 5, T1[last])
        setA(bN - 3, bN - 4, T2[last])
        setA(bN - 3, bN - 3, T3[last])
        setA(bN - 3, bN - 2, T4[last])
        setA(bN - 3, bN - 1, T5[last])

        setA(bN - 2, bN - 5, 1.0)
        setA(bN - 2, bN - 4, 2.0 * T1[last])
        setA(bN - 2, bN - 3, 3.0 * T2[last])
        setA(bN - 2, bN - 2, 4.0 * T3[last])
        setA(bN - 2, bN - 1, 5.0 * T4[last])

        setA(bN - 1, bN - 4, 2.0)
        setA(bN - 1, bN - 3, 6.0 * T1[last])
        setA(bN - 1, bN - 2, 12.0 * T2[last])
        setA(bN - 1, bN - 1, 20.0 * T3[last])

        b[bN - 3] = self.p_goal
        b[bN - 2] = self.vf
        b[bN - 1] = self.af

        self._coo = (np.asarray(rows), np.asarray(cols), np.asarray(vals))
        return ab, b

    def _solve(self):
        ab, b = self._build_banded()
        # banded LU solve, O(M) for fixed bandwidth
        # 解带状线性系统 M(T) c = b,得到全部多项式系数 c;带宽固定故是 O(M) 线性时间
        c = solve_banded((6, 6), ab, b)
        self.c = np.ascontiguousarray(c)  # (6M, 3)  每段 6 个系数,共 M 段,3 维
        self._ab = ab
        self._b = b

    # ---- dense / sparse introspection for tests --------------------------

    def dense_M(self) -> np.ndarray:
        """Reconstruct the full dense M(T) from the recorded entries.

        把带状存的 M 还原成普通稠密矩阵 (仅供测试核对用,正常跑不需要)。
        """
        n = self.NC * self.M
        A = np.zeros((n, n), dtype=np.float64)
        r, c, v = self._coo
        A[r, c] = v
        return A

    @property
    def banded_M(self) -> np.ndarray:
        """The banded storage (13, 6M) passed to solve_banded."""
        return self._ab

    # ---- domain helpers --------------------------------------------------

    def _locate(self, t: float):
        """Return (segment index i, local tau in [0, T_i]) for global time t.

        给一个全局时刻 t,找出它落在第几段 (i),以及在那一段内部走了多久 (tau)。
        每段多项式用的是从 0 开始的"本段局部时间",所以要先减掉前面段的累计时间。
        """
        t = float(np.clip(t, self.t_start, self.t_end))
        if t >= self.t_end:
            i = self.M - 1
            return i, self.T[i]
        # find i with cum[i] <= t < cum[i+1]
        i = int(np.searchsorted(self._cum, t, side="right") - 1)
        if i < 0:
            i = 0
        if i > self.M - 1:
            i = self.M - 1
        tau = t - self._cum[i]
        return i, tau

    # ---- evaluation ------------------------------------------------------

    @staticmethod
    def _basis(tau: float, order: int) -> np.ndarray:
        """Powers basis row for the `order`-th derivative of [1,t,...,t^5].

        d^o/dt^o (t^j) = j!/(j-o)! * t^(j-o) for j>=o else 0.

        给出一行系数,用来和某段的多项式系数点乘,就得到该段在 tau 处的第 order 阶导。
        本质就是对每个幂次 t^j 求 order 阶导 (求一次导幂次降 1、系数乘上原幂次)。
        """
        row = np.zeros(6, dtype=np.float64)
        for j in range(order, 6):
            coeff = 1.0
            for k in range(order):
                coeff *= (j - k)
            row[j] = coeff * (tau ** (j - order))
        return row

    def eval_deriv(self, t, order: int = 0):
        """`order`-th derivative at t (scalar or 1-D array). Returns (d,) or (K,d).

        order 0 = position; matches UniformBSpline.eval_deriv semantics.

        求 t 时刻轨迹的第 order 阶导 (0=位置,1=速度,2=加速度...)。
        接口和 B 样条那个故意保持一致,方便互换。这里是向量化实现:一次能算一批 t。
        """
        if order < 0:
            raise ValueError("order must be >= 0")
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        # 求导次数超过 5,导数恒为 0
        if order > self.DEG:
            out = np.zeros((ts.size, self.d), dtype=np.float64)
        else:
            # vectorised _locate: clip, segment via searchsorted, local tau
            # 向量化版的 _locate:夹到有效区间 -> 二分定位每个 t 在第几段 -> 算本段局部时间
            tc = np.clip(ts, self.t_start, self.t_end)
            idx = np.searchsorted(self._cum, tc, side="right") - 1
            idx = np.clip(idx, 0, self.M - 1)
            tau = tc - self._cum[idx]                      # (K,)
            # vectorised _basis row per tau: col p = (p!/(p-o)!) tau^(p-o)
            # 给每个 tau 拼一行求导基函数 (同 _basis,但批量做)
            B = np.zeros((ts.size, 6), dtype=np.float64)
            for p in range(order, 6):
                coeff = 1.0
                for m in range(order):
                    coeff *= (p - m)
                B[:, p] = coeff * (tau ** (p - order))     # same `**` for bit-parity
            cseg = self.c.reshape(self.M, 6, self.d)[idx]  # (K,6,d) gathered coeffs  按段取出各点对应的系数
            out = np.einsum("kj,kjd->kd", B, cseg)         # row @ ci per point  基函数行 点乘 该段系数
        scalar = np.isscalar(t) or np.ndim(t) == 0
        return out[0] if scalar else out

    def eval(self, t):
        """Position at t. Alias of eval_deriv(t, 0) for B-spline parity.

        求 t 时刻的位置,就是 eval_deriv(t, 0) 的别名 (为和 B 样条接口对齐而设)。
        """
        return self.eval_deriv(t, 0)

    # =====================================================================
    # Bernstein control points (Stage 3 — continuous-time clearance basis)
    # =====================================================================
    # 这一段:把每段曲线换算成 Bernstein 控制点。为什么要这玩意?
    # 因为有"凸包性质"——整段连续曲线一定在它 6 个 Bernstein 控制点围成的凸包里。
    # 于是避障可以只盯着这 6 个点,而不必沿曲线密集采样,这就是"连续时间的安全证明"
    # (人的硬约束就建在它上面)。
    #
    # 推导:每段在本段局部时间 t∈[0,T_i] 上是 p_i(t)=Σ c[6i+j] t^j。先把时间归一化成
    # tau=t/T_i∈[0,1],系数随之缩放 a_{i,j}=c[6i+j]*T_i^j;控制点 P_i = C2B @ a_i。
    # 注意这个 T_i^j 缩放让 P_i 显式地依赖 T_i (除了通过 c=M(T)^-1 b 的隐式依赖之外)。
    # 这个"显式那一份"对 T 的导数在做时间优化时绝不能漏 (漏了就会掉进 M2 那个坑)。
    #
    # Segment i is p_i(t) = sum_j c[6i+j] t^j on the LOCAL clock t in [0,T_i].
    # Reparametrize t = tau*T_i, tau in [0,1]:  a_{i,j} = c[6i+j] * T_i^j, so
    # p_i(tau) = sum_j a_{i,j} tau^j.  Control points  P_i = C2B @ a_i (6,3).
    # The T_i^j scaling makes P_i depend on T_i EXPLICITLY (besides implicitly
    # via c = M(T)^-1 b); the explicit derivative is dP_i/dT_i = C2B @ dD/dT_i @ c_i
    # with dD/dT_i = diag([0,1,2T_i,3T_i^2,4T_i^3,5T_i^4]).

    @staticmethod
    def _D_powers(Ti: float) -> np.ndarray:
        """diag scaling vector [1,T,T^2,T^3,T^4,T^5] for one segment."""
        return np.array([1.0, Ti, Ti**2, Ti**3, Ti**4, Ti**5], dtype=np.float64)

    @staticmethod
    def _dD_powers(Ti: float) -> np.ndarray:
        """d/dT of the diag scaling: [0,1,2T,3T^2,4T^3,5T^4]."""
        return np.array([0.0, 1.0, 2.0 * Ti, 3.0 * Ti**2,
                         4.0 * Ti**3, 5.0 * Ti**4], dtype=np.float64)

    def control_points(self) -> np.ndarray:
        """Bernstein control points of every segment, shape (M, 6, 3).

        P[i,k,:] is the k-th degree-5 Bezier control point of segment i on the
        normalized domain tau in [0,1]. The continuous curve of segment i lies
        in conv{P[i,0..5]} (convex-hull property).  Vectorized: a = D(T)c per
        segment then P = C2B @ a as a single batched matmul.

        算出每一段的 6 个 Bernstein 控制点,形状 (M, 6, 3)。
        第 i 段那条连续曲线整段都在它这 6 个点的凸包里——避障就靠它做安全保证。
        """
        M = self.M
        cseg = self.c.reshape(M, 6, 3)                    # (M,6,3) per-segment coeffs
        T = self.T
        Dvec = np.stack([np.ones(M), T, T**2, T**3, T**4, T**5], axis=1)  # (M,6)
        a = Dvec[:, :, None] * cseg                       # (M,6,3) = D(T) c
        return np.einsum("kj,mjd->mkd", C2B, a)           # (M,6,3) = C2B @ a

    def control_point_times(self) -> np.ndarray:
        """Wall-clock time of every Bernstein control point, shape (M, 6).

        Bernstein control point k of segment i sits at normalized tau_k = k/5,
        i.e. local time (k/5)*T_i, so its absolute time relative to the replan
        instant is t_{i,k} = _cum[i] + (k/5)*T_i.  Single source of truth for
        the space-time ALM: t[:,0]==_cum[:M] (segment starts) and
        t[:,5]==_cum[1:M+1] (segment ends == the Stage-3 _seg_rep_time).

        给出每个 Bernstein 控制点对应的"绝对时刻",形状 (M, 6)。
        第 i 段第 k 个控制点在归一化 tau=k/5 处,所以它的时刻 = 本段起点时刻 + (k/5)*T_i。
        躲动态障碍 (人) 时要知道"这个点是几秒钟时的",才能去对上人在那一刻的预测位置——
        这就是空时 (space-time) 避障;这个时刻表就是它的唯一依据。
        """
        M = self.M
        kfrac = (np.arange(6) / float(self.DEG))[None, :]   # (1,6) = k/5
        return self._cum[:M, None] + kfrac * self.T[:, None]   # (M,6)

    def control_points_dT_explicit(self) -> np.ndarray:
        """Explicit dP_i/dT_i (holding c fixed), shape (M, 6, 3).

        dP_i/dT_i = C2B @ (dD/dT_i) @ c_i.  This is the explicit-T piece that
        the dCost/dT chain MUST include (dropping it reproduces the M2 trap).

        控制点对段时长 T_i 的"显式偏导" (把系数 c 当成常数不动时的那一份)。
        在优化时长 T 的链式求导里,这一份一定要算进去——漏掉它就会复现 M2 那个 bug
        (优化器卡住/方向不对)。完整的 dCost/dT 还要加上 c 随 T 变化的隐式那份。
        """
        M = self.M
        cseg = self.c.reshape(M, 6, 3)
        T = self.T
        dDvec = np.stack([np.zeros(M), np.ones(M), 2.0 * T, 3.0 * T**2,
                          4.0 * T**3, 5.0 * T**4], axis=1)  # (M,6)
        da = dDvec[:, :, None] * cseg                       # (M,6,3)
        return np.einsum("kj,mjd->mkd", C2B, da)

    # =====================================================================
    # M1 — analytic gradient through the MINCO map  M(T) c = b(q)
    # =====================================================================
    # 这一段:解析地求"代价对 q 和 T 的梯度"。难点在于 c 不是自由变量,而是被
    # M(T) c = b(q) 牵着走,所以梯度必须穿过这个线性映射反传。
    #
    # 用的是"伴随法 (adjoint)"——一句话讲:不用一个个变量去数值微分,而是先解一个
    # 转置方程 M^T λ = dJ/dc 拿到 λ,再用 λ 一次性把梯度摊回给 q 和 T。便宜又精确。
    #   - dJ/dq_i = λ[6i+5]              (途经点 q_i 正好坐在右边项的第 6i+5 行)
    #   - dJ/dT_k = (能量对 T 的显式偏导) - λ^T (dM/dT_k) c
    # 这套和 GCOPTER 里的 getGrad/propogateGrad 完全对应,测试里用中心差分逐项核对过。
    #
    # Control cost (canonical MINCO energy):  J = integral ||jerk||^2 dt.
    # For segment i, p_i(t)=sum_j c[6i+j] t^j (ASCENDING), jerk=p'''=
    #   6 c3 + 24 c4 t + 60 c5 t^2  (c3,c4,c5 == c[6i+3],c[6i+4],c[6i+5]).
    #   J_i = 720 T^5 c5.c5 + 720 T^4 c4.c5 + 240 T^3 c3.c5
    #         + 192 T^3 c4.c4 + 144 T^2 c3.c4 + 36 T c3.c3
    # (derived by symbolic integration; matches GCOPTER MINCO_S3NU::getEnergy
    #  with its internal ascending coefficient order). The . is dot over the
    #  3 spatial dims. dJ/dc and the explicit dJ/dT below are the symbolic
    #  partials of this expression; verified row-by-row against sympy and
    #  end-to-end against central finite differences in the M1 tests.
    #
    # Adjoint backprop (chain rule through M(T) c = b):
    #   solve  M^T lambda = dJ/dc
    #   dJ/dq_i = lambda[6i+5]           (q_i sits in RHS row 6i+5)
    #   dJ/dT_k = (explicit energy dJ/dT_k) - lambda^T (dM/dT_k) c
    # Mirrors GCOPTER getGrad / propogateGrad (solveAdj == solve M^T).

    def energy(self) -> float:
        """Total jerk-squared integral of the solved trajectory (scalar).

        整条轨迹的 jerk 平方积分 (一个标量),衡量"轨迹够不够顺滑、不抽搐"。越小越平滑。
        """
        return float(self._energy_and_gradc()[0])

    def _energy_and_gradc(self):
        """Return (J, dJ/dc) with dJ/dc shape (6M, 3). Closed form.

        返回 (能量 J, J 对系数 c 的梯度)。都是闭式公式 (直接套符号积分推出来的式子),
        没有数值微分。下面那一长串系数就是符号积分的结果。
        """
        M = self.M
        c = self.c
        T = self.T
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        T5 = T4 * T
        gdC = np.zeros_like(c)
        J = 0.0
        for i in range(M):
            b = 6 * i
            c3 = c[b + 3]
            c4 = c[b + 4]
            c5 = c[b + 5]
            J += (36.0 * T[i] * c3.dot(c3)
                  + 144.0 * T2[i] * c3.dot(c4)
                  + 192.0 * T3[i] * c4.dot(c4)
                  + 240.0 * T3[i] * c3.dot(c5)
                  + 720.0 * T4[i] * c4.dot(c5)
                  + 720.0 * T5[i] * c5.dot(c5))
            gdC[b + 3] = 72.0 * T[i] * c3 + 144.0 * T2[i] * c4 + 240.0 * T3[i] * c5
            gdC[b + 4] = 144.0 * T2[i] * c3 + 384.0 * T3[i] * c4 + 720.0 * T4[i] * c5
            gdC[b + 5] = 240.0 * T3[i] * c3 + 720.0 * T4[i] * c4 + 1440.0 * T5[i] * c5
        return J, gdC

    def _energy_grad_time_explicit(self) -> np.ndarray:
        """Explicit dJ/dT_i (holding c fixed), shape (M,). Symbolic partials.

        能量 J 对各段时长 T 的"显式偏导" (c 当常数)。同样是符号推导的闭式公式。
        完整 dJ/dT 还要再叠加 c 随 T 变的隐式那份 (在 grad_from_dcost_dc 里补)。
        """
        M = self.M
        c = self.c
        T = self.T
        T2 = T * T
        T3 = T2 * T
        T4 = T2 * T2
        gdT = np.zeros(M, dtype=np.float64)
        for i in range(M):
            b = 6 * i
            c3 = c[b + 3]
            c4 = c[b + 4]
            c5 = c[b + 5]
            gdT[i] = (36.0 * c3.dot(c3)
                      + 288.0 * T[i] * c3.dot(c4)
                      + 576.0 * T2[i] * c4.dot(c4)
                      + 720.0 * T2[i] * c3.dot(c5)
                      + 2880.0 * T3[i] * c4.dot(c5)
                      + 3600.0 * T4[i] * c5.dot(c5))
        return gdT

    # ---- adjoint solve M^T x = rhs ---------------------------------------

    def _solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Solve  M(T)^T x = rhs  (rhs (6M,3) -> x (6M,3)) via banded LU.

        Reuses the same banded factor structure; M^T also has bandwidth (6,6)
        so this stays O(M).

        解转置方程 M^T x = rhs,这是伴随法的核心一步 (拿到 λ)。
        M 的转置也是带状的,所以照样 O(M) 的线性时间。
        """
        n = self.NC * self.M
        l = u = 6
        ab = self._ab  # storage of A:  ab[u + i - j, j] = A[i, j]
        # banded storage of A^T:  abT[u + i - j, j] = A[j, i] = A_T[i, j]
        # A[j,i] = ab[u + j - i, i]  ->  abT[u + i - j, j] = ab[u + j - i, i]
        # 在带状存储里就地"转置":把 M 的带状表搬成 M^T 的带状表
        abT = np.zeros_like(ab)
        for i in range(n):
            jlo = max(0, i - l)
            jhi = min(n - 1, i + u)
            for j in range(jlo, jhi + 1):
                abT[u + i - j, j] = ab[u + j - i, i]
        return solve_banded((l, u), abT, rhs)

    # ---- dM/dT_k applied to c  -------------------------------------------

    def _dM_dT_times_c(self, k: int) -> np.ndarray:
        """Return the sparse vector  (dM/dT_k) c  of shape (6M, 3).

        Only the rows of M that contain T_k are nonzero. For an interior
        segment k < M-1 these are the junction rows 6k+3..6k+8; for k==M-1
        they are the three tail rows. Each entry is d(M_row)/dT_k . c.

        算 (dM/dT_k) c 这个向量——伴随法里求 dJ/dT_k 要用到的那一项。
        矩阵 M 里只有用到了 T_k 的那几行会随 T_k 变,其余全是 0,所以这个向量很稀疏:
        中间段 k 影响的是第 6k+3..6k+8 这 6 行接缝约束;最后一段影响尾部 3 行。
        每个非零项 = 那一行对 T_k 求导后再点乘当前系数 c。下面逐行的注释标了是哪条约束。
        """
        n = self.NC * self.M
        c = self.c
        T = self.T[k]
        out = np.zeros((n, 3), dtype=np.float64)
        if k < self.M - 1:
            b = 6 * k
            ci = c[b:b + 6]  # coeffs of segment k (the one ending at T_k)
            # d/dT of the segment-k basis rows (see header derivation):
            # jerk-cont row 6k+3: [6,24T,60T^2] -> d: [0,24,120T] on cols 3,4,5
            out[b + 3] = 24.0 * ci[4] + 120.0 * T * ci[5]
            # snap-cont row 6k+4: [24,120T] -> d: [0,120] on cols 4,5
            out[b + 4] = 120.0 * ci[5]
            # waypoint row 6k+5: [1,T,T^2,T^3,T^4,T^5] -> d:[0,1,2T,3T^2,4T^3,5T^4]
            out[b + 5] = (ci[1] + 2.0 * T * ci[2] + 3.0 * T**2 * ci[3]
                          + 4.0 * T**3 * ci[4] + 5.0 * T**4 * ci[5])
            # C0 pos row 6k+6: same pos basis (the -c_{k+1,0} term is T-free)
            out[b + 6] = out[b + 5]
            # C1 vel row 6k+7: [1,2T,3T^2,4T^3,5T^4] -> d:[0,2,6T,12T^2,20T^3]
            out[b + 7] = (2.0 * ci[2] + 6.0 * T * ci[3] + 12.0 * T**2 * ci[4]
                          + 20.0 * T**3 * ci[5])
            # C2 acc row 6k+8: [2,6T,12T^2,20T^3] -> d:[0,6,24T,60T^2]
            out[b + 8] = 6.0 * ci[3] + 24.0 * T * ci[4] + 60.0 * T**2 * ci[5]
        else:
            b = 6 * (self.M - 1)
            ci = c[b:b + 6]
            bN = n
            # tail pos row bN-3: pos basis -> [0,1,2T,3T^2,4T^3,5T^4]
            out[bN - 3] = (ci[1] + 2.0 * T * ci[2] + 3.0 * T**2 * ci[3]
                           + 4.0 * T**3 * ci[4] + 5.0 * T**4 * ci[5])
            # tail vel row bN-2: vel basis -> [0,2,6T,12T^2,20T^3]
            out[bN - 2] = (2.0 * ci[2] + 6.0 * T * ci[3] + 12.0 * T**2 * ci[4]
                           + 20.0 * T**3 * ci[5])
            # tail acc row bN-1: acc basis -> [0,6,24T,60T^2]
            out[bN - 1] = 6.0 * ci[3] + 24.0 * T * ci[4] + 60.0 * T**2 * ci[5]
        return out

    # ---- generic backprop hook -------------------------------------------

    def grad_from_dcost_dc(self, dcost_dc: np.ndarray,
                           dcost_dT_explicit: Optional[np.ndarray] = None):
        """Backprop an external dCost/dc (and optional explicit dCost/dT).

        【通用反传入口 / 最重要的函数】
        外面任何代价 (能量、避障代价...) 只要能给出"对系数 c 的偏导",这个函数就能
        把它穿过 MINCO 映射,反传成"对途经点 q 和段时长 T 的梯度"——优化器真正要的就是这个。
        典型用法:沿轨迹采样算避障代价、把敏感度累到 c 上,再丢进来。
        可选的 dcost_dT_explicit 是"代价直接通过 T (不经过 c) 依赖的那份显式偏导"。
        三步就是伴随法:解 M^T λ = dCost/dc;dCost/dq_i = λ[6i+5];
                       dCost/dT_k = 显式那份 - λ^T (dM/dT_k) c。

        Given the partial gradient of ANY scalar cost w.r.t. the trajectory
        coefficients c (shape (6M,3)) -- e.g. obtained by sampling an obstacle
        cost on the trajectory and accumulating its sensitivity onto c -- and
        optionally the explicit partial dCost/dT (shape (M,), the part of the
        cost that depends on T directly, not through c), return:

            (dCost/dq, dCost/dT)   shapes ((M-1,3), (M,))

        This is the chain rule through the MINCO map M(T) c = b(q):
            M^T lambda = dCost/dc
            dCost/dq_i = lambda[6i+5]
            dCost/dT_k = dCost/dT_k|explicit - lambda^T (dM/dT_k) c
        """
        dcost_dc = np.asarray(dcost_dc, dtype=np.float64)
        if dcost_dc.shape != (self.NC * self.M, 3):
            raise ValueError(
                f"dcost_dc must be ({self.NC * self.M}, 3), got {dcost_dc.shape}")
        M = self.M
        # 第一步:解伴随方程拿到 λ
        lam = self._solve_adjoint(dcost_dc)  # (6M, 3)

        # waypoint gradient: q_i lives in RHS row 6i+5 (junction loop i=0..M-2)
        # 对途经点的梯度:q_i 就坐在右边项第 6i+5 行,所以直接取 λ 的那一行即可
        gq = np.zeros((M - 1, 3), dtype=np.float64)
        for i in range(M - 1):
            gq[i] = lam[6 * i + 5]

        gT = np.zeros(M, dtype=np.float64)
        # 先加上"直接经 T 的显式梯度"那份 (如果外面给了的话)
        if dcost_dT_explicit is not None:
            gT += np.asarray(dcost_dT_explicit, dtype=np.float64).reshape(M)
        # 再减去"经 c 隐式依赖 T"的那份:- λ^T (dM/dT_k) c,逐段算
        for k in range(M):
            dMc = self._dM_dT_times_c(k)  # (6M, 3)
            gT[k] -= float(np.sum(lam * dMc))
        return gq, gT

    def energy_grad(self):
        """Analytic gradient of the jerk-energy cost J.

        Returns (J, dJ/dq, dJ/dT) with shapes (scalar, (M-1,3), (M,)).
        Combines the closed-form energy partials with the MINCO adjoint.

        平滑度代价 J (jerk 能量) 的解析梯度,返回 (J, 对途经点的梯度, 对段时长的梯度)。
        做法就是把上面的闭式能量偏导喂进通用反传入口 grad_from_dcost_dc。
        这是优化目标里"让轨迹尽量顺滑"那一项的梯度来源。
        """
        J, gdC = self._energy_and_gradc()
        gdT_explicit = self._energy_grad_time_explicit()
        gq, gT = self.grad_from_dcost_dc(gdC, dcost_dT_explicit=gdT_explicit)
        return J, gq, gT
