"""3D uniform B-spline geometry + LSQ fit from a polyline.

Convention:
  degree p   (default 5, quintic — C^{p-1} continuous)
  ctrl       (N+1, d)  control points P_0..P_N in R^d (d=3 here)
  uniform knot spacing dt:   t_i = (i - p) * dt   for i in 0..N+p+1
  valid parameter domain     t in [t_p, t_{N+1}], length T = (N - p + 1) * dt
  evaluation                 De Boor recursion (Wikipedia indexing)
  k-th derivative            itself a uniform B-spline of degree p-k with
                             control points Q_i^{(1)} = (P_{i+1} - P_i) / dt
                             (the uniform-knot identity; recurse for higher k)

中文说明:
这个文件是一条 3D 均匀 B 样条曲线的几何工具:能根据控制点算曲线上任意时刻的
位置/各阶导数,也能用最小二乘把一条折线拟合成 B 样条。
在这套规划器里它是"早期版本/对照用"的轨迹底座——后来主力轨迹换成了 MINCO
(见 minco.py),MINCO 故意把接口 (eval / eval_deriv) 做得和这里一样,方便替换。

几个约定 (术语随手解释):
  - degree p 阶数:默认 5 (五次曲线,quintic),曲线到 p-1 阶导都连续 (够平滑)。
  - ctrl 控制点:(N+1, d) 的数组,d=3。控制点像"磁铁",曲线被它们牵着走但一般不穿过。
  - 均匀 (uniform):节点 (knot) 等间距 dt。节点就是把参数轴切成段的那些分界值。
  - 有效时间区间 [t_p, t_{N+1}]:只有这段参数上曲线才有完整定义,故意让起点 t_p=0。
  - 求值用 De Boor 递推 (一种数值稳定的 B 样条求值标准算法)。
  - 求 k 阶导有个好用的性质:均匀 B 样条的导数还是 B 样条,只是阶数降 1、控制点
    换成相邻控制点的差分 (P_{i+1}-P_i)/dt;要更高阶导就反复做这一步。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class UniformBSpline:
    """一条均匀 3D B 样条曲线。

    给一组控制点 ctrl、阶数 degree、节点间距 dt,就能 eval 位置、eval_deriv 各阶导,
    或用 fit_path 从折线拟合出来。
    """

    def __init__(self, ctrl: np.ndarray, degree: int = 5, dt: float = 1.0):
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.ndim != 2:
            raise ValueError(f"ctrl must be (N+1, d), got shape {ctrl.shape}")
        if int(degree) < 1:
            raise ValueError(f"degree must be ≥ 1, got {degree}")
        if float(dt) <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.ctrl = ctrl
        self.p = int(degree)
        self.dt = float(dt)
        self.n = ctrl.shape[0] - 1  # last index of P  控制点最后一个的下标 (共 N+1 个)
        self.d = ctrl.shape[1]
        if self.n < self.p:
            raise ValueError(f"need ≥ p+1={self.p+1} control points, got {self.n+1}")
        self.m = self.n + self.p + 1  # last knot index  最后一个节点的下标
        # uniform knots so that t_p = 0 (clean parameter origin)
        # 等间距节点,故意平移让 t_p = 0,使有效区间从 0 开始,后面算时间干净
        self.knots = (np.arange(self.m + 1) - self.p) * self.dt
        self.t_start = self.knots[self.p]
        self.t_end = self.knots[self.n + 1]
        self.T = self.t_end - self.t_start

    # ---- domain helpers ---------------------------------------------------

    # 找时间 t 落在哪个节点区间 (span) 里:返回满足 knots[k] <= t < knots[k+1] 的 k
    def _find_span(self, t: float) -> int:
        t = float(np.clip(t, self.t_start, self.t_end))
        # t == t_end belongs to the last valid span [t_n, t_{n+1}]
        # 终点 t_end 特判归到最后一个有效区间,避免越界
        if t >= self.t_end - 1e-12:
            return self.n
        # binary search for k with knots[k] <= t < knots[k+1], k in [p, n]
        # 在 [p, n] 上二分查找
        lo, hi = self.p, self.n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.knots[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ---- evaluation -------------------------------------------------------

    def eval(self, t):
        """Position at t (scalar or 1-D array). Returns (d,) or (K, d).

        求 t 时刻曲线的位置。t 可以是单个数 (返回一个 3D 点) 或一维数组 (返回一批点)。
        """
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((ts.size, self.d), dtype=np.float64)
        for i, ti in enumerate(ts):
            out[i] = self._de_boor(ti, self.ctrl, self.p, self.knots)
        return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out

    def eval_deriv(self, t, order: int = 1):
        """k-th derivative at t. Returns same shape as eval.

        求 t 时刻曲线的第 order 阶导数 (order=0 即位置,1 是速度,2 是加速度……)。
        返回形状和 eval 一样。
        """
        if order < 0:
            raise ValueError("order must be ≥ 0")
        if order == 0:
            return self.eval(t)
        # 求导次数超过阶数,导数恒为 0,直接返回零
        if order > self.p:
            zero = np.zeros(self.d)
            ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
            return zero if (np.isscalar(t) or np.ndim(t) == 0) else np.tile(zero, (ts.size, 1))
        # build derivative control points + degree + knots via the uniform identity
        # 用均匀 B 样条求导性质:每求一阶导,控制点换成相邻差分/dt,阶数减 1,
        # 节点掐掉首尾各一个;反复 order 次后,导数就是一条低阶 B 样条
        Q = self.ctrl
        knots = self.knots
        p = self.p
        for _ in range(order):
            Q = (Q[1:] - Q[:-1]) / self.dt
            knots = knots[1:-1]
            p -= 1
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        out = np.empty((ts.size, self.d), dtype=np.float64)
        for i, ti in enumerate(ts):
            out[i] = self._de_boor_with(ti, Q, p, knots, self.t_start, self.t_end)
        return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out

    # ---- nonzero basis (for LSQ assembly) --------------------------------

    def nonzero_basis(self, t: float) -> Tuple[int, np.ndarray]:
        """Return (span k, basis values [N_{k-p,p}(t) .. N_{k,p}(t)]).

        在 t 处,只有 p+1 个 B 样条基函数非零。返回 (t 所在区间 k, 这 p+1 个基函数的值)。
        拟合 (fit_path) 时要用它来一行一行地拼最小二乘矩阵。
        这里用的是数值稳定的 Cox-de Boor 三角递推算法。
        """
        t = float(np.clip(t, self.t_start, self.t_end))
        k = self._find_span(t)
        N = np.zeros(self.p + 1, dtype=np.float64)
        N[0] = 1.0
        left = np.zeros(self.p + 1)
        right = np.zeros(self.p + 1)
        for j in range(1, self.p + 1):
            left[j] = t - self.knots[k + 1 - j]
            right[j] = self.knots[k + j] - t
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = N[r] / denom if denom != 0 else 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        return k, N

    # ---- endpoint clamping (geometric) -----------------------------------

    def lock_endpoint(self, side: str, position: np.ndarray) -> None:
        """Pin the endpoint to `position` with zero velocity / accel / … up to
        (p-1)-th derivative — by setting the first or last p+1 control points
        all equal to `position` (geometric identity for uniform B-splines).

        把曲线端点钉死在 position,且端点处速度/加速度/...一直到 p-1 阶导全为 0
        (即"停稳"地起步或收尾)。做法:把开头 (或结尾) 的 p+1 个控制点全设成同一个点。
        这是均匀 B 样条的一个几何恒等式——控制点重合,端点导数自然归零。
        """
        position = np.asarray(position, dtype=np.float64).reshape(self.d)
        if side == "start":
            self.ctrl[: self.p + 1] = position
        elif side == "end":
            self.ctrl[-(self.p + 1) :] = position
        else:
            raise ValueError("side must be 'start' or 'end'")

    # ---- LSQ fit from a polyline -----------------------------------------

    @classmethod
    def fit_path(
        cls,
        path: np.ndarray,
        num_ctrl: int,
        degree: int = 5,
        dt: float = 1.0,
        *,
        clamp_endpoints: bool = False,
    ) -> "UniformBSpline":
        """Least-squares fit a uniform B-spline to a polyline.

        path: (M, d) waypoints; treated as arc-length samples mapped uniformly
              onto the valid domain [t_p, t_{N+1}].
        num_ctrl: number of control points (must be ≥ degree+1).
        clamp_endpoints: after LSQ, pin both endpoints to path[0] / path[-1]
              with zero derivatives (via lock_endpoint).

        用最小二乘把一条折线 (比如全局 A* 给的路径点) 拟合成一条平滑 B 样条。
        path: (M, d) 一串路径点;直接按顺序均匀铺到曲线的有效时间区间上。
        num_ctrl: 想用多少个控制点 (至少 degree+1 个)。点越多拟合越贴、越少越平滑。
        clamp_endpoints: 拟合完后,是否把首尾两端钉死在 path 的起点/终点并让导数归零。
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2:
            raise ValueError(f"path must be (M, d), got shape {path.shape}")
        M, d = path.shape
        N = num_ctrl - 1
        if N < degree:
            raise ValueError(f"num_ctrl ≥ degree+1={degree+1}, got {num_ctrl}")
        if clamp_endpoints and num_ctrl < 2 * (degree + 1):
            # locking both ends needs disjoint first/last p+1 ctrl points
            # 要钉死两端,首 p+1 个和尾 p+1 个控制点不能重叠,所以总数至少 2*(p+1)
            raise ValueError(
                f"clamp_endpoints needs num_ctrl ≥ 2*(degree+1)={2 * (degree + 1)}, got {num_ctrl}"
            )
        # build a temporary spline with zeros to use its basis machinery
        # 先建一条全零控制点的临时样条,只为借用它的节点/基函数计算
        bs = cls(np.zeros((num_ctrl, d)), degree=degree, dt=dt)
        # 在有效时间区间上均匀取 M 个时刻,和 path 的 M 个点一一对应
        ts = np.linspace(bs.t_start, bs.t_end, M)
        # assemble (M x (N+1)) basis matrix; each row has only p+1 nonzeros
        # 拼最小二乘矩阵 B:第 i 行 = 第 i 个时刻处各控制点的基函数值 (只有 p+1 个非零)
        B = np.zeros((M, N + 1), dtype=np.float64)
        for i, ti in enumerate(ts):
            k, vals = bs.nonzero_basis(ti)
            B[i, k - degree : k + 1] = vals
        # 解最小二乘 B @ ctrl ≈ path,得到最贴合这串点的控制点
        ctrl, *_ = np.linalg.lstsq(B, path, rcond=None)
        bs.ctrl = ctrl
        if clamp_endpoints:
            bs.lock_endpoint("start", path[0])
            bs.lock_endpoint("end", path[-1])
        return bs

    # ---- internal: De Boor on (ctrl, p, knots) ---------------------------

    def _de_boor(self, t: float, ctrl: np.ndarray, p: int, knots: np.ndarray) -> np.ndarray:
        return self._de_boor_with(t, ctrl, p, knots, self.t_start, self.t_end)

    # De Boor 递推 (B 样条求值的标准算法):对给定的控制点/阶数/节点,算出 t 处的点。
    # 把 t 所在区间相关的 p+1 个控制点,一轮轮做线性插值,最后收敛到曲线上那一个点。
    @staticmethod
    def _de_boor_with(t, ctrl, p, knots, t_start, t_end):
        t = float(np.clip(t, t_start, t_end))
        n_local = ctrl.shape[0] - 1
        if t >= t_end - 1e-12:
            k = n_local
        else:
            # binary search in [p, n_local]  二分定位 t 所在的节点区间
            lo, hi = p, n_local
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if knots[mid] <= t:
                    lo = mid
                else:
                    hi = mid - 1
            k = lo
        # 取出这个区间相关的 p+1 个控制点作为递推起点
        d = np.array([ctrl[k - p + j] for j in range(p + 1)], dtype=np.float64)
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i_lo = k - p + j  # knot index t_{j+k-p}
                # alpha = (t - knots[i_lo]) / (knots[i_lo + p + 1 - r] - knots[i_lo])
                # alpha 是这一轮的插值比例;分母为 0 (节点重合) 时取 0 防除零
                denom = knots[i_lo + p + 1 - r] - knots[i_lo]
                alpha = 0.0 if denom == 0 else (t - knots[i_lo]) / denom
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        return d[p]
