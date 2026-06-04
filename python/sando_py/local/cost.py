"""Per-class obstacle-avoidance cost along a B-spline trajectory.

  EGO-Planner cubic penalty:   φ(d) = (d_safe - d)^3 if d < d_safe else 0
                               (C^2 continuous at the safety boundary)

  obstacle_cost(spline, obstacles, config, K=None) → scalar
      (1/K) · Σ_obs Σ_i  weight(obs.class) · φ(signed_dist(spline.eval(t_i), t_i),
                                               d_safe(obs.class))
      Averaged over K samples (uniform in t) so the cost is K-invariant —
      otherwise K and weight are entangled and grid refinement silently
      re-scales the optimisation.

Hard vs soft is just a weight knob — same shape — so the cost stays smooth
and differentiable (numerical gradients work fine for L-BFGS).

中文说明:
这个文件算"轨迹离障碍太近"要付的代价 (越靠近障碍代价越高)。优化器靠它把轨迹推开。

惩罚公式用 EGO-Planner 那套三次方:  φ(d) = (d_safe - d)^3 (当 d < d_safe),否则 0。
意思是:点离障碍的距离 d 比安全距离 d_safe 远时不罚 (0);一旦比安全距离近,
就按"还差多少"的三次方来罚,越近罚得越狠。用三次方是为了在安全边界处接得平滑
(C^2 连续,二阶导都连续),这样优化器求梯度时不会在边界跳变。

总代价 = 把轨迹按时间均匀采 K 个点,对每个障碍、每个采样点算惩罚,乘上该类别的
权重,全加起来,最后除以 K 取平均。除以 K 很关键:这样代价不随采样点数量变化
(K-invariant)。否则 K 和 weight 会纠缠在一起——你只是把采样加密一点,代价的尺度
就悄悄变了,等于偷偷改了优化目标。

软/硬之分只体现在 weight 大小上 (公式形状完全一样),所以整个代价始终光滑可导,
L-BFGS 用数值梯度也能稳稳地优化。
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .avoid_config import AvoidParams


# EGO 三次方惩罚:离障碍比安全距离近才罚,(d_safe - d)^3;够远就是 0。
def penalty(d: float, d_safe: float) -> float:
    diff = d_safe - d
    return diff * diff * diff if diff > 0.0 else 0.0


def obstacle_cost(spline, obstacles: Iterable, config: Dict[str, AvoidParams],
                  K: int = None) -> float:
    """Sum of per-class EGO cubic penalties over K samples of the spline.

    把轨迹按时间均匀采 K 个点,对每个障碍逐点算 EGO 三次方惩罚 * 该类别权重,
    全加起来再除以 K 取平均,返回一个标量代价。
    参数:spline 当前轨迹;obstacles 障碍列表;config 按类别查参数的字典;
         K 采样点数 (默认按控制点数自动定一个值)。
    """
    obstacles = list(obstacles)
    if not obstacles:
        return 0.0
    # 默认采样数随控制点多少而定,且至少 50 个,保证采样够密
    if K is None:
        K = max(50, 10 * (spline.n + 1))
    # K<=0 无采样点 -> 没有可测代价,直接 0.0(否则下面 total/K 会 0/0 ZeroDivisionError)
    if K <= 0:
        return 0.0
    # 在轨迹时间区间上均匀取 K 个时刻,一次性算出这些时刻的轨迹位置
    ts = np.linspace(spline.t_start, spline.t_end, K)
    pts = spline.eval(ts)
    total = 0.0
    for obs in obstacles:
        # 按障碍类别名查出它的安全距离和权重
        params = config[obs.class_name]
        d_safe = params.d_safe
        w = params.weight
        for i in range(K):
            # 注意把时刻 ts[i] 也传进去:动态障碍 (人) 要用它算预测位置 (空时避障)
            d = obs.signed_dist(pts[i], float(ts[i]))
            diff = d_safe - d
            # 只有比安全距离近 (diff>0) 才累加惩罚;这里就地展开了 penalty 的三次方
            if diff > 0.0:
                total += w * diff * diff * diff
    # 除以 K 取平均,让代价不随采样密度变化 (见模块开头说明)
    return float(total / K)
