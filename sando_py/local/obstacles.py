"""Obstacle representations for local_opt.

All obstacles expose the same interface so cost.py can iterate uniformly:

  signed_dist(p, t) -> float
      Negative inside the obstacle, 0 on the boundary, positive outside.
      `t` is parameter along the trajectory — static obstacles ignore it,
      dynamic ones use it to evaluate their predicted pose at time t.

  class_name : str
      Links the obstacle to AvoidParams in the avoid_config dict.

Per the chosen first-version representations:
  walls  → AABBObstacle (axis-aligned box)
  humans → SphereObstacle (centre + radius, optional linear velocity)

中文说明:
这个文件定义"障碍长什么样"。局部优化器要算"轨迹离障碍多远",就靠这里的障碍对象。
所有障碍都长一个统一的接口,这样 cost.py 不用管具体是哪种,挨个调用就行:

  signed_dist(p, t) -> float  带符号距离
      点 p 在障碍内部时返回负数,正好在边界上是 0,在外面是正数。
      t 是"轨迹上的时间参数":静态障碍 (墙) 不理它;动态障碍 (人) 用它算出
      "在 t 时刻这个障碍预测会移动到哪儿",再算距离——这就是空时 (space-time) 避障。

  class_name 类别名字:把这个障碍和 avoid_config 里的参数对应起来 (查软/硬、安全距离)。

第一版选的两种几何表示:
  墙  → AABBObstacle  轴对齐包围盒 (一个不旋转的长方体)
  人  → SphereObstacle 球 (球心 + 半径,可选一个匀速速度)
"""
from __future__ import annotations

import numpy as np


class Obstacle:
    """Base interface. Subclasses must override signed_dist + class_name.

    障碍基类 (只是个接口约定)。子类必须实现 signed_dist 和填上 class_name。
    """
    class_name: str = ""

    def signed_dist(self, p: np.ndarray, t: float) -> float:
        raise NotImplementedError


class SphereObstacle(Obstacle):
    """Sphere with optional constant linear velocity (linear extrapolation).

    centre(t) = centre0 + vel * t
    signed_dist = ||p - centre(t)|| - radius

    球障碍 (用来表示"人"),可带一个匀速速度。
    预测很简单——假设它一直匀速直线走:t 时刻的球心 = 初始球心 + 速度 * t。
    带符号距离 = 点到 (移动后) 球心的距离 - 半径。
    """

    def __init__(self, centre0, radius: float, vel=None, class_name: str = "human"):
        self.centre0 = np.asarray(centre0, dtype=np.float64).reshape(3)
        self.radius = float(radius)
        self.vel = (np.zeros(3, dtype=np.float64) if vel is None
                    else np.asarray(vel, dtype=np.float64).reshape(3))
        self.class_name = class_name

    # 匀速外推:t 时刻球心 = 初始球心 + 速度*t
    def predict(self, t: float) -> np.ndarray:
        return self.centre0 + self.vel * float(t)

    def signed_dist(self, p: np.ndarray, t: float) -> float:
        c = self.predict(t)
        return float(np.linalg.norm(np.asarray(p) - c) - self.radius)


class AABBObstacle(Obstacle):
    """Axis-aligned bounding box; static (no .predict(t)).

    signed_dist for points outside is the euclidean distance to the closest
    face; for points inside it is the negative penetration depth (max of
    per-axis penetrations, i.e. L∞), which is sufficient for the (d_safe - d)
    penalty since any inside-point yields a large positive penalty already.

    轴对齐包围盒 (用来表示"墙")。轴对齐 = 盒子的边都顺着 xyz 坐标轴,不旋转。
    它是静态的,所以没有 predict(t),signed_dist 里的 t 直接忽略。
    带符号距离的算法:
      - 点在盒子外:返回点到最近那个面的真实欧氏距离 (正数)。
      - 点在盒子内:返回"扎进去多深"的负值。这里用的是各轴穿透量里最大的那个
        (L∞ 范数,不是真正的最短距离),但对避障代价来说够用了——只要点在墙里,
        (d_safe - d) 算出来本来就是个很大的惩罚,不差这点精度。
    """

    def __init__(self, lo, hi, class_name: str = "wall"):
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if np.any(hi < lo):
            raise ValueError(f"AABB hi must be ≥ lo elementwise; lo={lo}, hi={hi}")
        self.lo = lo
        self.hi = hi
        self.class_name = class_name

    def signed_dist(self, p: np.ndarray, t: float = 0.0) -> float:
        p = np.asarray(p, dtype=np.float64)
        # outside[axis]:点在这个轴上超出盒子边界多少 (没超出就是 0)
        outside = np.maximum(self.lo - p, 0.0) + np.maximum(p - self.hi, 0.0)
        # 任一轴超出 -> 点在盒外,各轴超出量当成向量取模就是到盒子的欧氏距离
        if np.any(outside > 0):
            return float(np.linalg.norm(outside))
        # 否则点在盒内:取各轴穿透量的最大值 (一定 ≤ 0),作为负的"穿透深度"
        inside = float(np.max(np.maximum(self.lo - p, p - self.hi)))  # ≤ 0
        return inside
