"""Common utilities — geometry, time, msg <-> dataclass conversions.

中文说明:
这是规划器的「公共工具箱」, 里面是一堆零散的小函数, 自己不构成任何算法主线,
而是被其他模块到处调用。大致分四类:
  1. 几何 / 数学: 角度归一化、四元数<->偏航、向量限幅、把点投影到球/盒子表面、
     一维/三维双积分器的最短到达时间、路径细分等。
  2. ROS 时间: 读节点时钟、ROS 时间戳和秒数互转。
  3. Timer: 用墙上时钟(和 ROS 时钟无关)做性能计时, 给 profiling 用。
  4. 消息<->数据结构转换: RobotState / PieceWisePol 和 dynus_interfaces 的 ROS 消息互转。
  5. 路径简化: 去共线点、合并过短的边、按转角稀疏化顶点(让全局路径更干净)。
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

from .types import PieceWisePol, RobotState


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def angle_wrap(a: float) -> float:
    """Wrap to (-pi, pi].

    中文: 把任意角度折算到 (-pi, pi] 区间内(消掉多转的整圈)。
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(target: float, current: float) -> float:
    """Smallest signed angular distance from current to target.

    中文: 从 current 转到 target 的「最短带符号角差」(走近的那一边, 正负表方向)。
    """
    return angle_wrap(target - current)


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX Tait-Bryan yaw.

    中文: 从四元数里只取出偏航角 yaw(绕 z 轴的转角), 忽略俯仰/横滚。
    """
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


# 反过来: 由偏航角 yaw 构造四元数(只绕 z 轴转), 返回 (x, y, z, w)
def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


# 把标量 x 限制在 [lo, hi] 区间里
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# 向量限幅: 若 v 的长度超过 vmax, 就保持方向把长度压到 vmax; 否则原样返回
def saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > vmax and n > 1e-9:
        return v * (vmax / n)
    return v


def project_point_to_sphere(P1: np.ndarray, P2: np.ndarray, radius: float) -> np.ndarray:
    """Project P2 onto a sphere of `radius` centered at P1.  Returns P2 unchanged
    if already inside the sphere. Mirrors sando_utils::projectPointToSphere.

    中文: 把点 P2 投到「以 P1 为球心、半径 radius」的球面上。
    如果 P2 已经在球内就原样返回; 否则沿 P1->P2 方向缩到球面上。
    常用于把目标点拉到当前局部范围内(太远的目标先朝它走一段)。
    """
    diff = P2 - P1
    n = float(np.linalg.norm(diff))
    if n <= radius:
        return np.array(P2, dtype=float)
    return P1 + diff * (radius / n)


def project_point_to_box(P1: np.ndarray, P2: np.ndarray,
                         wdx: float, wdy: float, wdz: float) -> np.ndarray:
    """Project P2 onto the surface of an AABB of half-widths (wdx/2, wdy/2, wdz/2)
    centered at P1. Returns P2 unchanged if already inside the box.

    中文: 和上面的球版类似, 只是把 P2 投到「以 P1 为中心、各边宽 wdx/wdy/wdz」的
    轴对齐立方盒(AABB)表面上。盒内就原样返回。
    做法: 沿 P1->P2 这条射线, 求它和盒子六个面的交点, 取最近(t 最小)的那个交点。
    """
    x_max = P1[0] + wdx / 2.0
    x_min = P1[0] - wdx / 2.0
    y_max = P1[1] + wdy / 2.0
    y_min = P1[1] - wdy / 2.0
    z_max = P1[2] + wdz / 2.0
    z_min = P1[2] - wdz / 2.0
    if (x_min < P2[0] < x_max and y_min < P2[1] < y_max and z_min < P2[2] < z_max):
        return np.array(P2, dtype=float)
    # Direction from P1 to P2; parametric search for the closest box face intersection
    v = P2 - P1
    if np.linalg.norm(v) < 1e-12:
        return np.array(P1, dtype=float)
    best_t = float("inf")
    best_pt = np.array(P2, dtype=float)
    # 六个面, 每个用 (法向量 n, 偏移 d) 表示平面 n·x + d = 0
    planes = [
        (np.array([1.0, 0.0, 0.0]), -x_max),
        (np.array([-1.0, 0.0, 0.0]), x_min),
        (np.array([0.0, 1.0, 0.0]), -y_max),
        (np.array([0.0, -1.0, 0.0]), y_min),
        (np.array([0.0, 0.0, 1.0]), -z_max),
        (np.array([0.0, 0.0, -1.0]), z_min),
    ]
    for n_pl, d_pl in planes:
        denom = float(np.dot(n_pl, v))
        # 射线和该面平行, 没有交点, 跳过
        if abs(denom) < 1e-12:
            continue
        # 解射线参数 t, 使 P1 + t·v 正好落在这个面上; 只要正方向(t>0)里最近的那个
        t = -(float(np.dot(n_pl, P1)) + d_pl) / denom
        if 0.0 < t < best_t:
            best_t = t
            best_pt = P1 + v * t
    return best_pt


def min_time_double_integrator_1d(p0: float, v0: float, pf: float, vf: float,
                                  v_max: float, a_max: float) -> float:
    """Time-optimal point-to-point transfer time for a 1D double integrator
    with bounded velocity and acceleration. Mirrors getMinTimeDoubleIntegrator1D.

    中文: 一维「双积分器」(就是把质点当成只能控加速度的模型)从 (位置 p0, 速度 v0)
    到 (pf, vf) 的「最短可行时间」, 约束是速度不超 v_max、加速度不超 a_max。
    这是用来估计一段路大概要飞多久(给轨迹分配时间)的解析公式, 不是真去优化轨迹。
    下面几个 if 分支对应「全程加速段/有匀速饱和段」等几种不同的运动剖面情况,
    公式直接照搬 C++ 基线, 不用逐项推。
    """
    x1, x2 = v0, p0
    x1r, x2r = vf, pf
    k1 = a_max
    k2 = 1.0
    x1_bar = v_max
    sign = 1.0 if (-x1 + x1r) > 0 else (-1.0 if (-x1 + x1r) < 0 else 0.0)
    B = (k2 / (2 * k1)) * sign * (x1 ** 2 - x1r ** 2) + x2r
    C = (k2 / (2 * k1)) * (x1 ** 2 + x1r ** 2) - (k2 / k1) * x1_bar ** 2 + x2r
    D = (-k2 / (2 * k1)) * (x1 ** 2 + x1r ** 2) + (k2 / k1) * x1_bar ** 2 + x2r
    if x2 <= B and x2 >= C:
        inside = (k2 ** 2) * (x1 ** 2) - k1 * k2 * ((k2 / (2 * k1)) * (x1 ** 2 - x1r ** 2) + x2 - x2r)
        inside = max(0.0, inside)
        t = (-k2 * (x1 + x1r) + 2 * math.sqrt(inside)) / (k1 * k2)
    elif x2 <= B and x2 < C:
        t = (x1_bar - x1 - x1r) / k1 + (x1 ** 2 + x1r ** 2) / (2 * k1 * x1_bar) + (x2r - x2) / (k2 * x1_bar)
    elif x2 > B and x2 <= D:
        inside = (k2 ** 2) * (x1 ** 2) + k1 * k2 * ((k2 / (2 * k1)) * (-x1 ** 2 + x1r ** 2) + x2 - x2r)
        inside = max(0.0, inside)
        t = (k2 * (x1 + x1r) + 2 * math.sqrt(inside)) / (k1 * k2)
    else:
        t = (x1_bar + x1 + x1r) / k1 + (x1 ** 2 + x1r ** 2) / (2 * k1 * x1_bar) + (-x2r + x2) / (k2 * x1_bar)
    return float(t)


def min_time_double_integrator_3d(p0: np.ndarray, v0: np.ndarray,
                                  pf: np.ndarray, vf: np.ndarray,
                                  v_max: np.ndarray, a_max: np.ndarray) -> float:
    """Maximum over axes of the per-axis 1D minimum-time transfer.

    中文: 三维版。x/y/z 三个轴各自算一维最短时间, 取最大值(三轴里最慢的那个决定总时长)。
    """
    tx = min_time_double_integrator_1d(p0[0], v0[0], pf[0], vf[0], v_max[0], a_max[0])
    ty = min_time_double_integrator_1d(p0[1], v0[1], pf[1], vf[1], v_max[1], a_max[1])
    tz = min_time_double_integrator_1d(p0[2], v0[2], pf[2], vf[2], v_max[2], a_max[2])
    return max(tx, ty, tz)


def create_more_vertexes(path: List[np.ndarray], d: float) -> List[np.ndarray]:
    """Subdivide each segment of `path` so that no two consecutive vertices are
    farther apart than `d`. Mirrors createMoreVertexes.

    中文: 给路径「加密顶点」。如果相邻两点离得比 d 远, 就在中间均匀插点,
    保证任意相邻两点间距不超过 d。后续按段处理(分配时间、建走廊)时点太稀会出问题, 先补密。
    """
    if len(path) < 2 or d <= 0.0:
        return [p.copy() for p in path]
    out: List[np.ndarray] = [path[0].copy()]
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L <= 1e-9:
            continue
        n = int(np.floor(L / d))
        u = seg / L
        for k in range(1, n + 1):
            out.append(a + u * (k * d))
        if np.linalg.norm(out[-1] - b) > 1e-6:
            out.append(b.copy())
    return out


def transform_stamped_to_matrix(transform_stamped) -> np.ndarray:
    """Convert a geometry_msgs/TransformStamped to a 4x4 numpy matrix.

    中文: 把 ROS 的 TransformStamped 坐标变换消息(平移 + 四元数旋转)
    拼成一个 4x4 齐次变换矩阵 [R t; 0 1], 方便直接做矩阵乘法变换点。
    """
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[0, 3] = float(t.x); M[1, 3] = float(t.y); M[2, 3] = float(t.z)
    return M


def transform_inverse_se3(M: np.ndarray) -> np.ndarray:
    """Analytic SE(3) inverse of a 4x4 transform: [R | t; 0 0 0 1]^{-1} = [R^T | -R^T t; 0 0 0 1].

    中文: 求 4x4 刚体变换矩阵的逆。不用通用矩阵求逆, 而是利用「旋转矩阵的逆=转置」
    这个性质直接写出来, 又快又稳: 逆的旋转是 R^T, 逆的平移是 -R^T·t。
    """
    R = M[:3, :3]
    t = M[:3, 3]
    RT = R.T
    M_inv = np.eye(4)
    M_inv[:3, :3] = RT
    M_inv[:3, 3] = -RT @ t
    return M_inv


# ---------------------------------------------------------------------------
# ROS time helpers
# ---------------------------------------------------------------------------

def ros_time_seconds(node) -> float:
    """Return the node's current ROS clock in seconds (float).

    中文: 读取节点当前的 ROS 时钟, 换成「秒」(浮点)。注意用的是 ROS 时间, 仿真里可能比真实快/慢。
    """
    t = node.get_clock().now()
    return float(t.nanoseconds) * 1e-9


# ROS 时间戳(sec 整秒 + nanosec 纳秒)转成秒
def builtin_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# 反过来: 秒数转成 ROS 的 builtin_interfaces/Time 时间戳消息
def seconds_to_builtin(seconds: float):
    from builtin_interfaces.msg import Time
    msg = Time()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - int(seconds)) * 1e9)
    return msg


# ---------------------------------------------------------------------------
# Wall-clock timing (for profiling — independent of ROS clock)
# ---------------------------------------------------------------------------

# 简单的墙上时钟计时器, 给性能 profiling 用(和 ROS 时钟无关, 量的是真实流逝时间)。
# 用法: t = Timer(); ...干活...; t.elapsed_ms()。
class Timer:
    def __init__(self):
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


# ---------------------------------------------------------------------------
# Message <-> dataclass conversion (dynus_interfaces)
# ---------------------------------------------------------------------------

def state_to_msg(s: RobotState):
    """Convert RobotState -> dynus_interfaces.msg.State (best effort).

    中文: 把内部的 RobotState 转成 ROS 的 State 消息(发布给别的节点)。
    best effort = 如果 dynus_interfaces 没装上(import 失败)就返回 None, 不报错。
    """
    try:
        from dynus_interfaces.msg import State as StateMsg
    except ImportError:
        return None
    from geometry_msgs.msg import Vector3, Quaternion

    m = StateMsg()
    m.pos = Vector3(x=float(s.pos[0]), y=float(s.pos[1]), z=float(s.pos[2]))
    m.vel = Vector3(x=float(s.vel[0]), y=float(s.vel[1]), z=float(s.vel[2]))
    qx, qy, qz, qw = quat_from_yaw(s.yaw)
    m.quat = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return m


# 反方向: 把收到的 State 消息解析回内部 RobotState; 用 hasattr 逐字段判断,
# 哪个字段消息里没有就跳过, 兼容不同来源的消息格式。
def msg_to_state(msg) -> RobotState:
    s = RobotState()
    if hasattr(msg, "pos"):
        s.pos = np.array([msg.pos.x, msg.pos.y, msg.pos.z])
    if hasattr(msg, "vel"):
        s.vel = np.array([msg.vel.x, msg.vel.y, msg.vel.z])
    if hasattr(msg, "quat"):
        s.yaw = yaw_from_quat(msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w)
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        s.t = builtin_to_seconds(msg.header.stamp)
    return s


def state_to_goal_msg(s: RobotState):
    """Convert RobotState -> dynus_interfaces.msg.Goal.

    中文: 把 RobotState 转成 Goal 消息(发给底层控制器的目标指令)。
    Goal 比 State 多带加速度 a、jerk j 和 yaw 角速度 dyaw, 还有 power/mode 等控制标志。
    """
    try:
        from dynus_interfaces.msg import Goal as GoalMsg
    except ImportError:
        return None
    from geometry_msgs.msg import Vector3

    g = GoalMsg()
    g.p = Vector3(x=float(s.pos[0]), y=float(s.pos[1]), z=float(s.pos[2]))
    g.v = Vector3(x=float(s.vel[0]), y=float(s.vel[1]), z=float(s.vel[2]))
    g.a = Vector3(x=float(s.accel[0]), y=float(s.accel[1]), z=float(s.accel[2]))
    g.j = Vector3(x=float(s.jerk[0]), y=float(s.jerk[1]), z=float(s.jerk[2]))
    g.yaw = float(s.yaw)
    g.dyaw = float(s.dyaw)
    g.power = True
    g.mode_xy = 0
    g.mode_z = 0
    return g


# 把分段多项式轨迹 PieceWisePol 转成 PWPTraj 消息: 每段每轴打包成 CoeffPoly3(a,b,c,d)
def pwp_to_msg(pwp: PieceWisePol):
    try:
        from dynus_interfaces.msg import PWPTraj, CoeffPoly3
    except ImportError:
        return None

    m = PWPTraj()
    m.times = list(map(float, pwp.times))
    for cx, cy, cz in zip(pwp.coeff_x, pwp.coeff_y, pwp.coeff_z):
        m.coeff_x.append(CoeffPoly3(a=float(cx[0]), b=float(cx[1]), c=float(cx[2]), d=float(cx[3])))
        m.coeff_y.append(CoeffPoly3(a=float(cy[0]), b=float(cy[1]), c=float(cy[2]), d=float(cy[3])))
        m.coeff_z.append(CoeffPoly3(a=float(cz[0]), b=float(cz[1]), c=float(cz[2]), d=float(cz[3])))
    return m


# 反方向: PWPTraj 消息解回 PieceWisePol
def msg_to_pwp(msg) -> PieceWisePol:
    pwp = PieceWisePol()
    pwp.times = list(msg.times)
    for cx in msg.coeff_x:
        pwp.coeff_x.append(np.array([cx.a, cx.b, cx.c, cx.d]))
    for cy in msg.coeff_y:
        pwp.coeff_y.append(np.array([cy.a, cy.b, cy.c, cy.d]))
    for cz in msg.coeff_z:
        pwp.coeff_z.append(np.array([cz.a, cz.b, cz.c, cz.d]))
    return pwp


# ---------------------------------------------------------------------------
# Path simplification helpers (collinear / corner removal)
# ---------------------------------------------------------------------------

# 去掉「在同一条直线上」的中间点: 若 a->b 和 b->c 几乎同向, b 是多余拐点, 删掉。
# 判据用叉积模长除以两段长(等于夹角的正弦), 小于 tol 视为共线。
def remove_collinear(points: List[np.ndarray], tol: float = 1e-3) -> List[np.ndarray]:
    if len(points) <= 2:
        return [p.copy() for p in points]
    out = [points[0].copy()]
    for i in range(1, len(points) - 1):
        a = out[-1]
        b = points[i]
        c = points[i + 1]
        ab = b - a
        bc = c - b
        n_ab = np.linalg.norm(ab)
        n_bc = np.linalg.norm(bc)
        if n_ab < 1e-9 or n_bc < 1e-9:
            continue
        cross = np.linalg.norm(np.cross(ab, bc))
        # 夹角的正弦超过阈值 = 确实在拐弯, 保留这个点
        if cross / (n_ab * n_bc) > tol:
            out.append(b.copy())
    out.append(points[-1].copy())
    return out


# 合并过短的边: 离上一个保留点不足 min_len 的点一般丢掉, 让路径稀疏一点。
# is_blocked(p,q) 是可选的「这两点直连会不会撞障碍」回调; 一旦丢点会导致跨过障碍, 就保留它(安全优先)。
def collapse_short_edges(points: List[np.ndarray], min_len: float,
                         is_blocked=None) -> List[np.ndarray]:
    if not points:
        return []
    out = [points[0].copy()]
    n = len(points)
    for i in range(1, n):
        p = points[i]
        if np.linalg.norm(p - out[-1]) >= min_len:
            out.append(p.copy())
        elif is_blocked is not None and i + 1 < n and is_blocked(out[-1], points[i + 1]):
            # Dropping p would merge across an obstacle -> keep it (safety over brevity).
            out.append(p.copy())
    # 收尾: 保证终点一定在结果里(末点被压短丢了就补回/替换上)
    if len(out) >= 2 and np.linalg.norm(out[-1] - points[-1]) > 1e-9:
        out[-1] = points[-1].copy()
    elif len(out) == 1:
        out.append(points[-1].copy())
    return out


# 按转角和段长稀疏化顶点: 转弯太小(几乎直行)或某段太短的中间点, 视为可丢。
# 同样有 is_blocked 安全闸: 丢点后若 a->c 直连会撞障碍, 就不丢。
def angle_spacing_filter(points: List[np.ndarray], min_turn_deg: float,
                         min_seg_len: float, is_blocked=None) -> List[np.ndarray]:
    if len(points) <= 2 or min_turn_deg <= 0:
        return [p.copy() for p in points]
    # 把「最小转角」换成余弦阈值, 比较 cos 比反算角度快
    cos_thr = math.cos(math.radians(min_turn_deg))
    out = [points[0].copy()]
    for i in range(1, len(points) - 1):
        a = out[-1]
        b = points[i]
        c = points[i + 1]
        ab = b - a
        bc = c - b
        nab = np.linalg.norm(ab)
        nbc = np.linalg.norm(bc)
        want_skip = False
        if nab < min_seg_len or nbc < min_seg_len:
            want_skip = True
        else:
            cosang = float(np.dot(ab, bc) / (nab * nbc))
            if cosang > cos_thr:  # big cosang (close to 1) means tiny turn
                want_skip = True
        # Never drop b if the merged segment a->c would clip an obstacle.
        if want_skip and (is_blocked is None or not is_blocked(a, c)):
            continue
        out.append(b.copy())
    out.append(points[-1].copy())
    return out
