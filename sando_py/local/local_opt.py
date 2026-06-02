"""Local trajectory optimisation: B-spline + L-BFGS-B + detour multi-start.

============================================================================
中文说明（这个文件在整个规划器里干什么）：
  这是「局部轨迹优化」环节。上游 heat-A*（hgp）给一条折线（粗糙的全局向导），
  这里把它变成一条又平滑又安全的飞行曲线。

  文件里其实有两套并行的实现，共用同一套「多起点 + 可行性挑选」的外壳：
    1. plan()        —— 旧版：用 B 样条曲线 + scipy 的 L-BFGS-B 数值优化。
    2. plan_minco()  —— 新版（当前主线）：用 MINCO（最小急动度五次曲线），
                        梯度是解析推导出来的（不是数值差分），更快更准；
                        而且按障碍「类别」分别处理：
                          - 人（human）= HARD 硬约束：用「控制点凸包 + 增广拉格朗日
                            (ALM)」把曲线挡在人之外，还考虑时间（space-time，
                            人会动，要在「正确的时刻」躲开）。
                          - 墙（wall）  = SOFT 软约束：用 EGO 那种「势场」，
                            离得近就推一把，但允许蹭着走。

  关键输入：A* 折线 astar_path、障碍列表 obstacles、每类障碍的避障参数 avoid_cfg。
  关键输出：(轨迹对象, info 字典)。info 里把两件事分开报：
        trajectory_valid   —— 轨迹「业务上」是否安全（离障碍够远、速度/加速度不超限）
        optimizer_success  —— scipy 数值求解器是否收敛（两者互相独立！求解器收敛
                              不代表轨迹安全，反之亦然）

  名词速记：
    MINCO   = 一种把「内部路点 q + 每段时间 T」当自由变量、内部用带状方程
              M(T)c=b 解出多项式系数的轨迹表示（min-jerk 五次）。
    ALM/PHR = 增广拉格朗日法，把硬约束变成可以用普通无约束优化器解的罚函数，
              外层循环慢慢加大罚重 rho、更新乘子 lambda，直到约束满足。
    EGO 场  = 一种软避障代价，离障碍越近代价越高（这里用三次方 hinge）。
============================================================================

Pipeline (per replan):
  1. take A* post-processed polyline + obstacles + per-class avoid config
  2. detour seed generation:
        original LSQ seed
        + (for top_k violating obstacles) push waypoints by ±u/±v
          where u,v are two orthogonal directions perpendicular to the
          obstacle-to-path axis. Each detour pushes by (r + d_safe + margin).
  3. for each seed, LSQ-fit a clamped quintic B-spline → L-BFGS-B over
        (interior ctrl, dt) → check_feasibility on the result
  4. select best:
        if any candidate is feasible (clearance / vel / accel within tol):
            pick min final_cost
        else:
            pick "least bad" by (max_clearance_violation, vel_violation, cost)
  5. return (spline, info) where info splits:
        trajectory_valid   ← business feasibility
        optimizer_success  ← scipy convergence flag (independent!)

Cost (per seed):
    obstacle_cost (EGO cubic, per class)
  + w_smooth · mean(||snap||²)
  + w_vel    · mean(max(0, |v| - vmax))²
  + w_accel  · mean(max(0, |a| - amax))²
  + w_time   · ((T - T_target) / T_target)²
        (without this anchor, dt blows up because smooth/vel/accel cost ~ 1/dt²)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.optimize

from .avoid_config import AvoidParams, resolve_mode
from .bspline import UniformBSpline
from .cost import obstacle_cost
from .minco import MinjerkTraj, C2B


@dataclass
class OptParams:
    """优化器的全部超参数（代价权重、限速、ALM 设置、space-time 设置等）。
    一个地方集中放，plan()/plan_minco() 都从这里读。"""
    # w_* 是各项代价的权重；w_time 是「时间锚」——不加它的话 dt 会被优化器无脑拉大，
    # 因为平滑/速度/加速度代价都 ~ 1/dt^2，时间越长代价越小，会跑飞。
    w_smooth: float = 1.0
    w_vel: float = 100.0
    w_accel: float = 100.0
    w_time: float = 10.0
    vmax: float = 3.0              # 速度上限 (m/s)
    amax: float = 3.0              # 加速度上限 (m/s^2)
    dt_min: float = 0.05           # 每段时间的下限，防止 dt 被压到 0 出数值病态
    K: int = 200                   # 沿轨迹采样点数（算代价、查可行性时用）
    maxiter: int = 200             # 求解器最大迭代次数
    degree: int = 5                # B 样条阶数（5 = 五次曲线）
    # feasibility tolerances (Q4 of Step 1 spec)
    # 可行性判据的容差：留一点点余量，免得差零点几就被判失败
    clearance_tol: float = 0.05    # accept if min_clearance ≥ d_safe - tol（离障碍够远）
    vel_tol: float = 0.10          # accept if max_v ≤ vmax · (1 + tol)（速度别超太多）
    accel_tol: float = 0.10        # accept if max_a ≤ amax · (1 + tol)（加速度别超太多）
    # --- MINCO path additions (plan_minco only; unused by the B-spline plan) ---
    # 下面这几个只给新版 plan_minco 用，旧版 B 样条 plan() 用不到
    w_obs: float = 1.0             # MINCO 代价里软障碍项的权重
    kappa: int = 16                # 每段时间积分的采样点数（中点求积，越多越准越慢）
    M_cap: int = 12                # MINCO 种子最多切几段（内部路点数的上限）
    # --- Stage 3: per-class HARD / SOFT + augmented-Lagrangian (ALM) ---
    # avoid_override: None -> per-class (mode from avoid_cfg[class].mode, fail-safe HARD);
    #                 'soft' -> force ALL obstacles soft (EGO field, ALM off) — the M2 baseline arm;
    #                 'hard' -> force ALL obstacles into the ALM convex-hull constraint (incl. walls).
    # avoid_override：避障模式总开关。None = 按每类障碍自己的 mode 走（人硬/墙软，
    #   读不到配置就保守地当硬约束）；'soft' = 强制所有障碍都走软场（关掉 ALM，
    #   这是论文里的对照组 M2 baseline）；'hard' = 强制所有障碍（连墙）都进 ALM 硬约束。
    avoid_override: object = None
    # ALM（增广拉格朗日）外层循环的设置：
    alm_rho0: float = 10.0         # 罚重 rho 的初值
    alm_rho_max: float = 1.0e8     # rho 上限，太大内层求解会病态
    alm_rho_grow: float = 2.0      # 外层进展卡住时，rho 乘这个倍数放大
    alm_outer_iters: int = 12      # 外层循环上限；到了还在违反约束就停（判为 STOP）
    alm_viol_tol: float = 1.0e-3   # 最大约束违反量小于它就认为 ALM 收敛
    alm_inner_maxiter: int = 40    # 每次内层 L-BFGS-B 的迭代上限（PHR 热启动效果好，不用太多）
    # --- Stage 4: per-control-point SPACE-TIME ALM (S4a) + trust window (S4b) ---
    # tau_trust: seconds of prediction trust AHEAD of the replan instant (offline
    #   t_now=0). A control point with wall-clock t_{i,k} <= tau_trust is HARD
    #   (ALM); beyond it the prediction is stale -> the point is DEMOTED from the
    #   hard set and covered by the SOFT EGO field instead (no double-count).
    # spacetime_hard: master switch. When False (or the human is static, vel==0)
    #   the EXACT Stage-3 segment-END single-time path runs (byte-identical).
    #   The _is_moving gate makes static humans Stage-3-identical even when True,
    #   so existing static tests are unaffected by the default-on switch.
    # tau_trust：从「这次重规划的此刻」往前预测多少秒算「可信」。一个控制点的墙上
    #   钟时间 t_{i,k} <= tau_trust 才算硬约束（ALM 顶住）；超过这个时间，对人未来位置的
    #   预测就不可信了，这个控制点会从硬约束集合里「降级」，改交给软场 EGO 管（避免重复算）。
    # spacetime_hard：space-time（考虑时间维度躲动态障碍）的总开关。设 False 或人是静止的
    #   （速度=0）时，走的是和 Stage 3 完全一样的「每段只取段末单一时刻」老路径（逐字节一致）。
    #   下面的 _is_moving 门控保证：哪怕这开关默认开着，静止的人也走 Stage 3 老路径，
    #   所以已有的静态测试不受影响。
    tau_trust: float = 0.75
    spacetime_hard: bool = True
    # --- conformal continuous-time certificate (novelty upgrade) ---
    # q_conformal: Q_alpha — the (1-alpha) quantile of the human-prediction
    #   sup-norm error sup_s||c_true(s)-c_pred(s)||, calibrated OFFLINE on a hold-out
    #   set (see test/stage4_conformal_coverage.py). Added to a HARD SPHERE's
    #   effective radius so R = radius + d_safe + Q_alpha. By the certificate
    #   theorem, enforcing the relative margin >= R then guarantees Pr(no collision
    #   over the whole interval) >= 1-alpha against the TRUE (unknown) human path.
    #   It is a CONSTANT w.r.t. the decision variables (q, T): dQ/dq = dQ/dT = 0, so
    #   every analytic gradient is byte-identical and only the constraint/cert LEVEL
    #   shifts (the optimizer simply pushes the trajectory out by Q_alpha more).
    #   Walls (AABB, soft/static) get no Q_alpha. Default 0.0 => deterministic
    #   Stage 4 (all existing behaviour byte-for-byte; this is the ablation OFF arm).
    # q_conformal：Q_alpha——人预测误差 sup_s||真-预测|| 的 (1-alpha) 分位，离线在校准集上
    #   标定（见 test/stage4_conformal_coverage.py）。加到硬球体的有效半径上：
    #   R = 半径 + d_safe + Q_alpha。证书定理保证：相对裕度 >= R 时，对人「真实(未知)轨迹」
    #   整段不撞的概率 >= 1-alpha。它对优化变量 (q,T) 是常数（dQ/dq=dQ/dT=0），所以所有解析
    #   梯度逐字节不变、只是把约束/证书的「水位」抬高（优化器只是把轨迹多推出去 Q_alpha）。
    #   墙(AABB，软/静态)不加。默认 0.0 = 确定性 Stage 4（旧行为逐字节一致；这就是消融的 OFF 臂）。
    q_conformal: float = 0.0
    # --- execution gap: plan -> flown tube (epsilon_track) ---
    # epsilon_track: a conservative upper bound on how far the FLOWN trajectory deviates
    #   from the PLANNED one (tracking error + state-estimation error + replan latency).
    #   The certificate is about the PLAN p(s); the drone flies p_actual(s) with
    #   ||p_actual(s)-p(s)|| <= epsilon_track. Adding it to R upgrades the guarantee from
    #   "the PLAN clears" to "the flown TUBE clears": plan margin >= R = r + d_safe + Q +
    #   epsilon_track => flown margin >= plan margin - epsilon_track - Q >= r + d_safe.
    #   UNLIKE Q_alpha (human-prediction error, SPHERES only), epsilon_track is the DRONE's
    #   own position uncertainty -> obstacle-agnostic, so it inflates EVERY hard obstacle
    #   (spheres AND hard AABB walls). Like Q_alpha it is CONSTANT in (q, T)
    #   (deps/dq = deps/dT = 0) -> no gradient term, only the constraint/cert level shifts.
    #   Default 0.0 (plan == flown, the ablation OFF arm; existing behaviour byte-for-byte).
    # epsilon_track：实飞轨迹偏离规划轨迹的保守上界（跟踪误差+状态估计误差+replan 延迟）。
    #   证书是对「规划」p(s) 的，无人机飞的是 p_actual(s)，||p_actual-p|| <= epsilon_track。
    #   把它加进 R，就把保证从「规划清出」升级成「实飞管子清出」：规划裕度 >= R = r+d_safe+Q+eps
    #   ⇒ 实飞裕度 >= 规划裕度 - eps - Q >= r+d_safe。
    #   与 Q_alpha（人预测误差，只对球）不同：epsilon_track 是无人机**自己**的位置不确定度，
    #   与障碍类型无关 → 对**每个**硬障碍都加（球 + 硬 AABB 墙）。同样对 (q,T) 是常数、不动梯度。
    #   默认 0.0（规划=实飞，消融 OFF 臂；旧行为逐字节一致）。
    epsilon_track: float = 0.0


@dataclass
class DetourConfig:
    """「绕行多起点」的设置。思路：A* 那条线如果离障碍太近，就在它两侧各鼓出一条
    备选路线当种子，多个种子各优化一遍，最后挑最好的——避免优化器从一个差起点掉进局部坑。"""
    enabled: bool = True
    directions: int = 4               # 每个障碍最多往 ±u, ±v 四个方向各鼓一条
    trigger_margin: float = 0.10      # m — 折线离障碍 < d_safe + margin 才触发绕行
    detour_margin: float = 0.30       # m — 把路点推到障碍外 (r + d_safe + margin) 处
    top_k_obstacles: int = 1          # 只给违反最严重的前 k 个障碍生成绕行
    max_seeds: int = 5                # 候选种子总数硬上限
    K_eval: int = 100                 # 沿折线采样多少点来查间隙/判断是否触发
    perturb_window: int = 3           # 鼓包时影响的路点半窗宽（前后各几个点跟着挪）


# ---------------------------------------------------------------------------
# obstacle geometry helpers (sphere + AABB)
# ---------------------------------------------------------------------------

def _obstacle_geom(obs) -> Tuple[np.ndarray, float]:
    """返回障碍的 (中心, 等效半径)。球体直接读；长方体(AABB)就用它的外接球近似
    （第一版偷懒，但拿来定个绕行方向够用了）。
    Return (centre, radius_equiv). For AABB we fall back to its bounding sphere
    (first-version simplification — good enough to choose a detour direction)."""
    if hasattr(obs, "centre0"):  # SphereObstacle
        return obs.centre0.copy(), float(obs.radius)
    if hasattr(obs, "lo"):       # AABBObstacle
        centre = 0.5 * (obs.lo + obs.hi)
        radius = 0.5 * float(np.linalg.norm(obs.hi - obs.lo))
        return centre, radius
    raise ValueError(f"unknown obstacle type {type(obs).__name__}")


def _orthonormal_pair(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """给定一个轴 axis，返回两个互相垂直、又都垂直于它的单位向量 (u, v)。
    用来确定「往障碍两侧鼓」的两个正交方向。
    Two unit vectors orthogonal to `axis` and to each other."""
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    a = axis / n
    world = np.eye(3)
    cosines = np.abs(world @ a)
    # 挑一个和 axis 最不平行的世界坐标轴，减去它在 a 上的分量，得到一个垂直方向 u
    pick = int(np.argmin(cosines))   # world axis least aligned with `axis`
    u = world[pick] - float(np.dot(world[pick], a)) * a
    u /= max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(a, u)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    return u, v


# ---------------------------------------------------------------------------
# detour seed generation
# ---------------------------------------------------------------------------

def _resample_polyline(path: np.ndarray, K: int) -> np.ndarray:
    """把折线按弧长（实际走过的距离）均匀重采样成 K 个点。
    用累计长度做线性插值，这样点在空间上分布均匀，不受原折线顶点疏密影响。"""
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total == 0:
        return np.tile(path[0], (K, 1))
    target = np.linspace(0, total, K)
    return np.stack([np.interp(target, cum, path[:, d]) for d in range(path.shape[1])], axis=1)


def _min_clearance_per_obstacle(path: np.ndarray, obstacles, K_eval: int):
    """对每个障碍，算折线离它最近的「带符号距离」以及最近发生在哪个折线顶点附近。
    返回每个障碍一条 (最小间隙, 最近折线点下标)。
    Return list of (min_clearance, closest_path_index) per obstacle."""
    samples = _resample_polyline(path, K_eval)
    out = []
    for obs in obstacles:
        ds = np.array([obs.signed_dist(samples[i], 0.0) for i in range(K_eval)])
        i_min = int(np.argmin(ds))
        # 采样点是在重采样后的密集序列上找的，这里按比例换算回原折线的顶点下标
        # map sample index back to path index (proportional)
        i_path = int(round(i_min * (len(path) - 1) / max(K_eval - 1, 1)))
        out.append((float(ds[i_min]), i_path))
    return out


def _make_detour_path(path: np.ndarray, obstacle, direction: np.ndarray,
                      d_safe: float, detour_cfg: DetourConfig) -> np.ndarray:
    """把折线沿给定方向「鼓」出一个包，绕开这个障碍。
    做法：找到离障碍中心最近的折线点，算出绕行目标点
    (中心 + 方向 · (半径 + 安全距离 + 余量))，然后把附近一窗口的点按钟形权重
    朝目标点挪过去。首尾两端钉死不动（保证起点终点不变）。
    Bulge the polyline around obstacle in the given direction.

    Find the closest path index to the obstacle centre, compute the detour
    target at (centre + direction · (r + d_safe + margin)), then shift a
    window of nearby path points toward that target with a bell-shape weight.
    Endpoints stay pinned.
    """
    centre, r_eq = _obstacle_geom(obstacle)
    dists = np.linalg.norm(path - centre, axis=1)
    i_close = int(np.argmin(dists))
    target = centre + direction * (r_eq + d_safe + detour_cfg.detour_margin)
    shift = target - path[i_close]
    w = detour_cfg.perturb_window
    new_path = path.copy()
    M = len(path)
    for j in range(max(0, i_close - w), min(M, i_close + w + 1)):
        if j == 0 or j == M - 1:
            continue  # 首尾钉死，不挪 / endpoints stay pinned
        # 离最近点越远，挪得越少（钟形/三角权重），形成一个平滑的鼓包
        weight = max(0.0, 1.0 - abs(j - i_close) / (w + 1))
        new_path[j] += shift * weight
    return new_path


def _generate_detour_seeds(astar_path: np.ndarray, obstacles, avoid_cfg, opt: OptParams,
                           detour_cfg: DetourConfig) -> List[Tuple[str, np.ndarray]]:
    """生成候选种子路径列表，第一个永远是原始 A* 折线 'original'。
    只对「离障碍太近」的障碍生成绕行种子，每个种子是一条带名字的折线。
    Build the list of candidate seed paths. Always includes 'original'."""
    seeds: List[Tuple[str, np.ndarray]] = [("original", astar_path)]
    if not detour_cfg.enabled or not obstacles:
        return seeds
    clearances = _min_clearance_per_obstacle(astar_path, obstacles, detour_cfg.K_eval)
    # 挑出「违反者」：折线离它太近（间隙 < d_safe + 触发余量）的障碍
    violators = []
    for i, obs in enumerate(obstacles):
        params = avoid_cfg.get(obs.class_name)
        d_safe = params.d_safe if params is not None else 0.8  # 读不到配置就用保守默认 0.8
        min_clr, i_path = clearances[i]
        violation = (d_safe + detour_cfg.trigger_margin) - min_clr
        if violation > 0:
            violators.append((violation, i, obs, d_safe, i_path))
    if not violators:
        return seeds
    violators.sort(key=lambda v: -v[0])  # 违反最严重的排前面 / most-violated first
    # 只给前 top_k 个最严重的障碍生成绕行
    for violation, i, obs, d_safe, i_path in violators[: detour_cfg.top_k_obstacles]:
        centre, _r = _obstacle_geom(obs)
        path_pt = astar_path[min(i_path, len(astar_path) - 1)]
        # 绕行的「轴」= 从最近折线点指向障碍中心；两侧 ±u/±v 就是垂直于这条轴的方向
        axis = centre - path_pt
        if float(np.linalg.norm(axis)) < 1e-6:
            # 折线点恰好压在障碍中心上，轴退化了，改用「起点->终点」方向兜底
            # path point coincides with obstacle centre — fall back to start-goal axis
            axis = astar_path[-1] - astar_path[0]
        u, v = _orthonormal_pair(axis)
        dirs = [u, -u, v, -v][: detour_cfg.directions]
        for j, d in enumerate(dirs):
            new_path = _make_detour_path(astar_path, obs, d, d_safe, detour_cfg)
            seeds.append((f"detour_obs{i}_dir{j}", new_path))
            if len(seeds) >= detour_cfg.max_seeds:  # 到种子上限就停，别生成太多拖慢求解
                return seeds
    return seeds


# ---------------------------------------------------------------------------
# feasibility check (Step 3: independent of optimizer flag)
# ---------------------------------------------------------------------------

def check_feasibility(spline, obstacles, avoid_cfg,
                      opt: OptParams, K_eval: int = 200) -> dict:
    """检查轨迹「业务上」是否可行（够安全 + 速度/加速度不超限），和优化器是否收敛无关。
    沿轨迹密集采 K_eval 个点：量到每个障碍的最近间隙、看最大速度/加速度有没有超限，
    任何一项越界就给出 failure_reason，否则 trajectory_valid=True。

    spline 可以是 B 样条(UniformBSpline)，也可以是 MINCO 轨迹(MinjerkTraj)——
    这里只用到它们共有的 .t_start/.t_end/.eval/.eval_deriv 接口，所以两种都能传。
    Compute trajectory-level feasibility, independent of optimiser convergence.

    `spline` may be a UniformBSpline OR a MinjerkTraj — only .t_start/.t_end/
    .eval/.eval_deriv are used, which both expose identically.

    Returns dict with min_clearance / max_clearance_violation / max_v / max_a /
    vel_violation / accel_violation / trajectory_valid / failure_reason.
    """
    ts = np.linspace(spline.t_start, spline.t_end, K_eval)
    pts = spline.eval(ts)
    min_clr = float("inf")
    max_violation = 0.0
    for obs in obstacles:
        params = avoid_cfg.get(obs.class_name)
        d_safe = params.d_safe if params is not None else 0.8  # 读不到配置就保守用 0.8
        # 注意 signed_dist 带时间参数 ts[i]：动态障碍在不同时刻位置不同
        for i in range(K_eval):
            d = obs.signed_dist(pts[i], float(ts[i]))
            if d < min_clr:
                min_clr = d
            violation = d_safe - d
            if violation > max_violation:
                max_violation = violation
    if min_clr == float("inf"):
        min_clr = 0.0  # no obstacles to measure against
    v_mag = np.linalg.norm(spline.eval_deriv(ts, 1), axis=1)
    a_mag = np.linalg.norm(spline.eval_deriv(ts, 2), axis=1)
    max_v = float(v_mag.max())
    max_a = float(a_mag.max())
    vel_violation = max(0.0, max_v - opt.vmax)
    accel_violation = max(0.0, max_a - opt.amax)

    # 按优先级定失败原因：先看离障碍够不够远，再看速度，再看加速度（带容差）
    failure_reason = None
    if obstacles and max_violation > opt.clearance_tol:
        failure_reason = "clearance_violation"
    elif max_v > opt.vmax * (1.0 + opt.vel_tol):
        failure_reason = "velocity_violation"
    elif max_a > opt.amax * (1.0 + opt.accel_tol):
        failure_reason = "acceleration_violation"

    return {
        "min_clearance": float(min_clr),
        "max_clearance_violation": float(max_violation),
        "max_v": max_v,
        "max_a": max_a,
        "vel_violation": float(vel_violation),
        "accel_violation": float(accel_violation),
        "trajectory_valid": failure_reason is None,
        "failure_reason": failure_reason,
    }


# ---------------------------------------------------------------------------
# inner optimisation primitives (shared)
# ---------------------------------------------------------------------------

def _build_spline(x: np.ndarray, num_ctrl: int, p: int,
                  start_pt: np.ndarray, end_pt: np.ndarray) -> UniformBSpline:
    """把优化变量 x = [内部控制点拉平..., dt] 还原成一条 B 样条。
    首尾各 p+1 个控制点固定成 start/end（clamped，保证起终点不动），中间是自由变量。"""
    dt = float(x[-1])
    interior = x[:-1].reshape(-1, 3)
    ctrl = np.empty((num_ctrl, 3), dtype=np.float64)
    ctrl[: p + 1] = start_pt           # 头部 p+1 个重复 = 钉住起点
    ctrl[p + 1 : -(p + 1)] = interior
    ctrl[-(p + 1) :] = end_pt          # 尾部 p+1 个重复 = 钉住终点
    return UniformBSpline(ctrl, degree=p, dt=dt)


def _total_cost(x: np.ndarray, num_ctrl: int, p: int,
                start_pt: np.ndarray, end_pt: np.ndarray,
                obstacles, avoid_cfg, opt: OptParams, T_target: float) -> float:
    """B 样条路径的总代价（标量），给 scipy 数值优化器当目标函数。
    = 障碍代价 + 平滑(snap^2) + 超速罚 + 超加速度罚 + 时间锚。注意这版没有解析梯度，
    L-BFGS-B 自己做有限差分；MINCO 那条路径才有解析梯度。"""
    bs = _build_spline(x, num_ctrl, p, start_pt, end_pt)
    K = opt.K
    ts = np.linspace(bs.t_start, bs.t_end, K)
    c_obs = obstacle_cost(bs, obstacles, avoid_cfg, K=K)
    # 平滑项：snap = 四阶导（急动度的导数），平方平均，越小曲线越顺
    snap = bs.eval_deriv(ts, order=4)
    c_smooth = opt.w_smooth * float(np.mean(np.sum(snap * snap, axis=1)))
    # 超速罚：只罚超过 vmax 的部分（hinge），平方
    vel = bs.eval_deriv(ts, order=1)
    v_mag = np.linalg.norm(vel, axis=1)
    v_over = np.maximum(v_mag - opt.vmax, 0.0)
    c_vel = opt.w_vel * float(np.mean(v_over * v_over))
    accel = bs.eval_deriv(ts, order=2)
    a_mag = np.linalg.norm(accel, axis=1)
    a_over = np.maximum(a_mag - opt.amax, 0.0)
    c_accel = opt.w_accel * float(np.mean(a_over * a_over))
    # 时间锚：把总时长拉回 T_target 附近，防止 dt 失控（见文件头说明）
    T_actual = (num_ctrl - p) * float(x[-1])
    rel = (T_actual - T_target) / max(T_target, 1e-6)
    c_time = opt.w_time * rel * rel
    return c_obs + c_smooth + c_vel + c_accel + c_time


def _optimise_one(seed_path: np.ndarray, obstacles, avoid_cfg, opt: OptParams,
                  num_ctrl: int) -> dict:
    """从一条种子折线出发，跑一次 B 样条 L-BFGS-B 优化，返回这个候选的记录。
    Run one LBFGS-B from a given seed polyline. Return candidate record."""
    p = opt.degree
    seg_lens = np.linalg.norm(np.diff(seed_path, axis=0), axis=1)
    path_len = float(seg_lens.sum())
    # 初始总时长按「路径长度 / 限速」估，再分摊到每段得到 dt0（不低于 dt_min 两倍）
    T0 = path_len / max(opt.vmax, 1e-6)
    dt0 = max(T0 / max(num_ctrl - p, 1), opt.dt_min * 2.0)
    bs0 = UniformBSpline.fit_path(seed_path, num_ctrl=num_ctrl, degree=p,
                                  dt=dt0, clamp_endpoints=True)
    start_pt = seed_path[0].copy()
    end_pt = seed_path[-1].copy()
    # 优化变量 x = [内部控制点拉平..., dt]，最后一维 dt 受 dt_min 下界约束
    interior_init = bs0.ctrl[p + 1 : -(p + 1)].copy().ravel()
    x0 = np.concatenate([interior_init, [dt0]])
    bounds = [(None, None)] * (x0.size - 1) + [(opt.dt_min, None)]
    T_target = path_len / max(opt.vmax, 1e-6)
    args = (num_ctrl, p, start_pt, end_pt, list(obstacles), avoid_cfg, opt, T_target)
    init_cost = _total_cost(x0, *args)
    result = scipy.optimize.minimize(
        _total_cost, x0, args=args, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": opt.maxiter, "gtol": 1e-6},
    )
    bs = _build_spline(result.x, num_ctrl, p, start_pt, end_pt)
    feas = check_feasibility(bs, obstacles, avoid_cfg, opt)
    return {
        "spline": bs,
        "init_cost": float(init_cost),
        "final_cost": float(result.fun),
        "iter": int(result.nit),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "dt0": float(dt0),
        "dt_final": float(result.x[-1]),
        "T_target": float(T_target),
        "T_final": float((num_ctrl - p) * float(result.x[-1])),
        "feasibility": feas,
    }


def _select_best(candidates: List[dict]) -> dict:
    """从多个种子优化结果里挑最好的（B 样条版）。
    可行优先：只要有可行的，就在可行里挑代价最低的；一个可行的都没有，就挑「最不糟」的
    （按 间隙违反量 -> 速度违反量 -> 代价 依次比较）。
    Feasible-first; failing that, the 'least bad' candidate."""
    feasible = [c for c in candidates if c["feasibility"]["trajectory_valid"]]
    if feasible:
        return min(feasible, key=lambda c: c["final_cost"])
    return min(candidates, key=lambda c: (
        c["feasibility"]["max_clearance_violation"],
        c["feasibility"]["vel_violation"],
        c["final_cost"],
    ))


def _select_best_minco(candidates: List[dict]) -> dict:
    """从多个种子结果里挑最好的（MINCO 版）。和 _select_best 的区别：撞了人(HARD)的
    候选永远不能算「可行」；都不可行时，「最不糟」的排序把『撞人 hard_max_breach』放在
    最前面比（因为违反的约束是个人，最要命），然后才是软墙间隙、速度、代价——
    这就是显式的 STOP 优先级。"""
    feasible = [c for c in candidates if c["feasibility"]["trajectory_valid"]]
    if feasible:
        return min(feasible, key=lambda c: c["final_cost"])
    return min(candidates, key=lambda c: (
        max(c.get("hard_max_breach", float("-inf")), 0.0),
        c["feasibility"]["max_clearance_violation"],
        c["feasibility"]["vel_violation"],
        c["final_cost"],
    ))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def plan(astar_path: np.ndarray,
         obstacles: Iterable,
         avoid_cfg: Dict[str, AvoidParams],
         *,
         num_ctrl: int = None,
         opt_params: OptParams = None,
         detour_cfg: DetourConfig = None) -> Tuple[UniformBSpline, dict]:
    """【旧版 B 样条对外接口】用五次 B 样条 + 绕行多起点 + 可行性挑选规划一条轨迹。
    （新版主线是下面的 plan_minco；这个保留作对照/历史。）
    Plan a quintic B-spline with detour multi-start + feasibility selection.

    Returns (spline, info). `info` has these keys:
        trajectory_valid    (bool) — business feasibility (clearance / vel / accel)
        optimizer_success   (bool) — scipy L-BFGS-B convergence (independent)
        failure_reason      (str|None) — 'clearance_violation' / 'velocity_violation' /
                                          'acceleration_violation'
        feasibility         (dict) — full feasibility record (see check_feasibility)
        seed_used           (str)  — name of the seed that won selection
        n_seeds_tried       (int)
        init_cost, final_cost, iter, dt0, dt_final, T_target, T_final, num_ctrl
        message             — scipy termination message
        converged           — alias of optimizer_success (legacy, will be removed)
    """
    opt = opt_params or OptParams()
    detour_cfg = detour_cfg or DetourConfig()
    p = opt.degree
    astar_path = np.asarray(astar_path, dtype=np.float64)
    if astar_path.ndim != 2 or astar_path.shape[0] < 2:
        raise ValueError(f"astar_path must be (M≥2, d), got {astar_path.shape}")
    # 控制点数：默认大约取折线点数的一半，但至少要够 clamped 五次样条用（头尾各 p+1 个）
    if num_ctrl is None:
        num_ctrl = max(2 * (p + 1), int(round(len(astar_path) / 2)))
    if num_ctrl < 2 * (p + 1) + 1:
        num_ctrl = 2 * (p + 1) + 1

    obstacles_list = list(obstacles)
    # 第一步：生成种子（原始 + 若干绕行）
    seeds = _generate_detour_seeds(astar_path, obstacles_list, avoid_cfg, opt, detour_cfg)

    # 第二步：每个种子各优化一遍；某个种子数值上炸了就跳过，继续别的
    candidates: List[dict] = []
    for name, seed_path in seeds:
        try:
            rec = _optimise_one(seed_path, obstacles_list, avoid_cfg, opt, num_ctrl)
        except Exception as e:
            # one seed failed numerically — note it, continue with the rest
            continue
        rec["seed_name"] = name
        candidates.append(rec)
    if not candidates:
        raise RuntimeError("all seeds failed to optimise")

    # 第三步：挑最好的
    best = _select_best(candidates)
    feas = best["feasibility"]
    info = {
        "trajectory_valid": feas["trajectory_valid"],
        "optimizer_success": best["optimizer_success"],
        "failure_reason": feas["failure_reason"],
        "feasibility": feas,
        "seed_used": best["seed_name"],
        "n_seeds_tried": len(candidates),
        "init_cost": best["init_cost"],
        "final_cost": best["final_cost"],
        "iter": best["iter"],
        "dt0": best["dt0"],
        "dt_final": best["dt_final"],
        "T_target": best["T_target"],
        "T_final": best["T_final"],
        "num_ctrl": int(num_ctrl),
        "message": best["message"],
        "converged": best["optimizer_success"],  # legacy alias
    }
    return best["spline"], info


# ===========================================================================
# 【下面是新版主线：MINCO 解析梯度求解路径 (M2)】
# 和上面的 B 样条 plan() 完全独立，互不影响。
# 优化变量 x = [内部路点 q 拉平 ((M-1)*3 个), 每段时间 T (M 个)]。
# 总代价 = 平滑能量 + 软障碍 + 超速罚 + 超加速度罚 + 时间锚，
#   而且对 (q, T) 的梯度全是「解析」推出来的（不是数值差分），所以又快又准。
#
# 障碍/速度/加速度这些是「沿时间的积分」，用每段中点求积近似（GCOPTER 那套）：
#   第 i 段在 tau = s_j·T_i 处采样，s_j=(j+0.5)/kappa，求积权重 T_i/kappa。
# 对时间 T 的梯度有两条来源（这是最容易算错的地方）：
#   隐式(IMPLICIT)：c = M(T)^{-1} b，系数 c 本身依赖 T   -> grad_from_dcost_dc(...)
#   显式(EXPLICIT)：采样时刻在动 + 求积权重在变           -> dcost_dT_explicit，见下
# ===========================================================================
# Cost = w_smooth·energy + w_obs·obstacle + w_vel·vel-hinge + w_accel·acc-hinge
#        + w_time·time-anchor, all with ANALYTIC gradient w.r.t. (q, T).
#
# Obstacle/vel/accel are time-integrals via per-segment midpoint quadrature
# (GCOPTER style): segment i sampled at tau = s_j·T_i, s_j=(j+0.5)/kappa, with
# quadrature weight T_i/kappa. The gradient w.r.t. T has TWO paths:
#   IMPLICIT (c = M(T)^{-1} b)            -> grad_from_dcost_dc(dcost_dc, ...)
#   EXPLICIT (moving sample times + qweight) -> dcost_dT_explicit, see below.
# ===========================================================================


def _signed_dist_and_grad(obs, p: np.ndarray, t: float):
    """单点版：解析地算「点 p 在时刻 t 离障碍 obs 的带符号距离 d，以及 d 对位置的梯度
    dd/dp 和对时间的梯度 dd/dt」。球体和长方体(AABB)分开处理。
    注意：调用方要保证采样点别落在不可导的拐点上（球心处、长方体表面/内部的棱角）。
    Return (d, dd/dp (3,), dd/dt scalar) analytically for sphere / AABB.

    Sphere(moving): d=||p-c(t)||-r, c(t)=centre0+vel·t.
        dd/dp = (p-c)/||p-c||,  dd/dt = -(p-c)/||p-c|| · vel = -(dd/dp)·vel.
    AABB outside: d=||outside_vec||, dd/dp = outside_vec/||outside_vec|| (static, dd/dt=0).
    AABB inside : d = L∞ penetration (negative); subgradient ±e on the active axis.
    Caller must keep samples off the kinks (||p-c||=0, ||outside||=0, AABB face/interior)."""
    # 球体（可能在动）：d = ||p - c(t)|| - r，c(t) = predict(t)（CA：centre0+vel·t+0.5·accel·t^2）
    if hasattr(obs, "centre0"):  # SphereObstacle
        c = obs.predict(float(t))
        diff = np.asarray(p, dtype=np.float64) - c
        nrm = float(np.linalg.norm(diff))
        n = diff / nrm                 # 单位外法向，也是 dd/dp
        d = nrm - obs.radius
        # 障碍在动 -> 距离随时间的变化率，用瞬时速度 velocity_at(t)=vel+accel·t（CA）
        dd_dt = -float(n.dot(obs.velocity_at(float(t))))
        return d, n, dd_dt
    # 长方体 AABB（静止 -> dd/dt = 0）
    p = np.asarray(p, dtype=np.float64)
    # over = 每个轴上「超出盒子」的量（在盒外才 > 0）
    over = np.maximum(obs.lo - p, 0.0) + np.maximum(p - obs.hi, 0.0)
    if np.any(over > 0.0):
        # 在盒外：d = 到盒子的欧氏距离 = ||over||
        nrm = float(np.linalg.norm(over))
        # dd/dp_axis = 被违反那个面的符号：p>hi 取 +1，p<lo 取 -1
        sign = np.where(p > obs.hi, 1.0, 0.0) - np.where(p < obs.lo, 1.0, 0.0)
        dd_dp = (over * sign) / nrm  # over 是 |违反量|，乘 sign 还原带方向
        return nrm, dd_dp, 0.0
    # 在盒内：d = 各轴 max(lo-p, p-hi) 的最大值（<=0，负的表示穿透深度）；用 L∞ 次梯度
    cand = np.maximum(obs.lo - p, p - obs.hi)  # per-axis penetration (signed, <=0)
    ax = int(np.argmax(cand))
    grad = np.zeros(3)
    # 在 ax 这个轴上比 lo-p 和 p-hi 谁大，谁就决定次梯度方向
    grad[ax] = -1.0 if (obs.lo[ax] - p[ax]) >= (p[ax] - obs.hi[ax]) else 1.0
    return float(cand[ax]), grad, 0.0


def _seg_vander(taus: np.ndarray):
    """造「幂基」矩阵：在一组局部时刻 taus 上，把 0~3 阶导数对多项式系数的依赖写成矩阵行。
    返回 (B0,B1,B2,B3)，每个形状 (kappa,6)。B_o 这一阶满足：deriv^(o) = B_o @ c。
    第 p 列、第 o 阶的值 = (p!/(p-o)!)·tau^(p-o)，就是对 tau 求 o 次导后剩下的多项式。
    Power-basis rows for orders 0..3 at local times taus (kappa,).

    Returns (B0,B1,B2,B3) each (kappa,6): B_o[j] = _basis(taus[j], o).
    Vectorised over the kappa samples (no per-point scalar _basis): column p of
    order o is (p!/(p-o)!)*tau^(p-o), using the same `**` power for bit-parity."""
    taus = np.asarray(taus, dtype=np.float64).ravel()
    k = taus.size
    out = []
    for o in range(4):
        B = np.zeros((k, 6), dtype=np.float64)
        for p in range(o, 6):
            coeff = 1.0
            for m in range(o):
                coeff *= (p - m)
            B[:, p] = coeff * (taus ** (p - o))
        out.append(B)
    return out[0], out[1], out[2], out[3]


_BASIS_CACHE = {}


def _fixed_basis(kappa: int):
    """带缓存的「固定比例基」：求积点取每段的等分中点比例 s=(j+0.5)/kappa（与段长无关），
    在这些比例上预算一次幂基 Bs=_seg_vander(s) 存起来。每段只要按各自 T_i 缩放即可（见
    _scale_basis），不用每段重造，省时间。idx[o][p]=max(p-o,0) 记录缩放时该乘 T_i 的几次方。
    Cached fixed-fraction basis: s=(j+0.5)/kappa and Bs_o=_seg_vander(s).

    Returns (s, (Bs0,Bs1,Bs2,Bs3), idx) where idx[o][p]=max(p-o,0) selects the
    Ti-power that scales column p of order o (see _scale_basis)."""
    if kappa not in _BASIS_CACHE:
        s = (np.arange(kappa) + 0.5) / kappa
        Bs = _seg_vander(s)
        idx = [np.array([max(p - o, 0) for p in range(6)]) for o in range(4)]
        _BASIS_CACHE[kappa] = (s, Bs, idx)
    return _BASIS_CACHE[kappa]


def _scale_basis(Bs, idx, Ti: float):
    """把「固定比例基」Bs（在 tau=s 上算的）缩放到实际段时长 Ti 上。
    因为 tau = s·Ti，所以 B_o 的第 p 列要乘 Ti^(p-o)。代替每段重新跑 _seg_vander。
    Scale fixed-fraction basis Bs (at tau=s) to segment time Ti.

    B_o(s·Ti)[:,p] = Bs_o[:,p]·Ti^(p-o).  Columns p<o are zero in Bs, so their
    Ti^0=1 scaling is harmless. Replaces a per-segment _seg_vander rebuild."""
    tp = np.array([1.0, Ti, Ti * Ti, Ti**3, Ti**4, Ti**5])
    return tuple(Bs[o] * tp[idx[o]][None, :] for o in range(4))


def _minco_obstacle_term(tr: MinjerkTraj, obstacles, avoid_cfg, opt: OptParams):
    """软障碍（EGO 势场）的代价项及其解析梯度。这里的 obstacles 是【软】障碍集合（墙）。
    返回 (代价, 对系数 c 的梯度 (6M,3), 对每段时间 T 的显式梯度 (M,))。

    代价 = Σ_i (T_i/κ) Σ_j Σ_obs W·φ(d)，其中 φ(d)=(d_safe-d)^3 只在 d<d_safe（离得太近）时生效。
    也就是离墙越近罚越多（三次方），离得够远就完全不罚（允许蹭着走，这就是软的意思）。
    (cost, dcost_dc (6M,3), dcost_dT_explicit (M,)) for the obstacle integral.

    cost = Σ_i (T_i/κ) Σ_j Σ_obs W·φ(d),  φ(d)=(d_safe-d)^3 for d<d_safe.
    See header / explicit_dT_term derivation: dcost_dT_explicit collects
      (a) quadrature-weight (1/κ)Σ_j Φ
      (b) moving-local-sample (T_i/κ)Σ_j φ'·(n·vel)·s_j      [tau=s_j·T_i]
      (c1) same-segment abs-time (moving obs) s_j·A_{i,j}
      (c2) cross-segment abs-time (moving obs) A_{i,j} added to every k<i
    where A_{i,j} = (T_i/κ)·W·φ'(d)·dd/dt."""
    M = tr.M
    n = tr.NC * M
    kap = max(int(opt.kappa), 1)
    s, Bs, bidx = _fixed_basis(kap)            # midpoint fractions (κ,) + fixed basis
    cost = 0.0
    dcost_dc = np.zeros((n, 3))
    dT_exp = np.zeros(M)
    if not obstacles:
        return cost, dcost_dc, dT_exp
    cum = tr._cum                              # 各段起始的累计墙上时间
    for i in range(M):                         # 逐段
        Ti = float(tr.T[i])
        taus = s * Ti                          # 这一段内的局部采样时刻
        B0, B1, _, _ = _scale_basis(Bs, bidx, Ti)
        ci = tr.c[6 * i:6 * i + 6]             # (6,3) 这段的多项式系数
        P = B0 @ ci                            # (κ,3) 采样点位置
        V = B1 @ ci                            # (κ,3) 采样点速度
        t_abs = cum[i] + taus                  # (κ,) 每个采样点的绝对（墙上）时刻
        wq = Ti / kap                          # 求积权重
        for obs in obstacles:
            params = avoid_cfg[obs.class_name]
            d_safe = params.d_safe
            W = params.weight
            # 一次性算这 κ 个采样点的 (距离, dd/dp, dd/dt)
            dvec, ndp, dddt = _signed_dist_and_grad_batch(obs, P, t_abs)  # (κ,),(κ,3),(κ,)
            diff = d_safe - dvec               # (κ,) 侵入量，>0 才有罚
            act = diff > 0.0                   # 只在「离得太近」的采样点上算（active）
            if not np.any(act):
                continue
            diff_a = diff[act]
            phi = diff_a ** 3                  # φ = 侵入量^3
            dphi = -3.0 * diff_a * diff_a      # φ'(d)（注意对 d 求导带负号）, (na,)
            B0a = B0[act]; ndpa = ndp[act]; Va = V[act]; sa = s[act]
            cost += wq * W * float(np.sum(phi))
            # 对系数 c 的梯度（隐式那条链）：把每个采样点的贡献按基函数摊回 6 个系数上
            # implicit-c: Σ_j (wq W dphi_j) outer(B0_j, ndp_j)
            coef = (wq * W) * dphi             # (na,)
            dcost_dc[6 * i:6 * i + 6] += B0a.T @ (coef[:, None] * ndpa)
            Phi = W * phi
            # 下面是对段时间 T 的「显式」梯度，三部分（见 docstring 的 a/b/c）：
            # (a) 求积权重 T_i/κ 里那个 T_i 求导
            dT_exp[i] += (1.0 / kap) * float(np.sum(Phi))
            # (b) 采样的局部时刻 tau=s_j·T_i 也随 T_i 变 -> 链式带速度项
            ndp_dot_V = np.einsum("kd,kd->k", ndpa, Va)
            dT_exp[i] += wq * float(np.sum(W * dphi * ndp_dot_V * sa))
            # (c) 障碍在动时，采样点的绝对时刻 t_abs 也依赖各段 T -> 额外的时间项
            dddt_a = dddt[act]
            if np.any(dddt_a != 0.0):
                A = (wq * W) * dphi * dddt_a   # (na,) = wq·∂Φ/∂t_abs per sample
                dT_exp[i] += float(np.sum(sa * A))   # (c1) 本段内（局部分量 s_j）
                if i > 0:
                    dT_exp[:i] += float(np.sum(A))    # (c2) 本段时刻还受前面所有段 T 影响
    return cost, dcost_dc, dT_exp


def _signed_dist_and_grad_batch(obs, P: np.ndarray, t_abs: np.ndarray):
    """_signed_dist_and_grad 的「一批 κ 个点一起算」向量化版本，数学完全一样。
    球体纯 numpy 矢量化；长方体也矢量化了（盒外/盒内分别处理）。
    Vectorised (d (κ,), dd/dp (κ,3), dd/dt (κ,)) over κ sample points P.

    Same math as _signed_dist_and_grad, batched. Sphere is pure-numpy; AABB
    falls back to the per-point scalar routine (rare, cheap)."""
    P = np.asarray(P, dtype=np.float64)
    k = P.shape[0]
    if hasattr(obs, "centre0"):  # sphere (possibly moving, CA model)
        c = obs.predict(t_abs)                                        # (κ,3) centre0+vel·t+0.5·accel·t^2
        diff = P - c
        nrm = np.linalg.norm(diff, axis=1)                            # (κ,)
        n = diff / nrm[:, None]
        d = nrm - obs.radius
        # dd/dt = -n·velocity_at(t)。accel=0 时瞬时速度退回常数 vel，走原来的 n@vel
        # （BLAS 同一算法，逐字节回退）；accel≠0 时瞬时速度逐点变化，逐点点积。
        if not np.any(obs.accel):
            dd_dt = -(n @ obs.vel)                                    # (κ,) CV 回退
        else:
            dd_dt = -np.einsum("kd,kd->k", n, obs.velocity_at(t_abs))  # (κ,) CA
        return d, n, dd_dt
    # AABB (static): vectorised over the k samples, same math as the scalar path.
    lo = obs.lo[None, :]; hi = obs.hi[None, :]
    over = np.maximum(lo - P, 0.0) + np.maximum(P - hi, 0.0)   # (k,3) |violation|
    outside = np.any(over > 0.0, axis=1)                       # (k,)
    d = np.empty(k); ndp = np.zeros((k, 3))
    if np.any(outside):
        ov = over[outside]
        nrm = np.linalg.norm(ov, axis=1)                      # (m,) > 0
        sign = (np.where(P[outside] > hi, 1.0, 0.0)
                - np.where(P[outside] < lo, 1.0, 0.0))         # (m,3)
        d[outside] = nrm
        ndp[outside] = (ov * sign) / nrm[:, None]
    ins = ~outside
    if np.any(ins):
        cand = np.maximum(lo - P[ins], P[ins] - hi)           # (m,3) per-axis (<=0)
        ax = np.argmax(cand, axis=1)                          # L∞ active axis
        m = np.arange(cand.shape[0])
        d[ins] = cand[m, ax]
        loP = (lo - P[ins])[m, ax]; Phi = (P[ins] - hi)[m, ax]
        g = np.zeros((cand.shape[0], 3))
        g[m, ax] = np.where(loP >= Phi, -1.0, 1.0)
        ndp[ins] = g
    return d, ndp, np.zeros(k)


def _minco_dynamic_term(tr: MinjerkTraj, order: int, limit: float,
                        weight: float, opt: OptParams):
    """动力学限制项（超速/超加速度罚）及其解析梯度。order=1 罚速度，order=2 罚加速度。
    被罚量 h = max(||某阶导|| - limit, 0)^2（只罚超出 limit 的部分），代价 = Σ_i (T_i/κ) Σ_j weight·h。
    这是纯几何项（不涉及障碍的绝对时刻），所以对 T 的显式梯度只有 (a) 求积权重 + (b) 采样时刻移动两部分。
    (cost, dcost_dc, dcost_dT_explicit) for a vel(order=1)/accel(order=2) hinge.

    integrand h = max(||deriv||-limit, 0)^2, cost = Σ_i (T_i/κ) Σ_j weight·h.
    ∂h/∂deriv = 2·over·(deriv/||deriv||).  No abs-time path (purely geometric),
    so explicit-T = (a) qweight + (b) moving-sample via dh/dtau (= 2 over (u·nextderiv))."""
    M = tr.M
    n = tr.NC * M
    kap = max(int(opt.kappa), 1)
    s, Bs, bidx = _fixed_basis(kap)
    cost = 0.0
    dcost_dc = np.zeros((n, 3))
    dT_exp = np.zeros(M)
    for i in range(M):
        Ti = float(tr.T[i])
        taus = s * Ti
        B0, B1, B2, B3 = _scale_basis(Bs, bidx, Ti)
        Bo = {0: B0, 1: B1, 2: B2, 3: B3}[order]
        Bnext = {0: B1, 1: B2, 2: B3}[order]    # 被限那阶导再对 tau 求一次导（算 (b) 用）
        ci = tr.c[6 * i:6 * i + 6]
        D = Bo @ ci                              # (κ,3) 被限制的那阶导（速度或加速度）
        Dn = Bnext @ ci                          # (κ,3) 它对 tau 的导数
        wq = Ti / kap
        mag = np.linalg.norm(D, axis=1)          # (κ,) 模长
        over = mag - limit
        act = (over > 0.0) & (mag > 0.0)         # 只在超限且模长非零（可导）的点上算
        if not np.any(act):
            continue
        Da = D[act]; Dna = Dn[act]; Boa = Bo[act]; sa = s[act]
        maga = mag[act]; overa = over[act]
        u = Da / maga[:, None]                    # (na,3) 单位方向
        h = overa * overa
        dh_dderiv = (2.0 * overa)[:, None] * u   # (na,3) ∂h/∂(那阶导)
        cost += wq * weight * float(np.sum(h))
        dcost_dc[6 * i:6 * i + 6] += (wq * weight) * (Boa.T @ dh_dderiv)
        # (a) 求积权重 T_i/κ 求导
        dT_exp[i] += (1.0 / kap) * weight * float(np.sum(h))
        # (b) 采样时刻 tau=s_j·T_i 随 T_i 变：dh/dtau = ∂h/∂导 · d(导)/dtau
        dh_dtau = np.einsum("kd,kd->k", dh_dderiv, Dna)
        dT_exp[i] += wq * weight * float(np.sum(dh_dtau * sa))
    return cost, dcost_dc, dT_exp


# ===========================================================================
# 【Stage 3：HARD（人）的硬约束实现】
# 核心想法：MINCO 每段五次曲线可以写成 6 个 Bernstein 控制点的凸组合，曲线整段都被
# 这 6 个控制点的「凸包」包住。所以只要让每个控制点都离人足够远，整段曲线就一定离人足够远
# （连续时间都保证，不止采样点）。再用增广拉格朗日(PHR)把这些约束变成可优化的罚函数。
#
# Stage 3 — HARD (humans) = continuous-time convex-hull constraint via the
# Bernstein control points + augmented-Lagrangian (PHR) penalty.
#
# For each hard obstacle h, segment i, control point k:
#   g_{i,k,h} = d_safe_h - signed_dist(P_{i,k}, t_rep_i)   (<=0 means clear)
# PHR augmented-Lagrangian term:
#   L_ikh = (rho/2) max(0, lambda/rho + g)^2 - lambda^2/(2 rho)
#   z = lambda + rho*g; if z>0 (active): dL/dg = z, else 0.
#   dg/dP = -dd/dp = -n  (sphere n=(P-c)/||P-c||; AABB-out outside_vec/||.||).
# dL/dP_{i,k} accumulates z*(-n); then chained to dCost/dc and explicit dCost/dT
# through P_i = C2B @ D(T_i) @ c_i (the T_i^j scaling is the explicit-T trap).
# ===========================================================================


def _partition_obstacles(obstacles, avoid_cfg, override):
    """把障碍按类别分成两堆：软的（走 EGO 场）和硬的（走 ALM 凸包约束）。
    分法由 resolve_mode 决定（人=硬，墙=软），override 可以强制全软/全硬。
    Split obstacles into (soft_obs, hard_obs) per resolve_mode + override."""
    soft_obs, hard_obs = [], []
    for o in obstacles:
        if resolve_mode(o, avoid_cfg, override) == "hard":
            hard_obs.append(o)
        else:
            soft_obs.append(o)
    return soft_obs, hard_obs


def _hard_d_safe(obs, avoid_cfg) -> float:
    """取硬障碍的安全距离 d_safe；读不到配置时保守用人的默认值 0.8。
    d_safe for a hard obstacle; fail-safe to the human default 0.8."""
    params = avoid_cfg.get(obs.class_name)
    if params is None:
        return 0.8
    return float(params.d_safe)


def _q_conformal(opt) -> float:
    """conformal 安全裕度 Q_alpha：加到硬球体有效半径上的「预测不确定度余量」。
    它是离线标定的常数（见 OptParams.q_conformal），对优化变量 (q,T) 不依赖
    -> 不动任何解析梯度，只抬高约束/证书水位。opt 为 None（独立 check_grad 闸用）时取 0。
    Conformal margin Q_alpha added to a hard sphere's effective radius. A constant
    w.r.t. (q, T) (see OptParams.q_conformal): leaves all gradients untouched, only
    shifts the constraint/certificate level. 0.0 when opt is None."""
    return float(getattr(opt, "q_conformal", 0.0)) if opt is not None else 0.0


def _eps_track(opt) -> float:
    """执行 gap 的管子半径 epsilon_track：实飞偏离规划的保守上界，加到**每个**硬障碍的
    有效半径上（球 + 墙，因为它是无人机自身位置误差、与障碍类型无关）。和 Q_alpha 一样是
    对 (q,T) 的常数 -> 不动梯度，只抬约束/证书水位。opt 为 None 时取 0。
    Execution-tube radius epsilon_track (plan->flown bound) added to EVERY hard obstacle's
    effective radius. Constant w.r.t. (q, T); 0.0 when opt is None."""
    return float(getattr(opt, "epsilon_track", 0.0)) if opt is not None else 0.0


def _seg_rep_time(tr: MinjerkTraj, i: int) -> float:
    """第 i 段用哪个「代表时刻」来定位人的位置（Stage 3 用法）。取「段末」时刻——
    对一个朝前走的人来说这是最保守的单一时刻；人静止(vel=0)时退化成静态。
    Stage 4 才改成每个控制点用各自的墙上时刻。
    Representative wall-clock time for segment i (Stage 3: segment END — the
    most conservative single time for a forward-moving human, and reduces to
    static for vel=0). Full per-control-point wall-clock is Stage 4."""
    return float(tr._cum[i + 1])


def _is_moving(obs) -> bool:
    """判断这个障碍是不是「会动的人」（球体且速度非零，假设匀速运动 CV）。
    这是个关键「门控」：保证静止的人走的还是 Stage 3 老路径、逐字节一致。
    原因——即使速度=0，按「每个控制点各自法向」算出来的几何方向也和按「段质心法向」
    算的不同，所以静态的人必须强制走旧的「单时刻 + 质心法向」那条路。
    True for a sphere-human with nonzero CV velocity. This is the GATE that
    keeps STATIC humans byte-identical to Stage 3: the per-(i,k) space-time
    normal differs geometrically from the centroid normal even at vel=0, so a
    static human MUST run the old single-time centroid path.

    CA upgrade: a human with nonzero accel (even if vel==0) is moving, so the
    gate also fires on ||accel||>0. vel==0 AND accel==0 -> False (static human
    stays byte-identical to Stage 3)."""
    return (hasattr(obs, "centre0")
            and (float(np.linalg.norm(obs.vel)) > 0.0
                 or float(np.linalg.norm(obs.accel)) > 0.0))


def _trust_mask(tr: MinjerkTraj, opt: OptParams) -> np.ndarray:
    """算「信任窗口」掩码 (M,6) 布尔：某控制点 (i,k) 的墙上时刻 t_{i,k} <= tau_trust 才算
    可信硬约束（True）。离线仿真里 t_now=0，所以直接用 control_point_times()。
    每轮外层 ALM 之间会冻结这个掩码（和支撑半空间法向一样的纪律），只在外层之间重算；
    超出信任窗口的点直接踢出硬约束集合（它的 g/权重 全归零，改交软场管）。
    (M,6) bool: control point (i,k) is trusted-HARD iff t_{i,k} <= tau_trust.

    Offline harness: t_now = 0 so t_{i,k} = control_point_times() directly.
    FROZEN per outer ALM iter (same discipline as the supporting-halfspace
    normals); recomputed only between outer iters. An untrusted point is
    excluded from the hard set entirely (its g/w/S all go to zero)."""
    return tr.control_point_times() <= float(opt.tau_trust)


def _seg_normals(tr: MinjerkTraj, hard_obs, P=None, opt: OptParams = None):
    """算「支撑半空间」的法向（要在每轮外层 ALM 里冻结）。
    直观理解：在控制点和人之间架一道「隔板」（半空间），法向就是从人指向控制点的单位方向。
    把约束写成「控制点要待在隔板外侧」后，约束对控制点是线性的，内层优化才好解。

    两种算法：
      Stage 3（静止的人 / 关掉 space-time）：每段每障碍只用一个法向
          a_{i,h} = 单位(该段6控制点质心 - 人在代表时刻的位置)。
      Stage 4（S4a，会动的人 + 开 space-time）：每个控制点各用各的法向
          a_{i,k,h} = 单位(P_{i,k} - 人在该控制点墙上时刻 t_{i,k} 的位置)，
          也就是「在正确的时刻躲开」的那个更紧的法向。

    不管哪种都统一返回 (M, 6, H, 3)：静态那支只是把每段的质心法向在 6 个控制点上复制一遍
    （几何上和老的 (M,H,3) 完全等价），保证静态障碍的 g 和 dg/dP 一字不变。
    Frozen supporting-halfspace normals per (segment i, control pt k, human h).

    Stage 3 (static human / spacetime off): ONE normal per (segment, obstacle)
        a_{i,h} = unit(centroid_k(P_{i,k}) - c_h(t_rep_i)).
    Stage 4 (S4a, MOVING human + opt.spacetime_hard): per-control-point
        a_{i,k,h} = unit(P_{i,k} - c_h(t_{i,k})) using control_point_times(),
        the "avoid at the right moment" tight normal.

    ALWAYS returns shape (M, 6, H, 3) for a uniform downstream contract: the
    static branch simply broadcasts the per-segment centroid normal across the
    6 control points (geometrically identical to the Stage-3 (M,H,3) form, just
    repeated), so g and dg/dP are unchanged for static obstacles."""
    M = tr.M
    H = len(hard_obs)
    if P is None:
        P = tr.control_points()
    spacetime = opt is not None and bool(opt.spacetime_hard)
    centroid = P.mean(axis=1)                                # (M,3) per-segment centroid
    t_rep = np.array([_seg_rep_time(tr, i) for i in range(M)])  # (M,)
    t_pt = tr.control_point_times()                         # (M,6)
    A = np.zeros((M, 6, H, 3))
    fallback = np.array([1.0, 0.0, 0.0])     # 法向退化（人正好压在点上）时的兜底方向
    for h, obs in enumerate(hard_obs):
        if spacetime and _is_moving(obs):
            # 会动的人：每个控制点按它自己的墙上时刻 t_{i,k} 定位人（CA: predict(t)）
            c = obs.predict(t_pt)                             # (M,6,3)
            a = P - c                                        # (M,6,3)
        else:
            if hasattr(obs, "centre0"):
                c = obs.predict(t_rep)                        # (M,3) CA: centre0+vel·t+0.5·accel·t^2
            else:
                c = np.broadcast_to(0.5 * (obs.lo + obs.hi), (M, 3))   # 墙(AABB)取中心
            a_seg = centroid - c                             # (M,3) 每段一个质心法向
            a = np.broadcast_to(a_seg[:, None, :], (M, 6, 3))  # 复制到 6 个控制点上
        nrm = np.linalg.norm(a, axis=2)                      # (M,6)
        ok = nrm > 1e-12                                     # 法向不退化才归一化，否则用兜底
        A[:, :, h, :] = np.where(ok[:, :, None],
                                 a / np.where(ok[:, :, None], nrm[:, :, None], 1.0),
                                 fallback)
    return A


def _alm_constraints(tr: MinjerkTraj, hard_obs, avoid_cfg, normals=None,
                     opt: OptParams = None, trust_mask=None):
    """逐个算出每条硬约束 g（针对每个「段 i × 控制点 k × 硬障碍 h」的组合）。
    约定 g <= 0 表示「这个点离障碍够远（安全）」，g > 0 表示「侵入了，要被罚」。
    球体用支撑半空间形式（静止/会动两种写法见 docstring）；墙(AABB)直接用带符号距离。
    S4b 信任窗口：超过 tau_trust 的控制点 g 设成一个极大负数 -> 永远不激活（交给软场）。
    返回 (一堆扁平数组的字典, 控制点坐标 P (M,6,3))。其中 n 存的是 dg/dP 的相反数
    （即外法向），这样别处用 dL/dP = -w*n 形式统一；dgdt 是 g 对墙上时刻的导数
    （只有「可信的会动球体」非零）；trust 是每条约束的 S4b 布尔掩码。
    Evaluate every (segment i, ctrl-pt k, hard-obs h) TIGHT constraint.

    SPHERE (static human / spacetime off) -> Stage-3 supporting-halfspace at the
      segment-END single time t_rep_i:
        g = R - a_{i,h}^T (P_{i,k} - c_h(t_rep)),  R = r_h + d_safe, dg/dP = -a.
    SPHERE (MOVING human, opt.spacetime_hard, S4a) -> per-control-point space-time
      halfspace at each point's OWN time t_{i,k}=control_point_times():
        g = R - a_{i,k,h}^T (P_{i,k} - c_h(t_{i,k})),  R = r_h + d_safe.
      This anchors every control point against the human's predicted position at
      the moment the drone reaches it ("avoid at the right moment", metric #1).
      It also carries dg/dt_{i,k} = +a_{i,k,h}^T vel_h (the moving-human absolute-
      time sensitivity that feeds Source 2 of dCost/dT in _alm_term).
    AABB (wall, all-hard ablation) -> per-control-point signed-dist surrogate:
        g = d_safe - signed_dist(P_ik),  dg/dP = -dd/dp,  dg/dt = 0.

    S4b TRUST WINDOW: under spacetime + a moving human, a control point whose
    time t_{i,k} > opt.tau_trust is EXCLUDED from the hard set (g set to a large-
    negative so w=0, trust=False); the stale-prediction tail is covered by the
    SOFT EGO field. The mask is FROZEN per outer iter via the call site.

    Returns (dict of flat arrays, P (M,6,3)).  `n` holds dg/dP's negative
    (i.e. the outward unit normal) so dL/dP = -w*n elsewhere stays uniform.
    `dgdt` holds dg/dt_{i,k} (nonzero only for trusted moving-sphere points);
    `trust` is the per-constraint boolean S4b mask."""
    M = tr.M
    H = len(hard_obs)
    P = tr.control_points()                      # (M,6,3)
    if normals is None:
        normals = _seg_normals(tr, hard_obs, P, opt=opt)
    spacetime = opt is not None and bool(opt.spacetime_hard)
    Nc = M * 6 * H                            # 约束总条数 = 段 × 6控制点 × 硬障碍
    # 把三维索引 (i,k,h) 压成一维 m = (i*6 + k)*H + h；下面三个数组是反查表
    # flat index m = (i*6 + k)*H + h
    seg = np.repeat(np.arange(M), 6 * H)
    kpt = np.tile(np.repeat(np.arange(6), H), M)
    hidx = np.tile(np.arange(H), M * 6)
    g = np.zeros(Nc); nrm = np.zeros((Nc, 3)); dist = np.zeros(Nc); Rarr = np.zeros(Nc)
    dgdt = np.zeros(Nc); trust = np.ones(Nc, dtype=bool)
    q_alpha = _q_conformal(opt)   # conformal 预测不确定度余量（常数，只对硬球体生效；墙不加）
    eps = _eps_track(opt)         # 执行管子半径（常数，对每个硬障碍都加：球 + 墙）
    t_rep = np.array([_seg_rep_time(tr, i) for i in range(M)])   # (M,)
    t_pt = tr.control_point_times()                             # (M,6)
    idx_all = (np.arange(M)[:, None] * 6 + np.arange(6)[None, :]) * H  # (M,6) base idx
    G_NEG = -1.0e9   # 极大负数：把不可信的点强制设为「永不激活」
    for h, obs in enumerate(hard_obs):
        d_safe = _hard_d_safe(obs, avoid_cfg)
        idx = idx_all + h                              # 这个障碍 h 对应的扁平下标 (M,6)
        if hasattr(obs, "centre0"):                   # 球体 -> 用支撑半空间
            R = float(obs.radius) + d_safe + q_alpha + eps  # 球半径 + d_safe + conformal + 管子
            moving_st = spacetime and _is_moving(obs)
            if moving_st:
                # 会动的人：每个控制点用自己的时刻定位人（S4a，CA: predict(t)）
                c = obs.predict(t_pt)                              # (M,6,3)
            else:
                # Stage 3：所有控制点共用「段末」单一时刻（CA: predict(t_rep)）
                c = np.broadcast_to(
                    obs.predict(t_rep)[:, None, :], (M, 6, 3))
            a = normals[:, :, h, :]                    # (M,6,3) 冻结的法向
            diff = P - c                               # (M,6,3)
            proj = np.einsum("mkd,mkd->mk", a, diff)   # a^T(P-c)：控制点在法向上离人多远
            g_h = R - proj                             # (M,6) 还差多少才够安全（>0=没躲够）
            dist_h = np.linalg.norm(diff, axis=2) - float(obs.radius)
            g[idx] = g_h
            nrm[idx] = a
            dist[idx] = dist_h
            Rarr[idx] = R
            if moving_st:
                # 会动的人才有：g 对墙上时刻的导数 dg/dt_{i,k} = +a·velocity_at(t_{i,k})
                # （正号，和 dL/dP 相反）。CA 下瞬时速度 vel+accel·t 逐控制点变化；
                # accel=0 时 velocity_at 退回常数 vel，逐字节回到 a·vel。
                if not np.any(obs.accel):
                    dgdt[idx] = np.einsum("mkd,d->mk", a, obs.vel)        # CV 回退
                else:
                    dgdt[idx] = np.einsum("mkd,mkd->mk", a, obs.velocity_at(t_pt))  # CA
                # S4b 信任掩码：优先用「冻结的」掩码，这样某点在内层迭代中跨过 tau_trust 时
                # 不会在硬/软之间来回切换（保证内层代价光滑，硬约束集合只在外层之间变）。
                # 只有在没传冻结掩码（T 固定的场景）时，才退而用当前 T 算的掩码。
                if trust_mask is not None:
                    tmask = np.asarray(trust_mask, dtype=bool)
                elif opt is not None:
                    tmask = t_pt <= float(opt.tau_trust)   # (M,6)
                else:
                    tmask = np.ones((M, 6), dtype=bool)
                trust[idx] = tmask
                g[idx] = np.where(tmask, g_h, G_NEG)       # 不可信的点 g 压成极大负数
                dgdt[idx] = np.where(tmask, dgdt[idx].reshape(M, 6), 0.0)
        else:                                          # 墙(AABB) -> 逐控制点用带符号距离
            for i in range(M):
                for k in range(6):
                    m = (i * 6 + k) * H + h
                    d, ddp, _ = _signed_dist_and_grad(obs, P[i, k], float(t_rep[i]))
                    # 墙也加执行管子 eps（无人机自身位置误差，与障碍类型无关）；不加 conformal
                    g[m] = (d_safe + eps) - d; nrm[m] = ddp; dist[m] = d; Rarr[m] = d_safe + eps
    return {"g": g, "n": nrm, "seg": seg, "kpt": kpt, "hidx": hidx,
            "dist": dist, "R": Rarr, "dgdt": dgdt, "trust": trust}, P


def _alm_term(tr: MinjerkTraj, hard_obs, avoid_cfg, lam: np.ndarray, rho: float,
              normals=None, opt: OptParams = None, trust_mask=None):
    """ALM（增广拉格朗日 PHR）罚函数的代价及其解析梯度。
    返回 (代价, 对系数 c 的梯度 (6M,3), 对每段时间 T 的显式梯度 (M,), 约束记录 cons)。
    PHR 罚的形式：对每条约束 g，令 z = lambda + rho·g；z>0 时这条约束「激活」、罚它，否则不罚。
    lambda 是乘子、rho 是罚重，由外层循环 _alm_solve 慢慢调。
    (cost, dcost_dc (6M,3), dcost_dT_explicit (M,), cons) for the ALM penalty.

    `normals` are the FROZEN supporting-halfspace normals (per outer iter); if
    None they are recomputed from tr (used for the standalone check_grad gate at
    a fixed (lambda,rho)). cons is reused for warm-start / certificate.

    dCost/dc is UNCHANGED from Stage 3 (c_h depends only on T, not on the MINCO
    coeffs c). dCost/dT_explicit has TWO sources:
      Source 1 (curve geometry): G . dP/dT  (Stage 3, always).
      Source 2 (NEW, moving-human absolute time): w * (a.vel) chained through
        dt_{i,k}/dT_j -> same-segment fraction k/5 + cross-segment UNIT for all
        j<i (reverse-cumsum). Identically 0 for static humans (dgdt=0)."""
    M = tr.M
    n6 = tr.NC * M
    dcost_dc = np.zeros((n6, 3))
    dT_exp = np.zeros(M)
    if not hard_obs:
        return 0.0, dcost_dc, dT_exp, None
    cons, P = _alm_constraints(tr, hard_obs, avoid_cfg, normals=normals, opt=opt,
                               trust_mask=trust_mask)
    g = cons["g"]
    z = lam + rho * g                            # (Nc,)
    active = z > 0.0                             # z>0 的约束才被罚
    # PHR 罚的代价值：激活的约束算 (1/2rho)z^2，再统一减掉 lambda^2/(2rho)
    # PHR cost: (rho/2) z^2 - lambda^2/(2 rho) on active, else -lambda^2/(2 rho)
    cost = 0.0
    cost += float(np.sum((0.5 / rho) * (z[active] ** 2)))
    cost -= float(np.sum((lam ** 2) / (2.0 * rho)))
    # 罚对约束的导数 dL/dg = z（激活时，否则 0）；再链到控制点：dL/dP = z·dg/dP = z·(-n)
    w = np.where(active, z, 0.0)                 # (Nc,) — 不可信的行 w 已经是 0
    dLdP_seg = np.zeros((M, 6, 3))               # 按 (段,控制点) 累加梯度
    seg = cons["seg"]; kpt = cons["kpt"]; nrm = cons["n"]
    contrib = (-w)[:, None] * nrm                # (Nc,3) = z·(-n)
    np.add.at(dLdP_seg, (seg, kpt), contrib)     # 把每条约束的贡献加到对应控制点上
    # 接下来把「对控制点 P 的梯度」往回链到优化变量 (c, T)（MINCO 那套链式法则）：
    # (A) 隐式那条：dL/dc_i = D(T_i) · (C2B^T @ G_i)，C2B 是系数->Bernstein 控制点的转换
    T = tr.T
    Dvec = np.stack([np.ones(M), T, T**2, T**3, T**4, T**5], axis=1)  # (M,6) 时间幂缩放
    CtG = np.einsum("kj,mkd->mjd", C2B, dLdP_seg)            # (M,6,3) = C2B^T @ G
    dc = Dvec[:, :, None] * CtG                              # (M,6,3)
    dcost_dc += dc.reshape(n6, 3)
    # (B) 对 T 的显式梯度·来源1（曲线几何）：控制点位置 P 本身随 T 变（那个 T^j 缩放是个坑）
    dP_dT = tr.control_points_dT_explicit()                 # (M,6,3)
    dT_exp += np.einsum("mkd,mkd->m", dLdP_seg, dP_dT)
    # (B) 对 T 的显式梯度·来源2（新增，会动的人的绝对时刻链）：
    # dL/dt_{i,k} = Σ_h w·(a·vel) = w·dgdt；再通过 dt_{i,k}/dT_j 链到各段 T。静态的人 dgdt=0，这段恒为 0。
    dgdt = cons.get("dgdt")
    if dgdt is not None and np.any(dgdt != 0.0):
        S_flat = w * dgdt                                    # (Nc,) = 每条约束对其时刻的 dCost/dt
        S = np.zeros((M, 6))
        np.add.at(S, (seg, kpt), S_flat)                    # (M,6) 对障碍 h 求和后
        kfrac = np.arange(6) / float(MinjerkTraj.DEG)       # k/5：本段内第 k 个控制点的时间占比
        dT_abs = np.einsum("k,mk->m", kfrac, S)             # 本段内的局部分量
        seg_sum = S.sum(axis=1)                             # (M,)
        # 跨段：t_{i,k} 还等于前面所有段时长之和，所以 dt/dT_j=1 对所有 j<i -> 用反向累加加回去
        if M > 1:
            dT_abs[:-1] += np.cumsum(seg_sum[::-1])[::-1][1:]
        dT_exp += dT_abs
    return cost, dcost_dc, dT_exp, cons


def _minco_cost_grad(x: np.ndarray, start: np.ndarray, goal: np.ndarray, M: int,
                     obstacles, avoid_cfg, opt: OptParams, T_target: float,
                     v0=None, a0=None, vf=None, af=None,
                     hard_obs=None, alm_lambda=None, alm_rho=0.0, alm_normals=None,
                     alm_trust_mask=None):
    """【整个 MINCO 求解的核心】解析地算出总代价 f 和它对优化变量的梯度 grad。
    这就是喂给 scipy L-BFGS-B 的「目标函数 + 梯度」。x = [q 拉平, T]；
    grad 顺序和 x 一致：[dCost/dq 拉平, dCost/dT]。
    把所有代价项（软障碍 + 速度 + 加速度 + 时间锚 + 平滑 + 硬约束 ALM）的梯度都累加到
    同一组 (dcost_dc, dT_exp) 里，最后用 MINCO 的链式法则一次性转成对 (q, T) 的梯度。
    THE analytic value+gradient. x = [q.ravel(), T]; returns (f, grad) with
    grad ordered identically to x: [dCost/dq.ravel(), dCost/dT].

    `obstacles` here is the SOFT set (EGO field). `hard_obs`/`alm_lambda`/
    `alm_rho`/`alm_normals` activate the augmented-Lagrangian penalty for the
    HARD set (with FROZEN supporting-halfspace normals). `alm_trust_mask` is the
    FROZEN (M,6) S4b hard/soft mask (frozen alongside the normals per outer
    iter so the inner cost stays smooth as T moves)."""
    # 从优化变量拆出内部路点 q 和各段时间 T，重建 MINCO 轨迹（端点结构性钉死，v0/a0 等是起终边界条件）
    nq = 3 * (M - 1)
    q = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    T = x[nq:].astype(np.float64)
    tr = MinjerkTraj.from_endpoints(start, goal, q, T, v0=v0, a0=a0, vf=vf, af=af)

    # 把各项的梯度都累加进同一对 (对系数 c 的梯度, 对段时间 T 的显式梯度)，最后统一回链
    n = tr.NC * M
    dcost_dc = np.zeros((n, 3))
    dT_exp = np.zeros(M)

    # 软障碍（EGO 场）
    c_obs, gc, gT = _minco_obstacle_term(tr, obstacles, avoid_cfg, opt)
    dcost_dc += opt.w_obs * gc
    dT_exp += opt.w_obs * gT
    f = opt.w_obs * c_obs

    # 超速罚
    c_vel, gc, gT = _minco_dynamic_term(tr, 1, opt.vmax, opt.w_vel, opt)
    dcost_dc += gc; dT_exp += gT; f += c_vel

    # 超加速度罚
    c_acc, gc, gT = _minco_dynamic_term(tr, 2, opt.amax, opt.w_accel, opt)
    dcost_dc += gc; dT_exp += gT; f += c_acc

    # 时间锚：只跟总时长有关，纯显式-T 项
    Ttot = float(np.sum(T))
    rel = (Ttot - T_target) / max(T_target, 1e-9)
    f += opt.w_time * rel * rel
    dT_exp += 2.0 * opt.w_time * (Ttot - T_target) / (max(T_target, 1e-9) ** 2)

    # 硬约束（人）的 ALM 罚——再叠一项进同一组梯度里（只有传了 hard_obs 且 rho>0 才算）
    if hard_obs and alm_rho > 0.0:
        c_alm, gc, gT, _ = _alm_term(tr, hard_obs, avoid_cfg, alm_lambda, alm_rho,
                                     normals=alm_normals, opt=opt,
                                     trust_mask=alm_trust_mask)
        dcost_dc += gc; dT_exp += gT; f += c_alm

    # 关键一步：把「对系数 c 的梯度 + 对 T 的显式梯度」一次性转成「对 (q, T) 的梯度」
    # （这里面包含了 c=M(T)^{-1}b 的隐式依赖，由 MINCO 自己处理）
    gq, gT_total = tr.grad_from_dcost_dc(dcost_dc, dcost_dT_explicit=dT_exp)

    # 平滑项（最小急动度能量）用 MINCO 自带的、已验证过的 energy_grad（它自己处理 c 和 T 的依赖）
    J, gq_e, gT_e = tr.energy_grad()
    f += opt.w_smooth * J
    gq = gq + opt.w_smooth * gq_e
    gT_total = gT_total + opt.w_smooth * gT_e

    grad = np.concatenate([gq.ravel(), gT_total]) if M > 1 else gT_total.copy()
    return float(f), grad


# ---- MINCO seed + per-seed solve ------------------------------------------

def _minco_seed(seed_path: np.ndarray, opt: OptParams):
    """从 A* 折线初始化 MINCO 优化变量：按弧长重采样成 M+1 个路点（段数 M 受 M_cap 限制），
    每段初始时间 T0 = 段长 / vmax（不低于 dt_min 两倍）。
    返回 (起点, 终点, 内部路点 q0 (M-1,3), 各段时间 T0 (M,))。
    Init from an A* polyline: arc-length resample to M+1 pts, T0=len/vmax.

    Returns (start, goal, q0 (M-1,3), T0 (M,))."""
    seed_path = np.asarray(seed_path, dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(seed_path, axis=0), axis=1)
    path_len = float(seg_lens.sum())
    M = int(np.clip(round(len(seed_path) / 2.0), 2, opt.M_cap))   # 段数，夹在 [2, M_cap]
    wp = _resample_polyline(seed_path, M + 1)
    wp[0] = seed_path[0]; wp[-1] = seed_path[-1]   # 强制对齐精确的起终点 / pin exact endpoints
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    T0 = np.maximum(seg / max(opt.vmax, 1e-6), opt.dt_min * 2.0)
    start = wp[0].copy(); goal = wp[-1].copy()
    q0 = wp[1:-1].copy()
    return start, goal, q0, T0


# ---- Stage 3 feasibility = continuous-time convex-hull / dense clearance ---

def hard_clearance(tr: MinjerkTraj, hard_obs, avoid_cfg, K_eval: int = 2000):
    """沿轨迹密集采样（默认 2000 点），算对【硬】障碍（人）的最小间隙和最严重侵入量。
    只看硬障碍，墙/软的排除在外。这是「安全」指标的真实度量，比凸包证书更直接。
    返回 (最小间隙, 最大侵入量)。最大侵入量 <= 0 表示每个采样点都离每个硬障碍够远（安全）。
    Densely-sampled continuous-time min HARD-human clearance (signed_dist) and
    the worst per-class breach (d_safe - signed_dist).  Walls/soft excluded.

    Returns (min_clearance, max_breach). max_breach <= 0 means every hard
    obstacle is cleared by its d_safe at every one of the K_eval samples."""
    if not hard_obs:
        return float("inf"), float("-inf")
    ts = np.linspace(tr.t_start, tr.t_end, K_eval)
    pts = tr.eval(ts)
    min_clr = float("inf"); max_breach = float("-inf")
    for obs in hard_obs:
        d_safe = _hard_d_safe(obs, avoid_cfg)
        for j in range(K_eval):
            d = obs.signed_dist(pts[j], float(ts[j]))
            if d < min_clr:
                min_clr = d
            br = d_safe - d
            if br > max_breach:
                max_breach = br
    return float(min_clr), float(max_breach)


def _certificate_margin(cons) -> float:
    """凸包「安全证书」的裕度 = 所有控制点中 (带符号距离 - d_safe) 的最小值 = -max g。
    >= 0 就表示整个控制多边形都安全，因此（凸包性质）整条曲线连续时间都安全。
    这是静态/关 space-time 时用的紧裕度。
    Analytic convex-hull certificate margin = min over ctrl-pts of
    (signed_dist - d_safe) = -max_m g_m.  >= 0 => control polygon clears."""
    if cons is None:
        return float("inf")
    return float(-np.max(cons["g"]))


def _certificate_margin_spacetime(tr: MinjerkTraj, hard_obs, avoid_cfg,
                                  opt: OptParams, trust_mask=None) -> float:
    """会动的人的「连续时间」安全证书（S4a/S4b）。保证一整段时间都安全，不只是采样点。
    用的是「相对轨迹」支撑半空间（**不是**早期那种各向同性球膨胀 R=r+d_safe+||v||·T_i——
    那个对快速人严重过保守、会假报 cert<0，已被取代）。

    做法：把人在该段的**完整 CA 运动** c(s)=c0+v_eff·s+½·accel·s²（s∈[0,T_i]，
    v_eff=velocity_at(t0)）升次成它的五次 Bernstein 控制点 c_k，与轨迹控制点 P_{i,k} 做差得
    相对控制点 D_{i,k}=P_{i,k}-c_k，再对 D 架一道（每段冻结的）支撑半空间。
    ★关键：c(s) 是 degree-2 多项式，升到 degree-5 Bernstein 是**精确**的——所以加速度项 ½a·s²
    被 c_k 一字不差地吸收（见代码里的 quad 项 = ½a·s² 的精确 Bernstein 系数），R **不需要**任何
    速度或加速度膨胀。只在「可信」控制点上检查（信任窗口内）；静态/关 space-time 用 _certificate_margin。
    CONTINUOUS-TIME clearance certificate for MOVING humans (S4a/S4b).

    Uses a RELATIVE-TRAJECTORY supporting halfspace (NOT the early isotropic ball
    inflation R = r + d_safe + ||v||*T_i, which was over-conservative for fast humans
    and has been replaced). Per segment i, degree-elevate the human's FULL CA motion
        c(s) = c0 + v_eff*s + 0.5*accel*s^2,   s in [0, T_i],  v_eff = velocity_at(t0)
    to its exact degree-5 Bernstein control points c_k, subtract from the trajectory
    control points (D_{i,k} = P_{i,k} - c_k), freeze a normal a_i = unit(mean_k D_{i,k}),
    and certify:
        R_i = r_h + d_safe + Q_alpha        (Q_alpha = conformal margin, see _q_conformal)
        margin_i = min over trusted k of (a_i^T D_{i,k} - R_i)
    SOUNDNESS: both p(s) and c(s) are degree-5 Bernstein, so p(s)-c(s) = sum_k B_k(s) D_{i,k}
    is a convex combination; a^T D_k >= R for all k => a^T(p(s)-c(s)) >= R for ALL s in the
    segment => ||p(s)-c(s)|| >= R (continuous-time, not just the 6 nodes).
    ACCELERATION is handled EXACTLY, not by inflation: c(s) is degree 2, and degree
    elevation to degree 5 is exact, so the 0.5*accel*s^2 term lives in c_k verbatim
    (the `quad` term below). R therefore needs NO ||v||*T or 0.5*||a||*T^2 padding — the
    only residual assumption is that the human follows CA up to the conformal error Q_alpha
    (unmodelled jerk etc.), which is exactly what Q_alpha absorbs. Asserted only over
    TRUSTED control points (the trust window). Static / spacetime-off use _certificate_margin.

    Returns the minimum margin over all (segment, trusted-k, moving-human)."""
    M = tr.M
    P = tr.control_points()                                  # (M,6,3)
    centroid = P.mean(axis=1)                                # (M,3)
    t0 = tr._cum[:M]                                         # (M,) segment-start times
    if trust_mask is None:
        trust_mask = _trust_mask(tr, opt)                   # (M,6)
    trust_mask = np.asarray(trust_mask, dtype=bool)
    margin = float("inf")
    for obs in hard_obs:
        if not (hasattr(obs, "centre0") and bool(opt.spacetime_hard)
                and _is_moving(obs)):
            continue
        d_safe = _hard_d_safe(obs, avoid_cfg)
        T = tr.T                                            # (M,) 各段时长
        c0 = obs.predict(t0)                                # (M,3) 人在各段起始时刻位置
        # 相对轨迹证书（取代旧的各向同性球膨胀）：旧法用 c0 + ||v||·T_i 球膨胀，人朝单一方向
        # 快速移动时各向同性放大严重过保守，高速人下会把安全轨迹假报成 cert<0（实测 -5.7 vs 真实
        # +4.0）。改法：把人的局部 CA 运动 c(s)=c0+v_eff·s+0.5·accel·s^2（s∈[0,T_i]，
        # v_eff=velocity_at(t0)=vel+accel·t0）升次成它的五次 Bernstein 控制点 c_k，与轨迹控制点
        # 做差得相对控制点 D_{i,k}=P_{i,k}-c_k，对 D 做一道支撑半空间。
        # Soundness：c(s)、P(s) 都是五次 Bernstein 凸组合 → P(s)-c(s)=Σ_k B_k(s)·D_{i,k}（凸组合），
        # a^T D_k>=R ∀k ⇒ a^T(P(s)-c(s))>=R ⇒ ||P(s)-c(s)||>=a^T(...)>=R，整段连续时间成立，
        # 且精确处理人移动方向、不各向同性放大 → 比球膨胀紧得多，高速人下不再假阴性。
        v_eff = obs.vel[None, :] + obs.accel[None, :] * t0[:, None]    # (M,3) velocity_at(t0)
        kf = np.arange(6, dtype=np.float64)                            # 0..5
        # 五次 Bernstein 控制点：单项 u^j 的 degree-5 控制点系数 = C(k,j)/C(5,j)，s=T·u
        #   线性项 v_eff·s -> v_eff·T·(k/5)；二次项 0.5·accel·s^2 -> 0.5·accel·T^2·k(k-1)/10
        lin = v_eff[:, None, :] * ((kf / 5.0)[None, :, None] * T[:, None, None])
        quad = obs.accel[None, None, :] * ((T * T)[:, None, None]
                                           * (kf * (kf - 1.0) / 40.0)[None, :, None])
        c_bern = c0[:, None, :] + lin + quad                # (M,6,3) c(s) 的五次 Bernstein 控制点
        D = P - c_bern                                      # (M,6,3) 相对控制点
        cD = D.mean(axis=1)                                 # (M,3) 相对质心 -> 冻结法向
        nrm = np.linalg.norm(cD, axis=1)
        ok = nrm > 1e-12
        a = np.where(ok[:, None], cD / np.where(ok[:, None], nrm[:, None], 1.0),
                     np.array([1.0, 0.0, 0.0]))
        proj = np.einsum("md,mkd->mk", a, D)                # (M,6) 相对控制点在法向上的投影
        R = float(obs.radius) + d_safe + _q_conformal(opt) + _eps_track(opt)  # +conformal +执行管子
        m_ik = proj - R                                     # (M,6) 每点裕度
        if np.any(trust_mask):
            margin = min(margin, float(np.min(m_ik[trust_mask])))
    return margin


def _build_certificates(tr: MinjerkTraj, hard_obs, avoid_cfg,
                        lam: np.ndarray, rho: float, normals=None,
                        opt: OptParams = None, trust_mask=None) -> list:
    """给「可解释性」用的可视化记录：每条 (人, 段, 控制点) 约束都生成一条可画出来的记录，
    包含控制点位置 P、对应时刻人的位置 c_h、有效半径 R、间隙、约束值 g、乘子 lambda、
    是否激活 active、是否在信任窗口内 trust、以及这条约束施加的「推力」force=lambda·法向。
    这就是 SANDO 那种「每一步都看得见为什么」的卖点：能直接画出来人在哪、约束在推哪。
    Per-(human,segment,ctrl-pt) drawable records for the interpretability API.

    For a MOVING human under spacetime the record anchors c_h at the control
    point's OWN time t_{i,k} (the avoid-at-the-right-moment instant) and exposes
    a `trust` bool = whether the point was a TRUSTED-hard ALM constraint (vs a
    stale-prediction point demoted to the SOFT field). Static humans keep the
    Stage-3 segment-END time (trust always True)."""
    if not hard_obs:
        return []
    cons, P = _alm_constraints(tr, hard_obs, avoid_cfg, normals=normals, opt=opt,
                               trust_mask=trust_mask)
    g = cons["g"]; z = lam + rho * g
    t_pt = tr.control_point_times()                          # (M,6)
    trust = cons["trust"]
    q_alpha = _q_conformal(opt)        # conformal 余量（硬球体 R 里已含；这里单独暴露给可视化）
    eps_tr = _eps_track(opt)           # 执行管子半径（每个硬障碍 R 里已含；单独暴露给可视化）
    recs = []
    for m in range(g.size):
        i = int(cons["seg"][m]); k = int(cons["kpt"][m]); h = int(cons["hidx"][m])
        obs = hard_obs[h]
        if hasattr(obs, "centre0"):
            moving_st = (opt is not None and bool(opt.spacetime_hard)
                         and _is_moving(obs))
            t_eval = float(t_pt[i, k]) if moving_st else _seg_rep_time(tr, i)
            c_h = obs.predict(t_eval)
        else:
            c_h = 0.5 * (obs.lo + obs.hi)
        lam_m = float(lam[m])
        recs.append({
            "human_idx": h,
            "class": obs.class_name,
            "seg": i,
            "ctrl_pt": k,
            "P": P[i, k].copy(),
            "c_h": np.asarray(c_h, dtype=np.float64).copy(),
            # R = 强制的有效半径（球半径 + d_safe + conformal 余量 Q_alpha）；
            # clearance = 原始几何间隙 dist - d_safe（不含 Q_alpha，所以加大 Q 时它不变）；
            # q_conformal = 这条约束里含的 conformal 余量，单独画出来 = 「为预测不确定度多推的量」。
            # R = enforced effective radius (radius + d_safe + Q_alpha + epsilon_track).
            # `clearance` is the RAW geometric gap dist - d_safe (margin-INDEPENDENT — it does
            # NOT drop as Q or eps grow). `q_conformal` (spheres only) and `epsilon_track`
            # (every hard obstacle) expose the two uncertainty slices of R so a viewer sees how
            # much extra the prediction margin vs the execution-tube margin pushed this point.
            "R": float(cons["R"][m]),
            "dist": float(cons["dist"][m]),
            "clearance": float(cons["dist"][m] - _hard_d_safe(obs, avoid_cfg)),
            "q_conformal": float(q_alpha) if hasattr(obs, "centre0") else 0.0,
            "epsilon_track": float(eps_tr),
            "g": float(g[m]),
            "lambda": lam_m,
            "rho": float(rho),
            "active": bool(z[m] > 0.0),
            "trust": bool(trust[m]),
            "force": (lam_m * cons["n"][m]).copy(),
        })
    return recs


def _alm_solve(start, goal, q0, T0, soft_obs, hard_obs, avoid_cfg, opt: OptParams,
               v0=None, a0=None):
    """增广拉格朗日的「外层循环」，里面套着 L-BFGS-B 的「内层无约束求解」。
    这是 MINCO 硬约束求解的总调度。每轮外层：
      1) 冻结当前迭代的支撑半空间法向 + 信任窗口掩码（这样内层里 g 对 P 线性、代价光滑）；
      2) 内层跑一遍 L-BFGS-B（在当前 lambda/rho/法向下做无约束优化）；
      3) 更新乘子 lambda <- max(0, lambda + rho·g)；约束没改善（卡住）就放大罚重 rho；
      4) 最大违反量 < 容差 就收敛退出；到外层上限还没好就停（判 STOP）。
    x 和 lambda 在外层之间热启动（接着上一轮继续）。
    返回一个候选字典，带 ALM 的各种遥测信息 + 连续时间安全证书。
    Augmented-Lagrangian outer loop wrapping the L-BFGS-B inner solve.

    Outer: lambda <- max(0, lambda + rho*g); grow rho on stalled violation;
    stop at max-violation < tol or outer cap. Warm-start x and lambda.

    Returns a candidate dict with the ALM telemetry + certificate."""
    M = T0.size
    nq = 3 * (M - 1)
    x = np.concatenate([q0.ravel(), T0]) if M > 1 else T0.copy()
    T_target = float(np.sum(T0))
    # q 无界，T 受 dt_min 下界
    bounds = [(None, None)] * nq + [(opt.dt_min, None)] * M

    # 先数一下硬约束有多少条，初始化乘子 lambda（全 0）和罚重 rho
    tr0 = MinjerkTraj.from_endpoints(start, goal, q0, T0, v0=v0, a0=a0)
    if hard_obs:
        cons0, _ = _alm_constraints(tr0, hard_obs, avoid_cfg)
        Nc = cons0["g"].size
    else:
        Nc = 0
    lam = np.zeros(Nc)
    rho = float(opt.alm_rho0)

    soft_args = (start, goal, M, list(soft_obs), avoid_cfg, opt, T_target, v0, a0)
    f0, _ = _minco_cost_grad(x, *soft_args)   # 记一下初始代价（只算软项，给 info 用）

    lambda_hist = []; rho_hist = []
    prev_viol = float("inf")
    outer_iters = 0
    total_inner_iters = 0
    result = None
    if not hard_obs:
        # 没有硬约束（纯软/纯平滑）-> 直接一次无约束求解搞定，不用外层
        # no hard constraints -> a single unconstrained solve (pure soft / smooth)
        result = scipy.optimize.minimize(
            _minco_cost_grad, x, args=soft_args, jac=True, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": opt.maxiter, "gtol": 1e-6})
        x = result.x
        total_inner_iters = int(result.nit)
    else:
        normals = None
        for outer in range(opt.alm_outer_iters):    # ALM 外层循环
            outer_iters = outer + 1
            # FREEZE the supporting-halfspace normals from the current iterate
            # (GCOPTER SFC discipline) so g is linear in P during the inner solve
            qf = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
            Tf = x[nq:]
            tr = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0)
            normals = _seg_normals(tr, hard_obs, opt=opt)
            # FREEZE the S4b trust mask from the same iterate (so a point crossing
            # tau_trust during the inner solve does NOT flip HARD<->SOFT -> the
            # inner cost stays smooth; the active hard set changes only here).
            tmask = _trust_mask(tr, opt)
            # 内层：在当前冻结的 (lambda, rho, 法向, 信任掩码) 下做一次无约束 L-BFGS-B。
            # 用闭包把 ALM 的那些关键字参数塞进去（scipy 只会按位置传 args）
            result = scipy.optimize.minimize(
                lambda xx, lam=lam, rho=rho, nrm=normals, tm=tmask: _minco_cost_grad(
                    xx, *soft_args, hard_obs=hard_obs, alm_lambda=lam, alm_rho=rho,
                    alm_normals=nrm, alm_trust_mask=tm),
                x, jac=True, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": opt.alm_inner_maxiter, "gtol": 1e-7})
            x = result.x
            total_inner_iters += int(result.nit)
            qf = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
            Tf = x[nq:]
            tr = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0)
            # 在「同一套冻结的法向+掩码」下重新算 g（保证和内层用的约束一致）
            cons, _ = _alm_constraints(tr, hard_obs, avoid_cfg, normals=normals,
                                       opt=opt, trust_mask=tmask)
            g = cons["g"]
            # 乘子更新 (PHR)：lambda <- max(0, lambda + rho·g)
            lam = np.maximum(0.0, lam + rho * g)
            max_viol = float(np.max(np.maximum(g, 0.0)))   # 当前最大违反量
            lambda_hist.append(float(np.max(lam)) if lam.size else 0.0)
            rho_hist.append(float(rho))
            if max_viol < opt.alm_viol_tol:   # 约束都满足了 -> 收敛
                break
            # 卡死提前退出：rho 已经顶到上限、且违反量几乎没改善 -> 再迭代也没用，停（表现为 STOP）
            if rho >= opt.alm_rho_max and max_viol > 0.98 * prev_viol:
                break
            # 违反量没缩小够（卡住了）才放大 rho
            if max_viol > 0.5 * prev_viol:
                rho = min(rho * opt.alm_rho_grow, opt.alm_rho_max)
            prev_viol = max_viol

    qf = x[:nq].reshape(M - 1, 3) if M > 1 else np.zeros((0, 3))
    Tf = x[nq:]
    tr = MinjerkTraj.from_endpoints(start, goal, qf, Tf, v0=v0, a0=a0)

    cons_final = None; max_viol_final = 0.0; cert_margin = float("inf")
    final_normals = None
    final_trust = None
    if hard_obs:
        # 证书裕度在「最终曲线」上用「重新算的」法向来检（对最终轨迹做紧的半空间检验）
        final_normals = _seg_normals(tr, hard_obs, opt=opt)
        final_trust = _trust_mask(tr, opt)
        cons_final, _ = _alm_constraints(tr, hard_obs, avoid_cfg,
                                         normals=final_normals, opt=opt,
                                         trust_mask=final_trust)
        max_viol_final = float(np.max(np.maximum(cons_final["g"], 0.0)))
        # 有会动的人 -> 用保守的「膨胀半径单半空间」连续时间证书；
        # 静态/关 space-time -> 用 Stage 3 的紧逐点裕度。墙(AABB)也走紧裕度那条。
        has_moving = any(opt.spacetime_hard and _is_moving(o) for o in hard_obs)
        if has_moving:
            cert_margin = _certificate_margin_spacetime(
                tr, hard_obs, avoid_cfg, opt, trust_mask=final_trust)
        else:
            cert_margin = _certificate_margin(cons_final)

    return {
        "tr": tr, "x": x, "lam": lam, "rho": rho,
        "outer_iters": outer_iters,
        "total_inner_iters": total_inner_iters,
        "max_violation": max_viol_final,
        "certificate_margin": cert_margin,
        "final_normals": final_normals,
        "final_trust": final_trust,
        "lambda_history": lambda_hist, "rho_history": rho_hist,
        "result": result, "f0": float(f0), "T_target": T_target,
        "T0": T0,
    }


def _optimise_one_minco(seed_path: np.ndarray, obstacles, avoid_cfg,
                        opt: OptParams, v0=None, a0=None) -> dict:
    """从一条种子折线出发，跑一次「硬约束 ALM + 软场」的 MINCO 求解，返回这个候选的记录字典。
    按类别分派：障碍用 resolve_mode 分成软（EGO 场）和硬（ALM 凸包约束）两堆。
    One ALM(hard) + soft-field solve from a seed polyline. Candidate dict.

    Class dispatch: obstacles split by resolve_mode(obs, avoid_cfg, override)
    into soft (EGO field) and hard (ALM convex-hull constraint)."""
    start, goal, q0, T0 = _minco_seed(seed_path, opt)
    M = T0.size
    nq = 3 * (M - 1)
    soft_obs, hard_obs = _partition_obstacles(obstacles, avoid_cfg, opt.avoid_override)

    sol = _alm_solve(start, goal, q0, T0, soft_obs, hard_obs, avoid_cfg, opt,
                     v0=v0, a0=a0)
    tr = sol["tr"]
    result = sol["result"]
    Tf = sol["x"][nq:]

    # 可行性 = 连续时间间隙 + 运动学限制，不看 scipy 的收敛标志（两者独立）
    feas = dict(check_feasibility(tr, obstacles, avoid_cfg, opt))
    K_dense = max(int(opt.K), 200)
    # 安全指标/STOP 判定一律用「天然的」硬集合（override=None），这样消融对照组才可比：
    # 哪怕这次跑的是「全软」消融，人撞了照样要报出来——人就是人，跟消融开关无关
    _, safety_obs = _partition_obstacles(obstacles, avoid_cfg, None)
    min_clr, max_breach = hard_clearance(tr, safety_obs, avoid_cfg, K_eval=K_dense)

    # 「间隙是否合格」只用硬障碍来判。软障碍（墙）本来就是设计成可以蹭着走的——这正是
    # per-class 的核心意义——所以它们的接近度只记录进 feas 指标，绝不能因此把轨迹判为无效。
    # check_feasibility 是一视同仁量所有障碍的，软墙蹭一下就可能触发它的 clearance_violation；
    # 这里重新只按硬集合下结论，若硬集合没问题就退回到（与类别无关的）运动学检查。
    hard_violation = bool(safety_obs) and (max_breach > opt.clearance_tol)
    if hard_violation:
        feas["trajectory_valid"] = False
        feas["failure_reason"] = "clearance_violation"
    elif feas["failure_reason"] == "clearance_violation":
        if feas["max_v"] > opt.vmax * (1.0 + opt.vel_tol):
            feas["failure_reason"] = "velocity_violation"
        elif feas["max_a"] > opt.amax * (1.0 + opt.accel_tol):
            feas["failure_reason"] = "acceleration_violation"
        else:
            feas["failure_reason"] = None
        feas["trajectory_valid"] = feas["failure_reason"] is None

    certs = _build_certificates(tr, hard_obs, avoid_cfg, sol["lam"], sol["rho"],
                                normals=sol.get("final_normals"), opt=opt,
                                trust_mask=sol.get("final_trust"))
    avoid_modes = {}
    for o in obstacles:
        avoid_modes[o.class_name] = resolve_mode(o, avoid_cfg, opt.avoid_override)

    return {
        "spline": tr,
        "init_cost": float(sol["f0"]),
        "final_cost": float(result.fun) if result is not None else float(sol["f0"]),
        "iter": int(sol["total_inner_iters"]),
        "optimizer_success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "no-op",
        "dt0": float(T0.min()),
        "dt_final": float(Tf.min()),
        "T_target": float(sol["T_target"]),
        "T_final": float(np.sum(Tf)),
        "feasibility": feas,
        "M": int(M),
        "hard_violation": hard_violation,
        "hard_max_breach": float(max_breach),
        "hard_min_clearance": float(min_clr),
        "n_hard": len(hard_obs),
        "n_soft": len(soft_obs),
        "alm": {
            "outer_iters": sol["outer_iters"],
            "rho_final": float(sol["rho"]),
            "max_violation": float(sol["max_violation"]),
            "lambda_max": float(np.max(sol["lam"])) if sol["lam"].size else 0.0,
            "lambda_history": sol["lambda_history"],
            "rho_history": sol["rho_history"],
        },
        "hard_certificates": certs,
        "continuous_min_clearance": float(min_clr),
        "certificate_margin": float(sol["certificate_margin"]),
        "avoid_modes": avoid_modes,
    }


def plan_minco(astar_path: np.ndarray,
               obstacles: Iterable,
               avoid_cfg: Dict[str, AvoidParams],
               *,
               opt_params: OptParams = None,
               detour_cfg: DetourConfig = None,
               v0=None, a0=None) -> Tuple[MinjerkTraj, dict]:
    """【新版主线对外接口】MINCO 解析梯度求解。签名和返回的 info 键尽量和旧版 plan() 对齐。
    优化变量 = (内部路点 q, 各段时间 T)；起终点是「结构性钉死」的（不是用罚函数压的）。
    绕行多起点 + 可行优先挑选这两步直接复用 B 样条那套（它们只跟 A* 折线打交道，跟轨迹类型无关）。

    v0/a0 = 起点的速度/加速度边界（默认从静止开始）。做 receding-horizon（滚动重规划）时，
    把当前正在执行的状态传进来，这样每次重规划不会都从静止重新起步、能接上。
    """
    opt = opt_params or OptParams()
    detour_cfg = detour_cfg or DetourConfig()
    astar_path = np.asarray(astar_path, dtype=np.float64)
    if astar_path.ndim != 2 or astar_path.shape[0] < 2:
        raise ValueError(f"astar_path must be (M>=2, d), got {astar_path.shape}")

    obstacles_list = list(obstacles)
    seeds = _generate_detour_seeds(astar_path, obstacles_list, avoid_cfg, opt, detour_cfg)

    candidates: List[dict] = []
    for name, seed_path in seeds:
        try:
            rec = _optimise_one_minco(seed_path, obstacles_list, avoid_cfg, opt,
                                      v0=v0, a0=a0)
        except Exception:
            continue
        rec["seed_name"] = name
        candidates.append(rec)
    if not candidates:
        raise RuntimeError("all seeds failed to optimise")

    best = _select_best_minco(candidates)
    feas = best["feasibility"]
    info = {
        "trajectory_valid": feas["trajectory_valid"],
        "optimizer_success": best["optimizer_success"],
        "failure_reason": feas["failure_reason"],
        "feasibility": feas,
        "seed_used": best["seed_name"],
        "n_seeds_tried": len(candidates),
        "init_cost": best["init_cost"],
        "final_cost": best["final_cost"],
        "iter": best["iter"],
        "dt0": best["dt0"],
        "dt_final": best["dt_final"],
        "T_target": best["T_target"],
        "T_final": best["T_final"],
        "M": best["M"],
        "message": best["message"],
        "converged": best["optimizer_success"],  # legacy alias
        # --- Stage 3 interpretability + clearance certificate ---
        "n_hard": best["n_hard"],
        "n_soft": best["n_soft"],
        "hard_violation": best["hard_violation"],
        "hard_max_breach": best["hard_max_breach"],
        "alm": best["alm"],
        "hard_certificates": best["hard_certificates"],
        "continuous_min_clearance": best["continuous_min_clearance"],
        "certificate_margin": best["certificate_margin"],
        "avoid_modes": best["avoid_modes"],
    }
    return best["spline"], info
