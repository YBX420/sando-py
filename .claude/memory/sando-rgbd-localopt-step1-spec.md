---
name: sando-rgbd-localopt-step1-spec
description: local_opt 二代 Step 1(detour multi-start seed)的完整拍板规格。用户 2026-05-28 给出。
metadata: 
  node_type: memory
  type: project
  originSessionId: 95acb254-cbe2-4e07-9a70-0255d93c13fc
---

# Step 1: detour multi-start seed — 完整规格(2026-05-28 用户拍板)

**目标**(明确范围,不要 over-engineer):
不是做完整 topology-aware planning。是把 R.A failure case(直穿球心 seed)在 formulation 层面修掉:
1. 验证原 LSQ seed 直穿球心会失败
2. 4-direction detour seed 至少一个产生 collision-free 初值
3. LBFGS 从该 seed 出发收敛到 valid trajectory
4. `result.success` 和 `trajectory_valid` 分开记录

## 详细决定

### Q1 detour 方向数 = 4(±u, ±v)
- u 和 v 是两个互相垂直的方向,都垂直于 start-goal 主方向
- 加原 LSQ seed = 5 个候选
- 不做 6+,沿轴向偏移意义小

### Q2 多 obstacle: top-k 排序,不做笛卡尔积
- `top_k_detour_obstacles = 1`(默认)
- 允许 = 2,**不更大**
- 按 violation severity 排序找 top-k 各自生成 4 个 detour
- top-1: 1 + 4 = 5 seeds; top-2: 1 + 4 + 4 = 9 seeds
- `max_seeds = 9`,硬 cap
- 不要"平均所有 center"那种,几何意义弱

### Q3 detour 触发: 几何判定(不用 cost 阈值)
- `seed_hits_obstacle = min_clearance_along_seed < d_safe + trigger_margin`
- `trigger_margin = 0.10 m`(范围 0.05-0.15)
- signed clearance:sphere = `dist - r`,AABB = 外部正/内部负
- violation = `d_safe - clearance`,越大穿得越严重
- 算法:LSQ seed 采 K_eval 点 → 每 obstacle 取 min clearance → violation > 0 进 candidate list → top-k

### Q4 选择 best: feasible-first,失败 fallback 最不坏
- **feasible 定义**:
  - `min_clearance ≥ d_safe - clearance_tol`
  - `max_v ≤ vmax*(1 + vel_tol)`
  - `max_a ≤ amax*(1 + accel_tol)`
- 有 feasible 候选:挑 `total_cost` 最小
- 没 feasible 候选(全失败):按下面 fallback 排序,挑"最不坏":
  1. primary:`max_clearance_violation` 最小
  2. secondary:`velocity_violation` 最小
  3. tertiary:`total_cost` 最小
- **info 字段必须分开**:
  - `trajectory_valid` (bool):业务可行
  - `optimizer_success` (bool):scipy `result.success`
  - `failure_reason`:`"clearance_violation"` / `"velocity_violation"` / `"acceleration_violation"` / `None`

### Q5 顺序跑(不并行)
- 第一版用 `for seed in seeds: ...` Python loop
- 调试稳定后再换 multiprocessing/joblib
- seed 间天然独立,后续并行容易

### Q6 detour 推多远
- 公式:`detour_offset = obstacle.radius + d_safe + detour_margin`
- `detour_margin = 0.30 m`(范围 0.2-0.5)
- sphere:`target = center + direction * (r + d_safe + margin)`
- AABB:第一版用 bounding sphere 简化(`r_equiv = 0.5·max_extent`)
- **不要只推 d_safe**(刚擦边会被 smooth/vel 拉回)

### Q7 perturb 在 path 层,不在 ctrl 层
- **不要**直接改 ctrl 点
- 流程:
  1. 找离 obstacle 最近的 path index
  2. 在该附近插入 detour waypoint(可以多个:`start → pre → detour → post → goal`)
  3. modified path 喂 `UniformBSpline.fit_path()` 重新拟
  4. 得到 detour spline 的 ctrl,作为该 seed 的初值
- 几何直觉更稳;不同 degree/knot spacing 下不会跑偏

## 完整 config

```python
detour_cfg = {
    "enabled": True,
    "directions": 4,                 # ±u, ±v
    "trigger": "geometric_clearance",
    "trigger_margin": 0.10,          # m
    "detour_margin": 0.30,           # m  -- 推到 r+d_safe 之外多少
    "top_k_obstacles": 1,            # 默认 1, 可改 2
    "max_seeds": 5,                  # top_k=1 → 5; top_k=2 → 9
    "run_mode": "sequential",
    "selection": "feasible_first_min_cost",
}
```

## feasibility check 接口

```python
def check_feasibility(spline, obstacles, avoid_cfg, opt) -> dict:
    """Return:
       {
         "min_clearance": float,
         "max_clearance_violation": float,   # max(0, d_safe - min_clearance)
         "max_v": float,
         "max_a": float,
         "vel_violation": float,             # max(0, max_v - vmax)
         "accel_violation": float,
         "trajectory_valid": bool,
         "failure_reason": str | None,
       }
    """
```

参考:[[sando-rgbd-localopt-issues]] 整个 5 步路线;[[feedback-rootcause-first]] 方法论。
