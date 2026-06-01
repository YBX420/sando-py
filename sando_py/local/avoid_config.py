"""Per-class obstacle-avoidance parameters.

Open-ended schema: a config is a dict[class_name -> AvoidParams]. Add new
classes by adding new entries — cost.py looks each obstacle's class_name up
in the config and applies its (mode, d_safe, weight) at evaluation time.

mode:
  "soft"  — gentle field, low weight (walls, ceilings)
  "hard"  — very stiff field via a huge weight (people, robots)
  Same EGO-Planner cubic penalty shape for both; the only difference is
  weight magnitude (so optimisation stays unconstrained / differentiable).

中文说明:
这个文件是"按障碍类别配避障参数"的小配置表。整个规划器里,不同类别的障碍
(墙、人……) 用不同力度去躲;这里就用一个字典把"类别名字" -> "一组参数"对应起来。
cost.py 在算代价时,会拿每个障碍的 class_name 来这张表里查,取出它的
(mode 模式 / d_safe 安全距离 / weight 权重) 三个数来用。

mode 两档:
  - "soft" 软:权重小,像一个温柔的"推力场",撞上去代价不大 (用于墙、天花板这类)
  - "hard" 硬:权重特别大,等于"绝对不能碰" (用于人、机器人这类会动、撞了很危险的)
注意:软和硬用的是同一个 EGO 三次方惩罚公式 (形状一样),唯一区别就是 weight 的大小。
这样做的好处:代价函数始终是光滑可导的,优化器 (L-BFGS) 不用处理真正的硬约束也能跑。
(真正的"硬约束 = 凸包 + ALM"那一套在别的文件里;这里只是用大权重近似硬。)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# 一条避障参数记录:一个障碍类别对应一组数。用 dataclass 只是为了少写样板代码。
@dataclass
class AvoidParams:
    name: str
    mode: str       # "soft" or "hard"  软/硬两档
    d_safe: float   # metres — penalty kicks in when point-to-obstacle distance < d_safe
                    # 安全距离 (米):点离障碍小于这个值才开始有惩罚,大于它代价为 0
    weight: float   # cost weight  代价权重;hard 类别就是把它调到很大来"近似硬约束"


def default_config() -> Dict[str, AvoidParams]:
    """Two-class baseline. Tune via overrides; add classes by adding entries.

    默认两类配置 (墙=软、人=硬)。想调参就在外面覆盖,想加新类别就往字典里加一项。
    注意人的 d_safe (0.8m) 比墙 (0.4m) 大、weight (1e4) 比墙 (1e1) 大三个数量级——
    就是"离人要离得更远、而且基本不许靠近"的意思。
    """
    return {
        "wall":  AvoidParams("wall",  mode="soft", d_safe=0.4, weight=1.0e1),
        "human": AvoidParams("human", mode="hard", d_safe=0.8, weight=1.0e4),
    }


def resolve_mode(obs, avoid_cfg: Dict[str, AvoidParams], override) -> str:
    """Resolve an obstacle's avoidance mode -> 'hard' | 'soft'.

    override: None -> per-class (mode from avoid_cfg[class]);
              'soft'/'hard' -> GLOBAL ablation force on every obstacle.
    FAIL-SAFE: an unknown/missing class, or a malformed mode, defaults to
    'hard' — the violated constraint could be a person, so the safe default
    is the stiff continuous-time constraint, never a silent soft trade-off.

    决定某个障碍到底按"软"还是"硬"来躲,返回 'hard' 或 'soft' 字符串。
    - override 是 None:正常按类别走,从配置表里查这个障碍的 mode。
    - override 是 'soft'/'hard':一刀切,强制所有障碍都用同一档 (做消融实验时用,
      比如想看"全软"或"全硬"是什么效果)。
    安全兜底:如果这个类别查不到、或者 mode 写错了,默认返回 'hard'。
    原因:万一这个不认识的障碍其实是个人,宁可当成硬约束严格躲开,也绝不能悄悄
    退成软的、用代价去换——那样可能真撞上人。
    """
    if override in ("soft", "hard"):
        return override
    params = avoid_cfg.get(obs.class_name)
    if params is None:
        return "hard"
    if params.mode not in ("soft", "hard"):
        return "hard"
    return params.mode
