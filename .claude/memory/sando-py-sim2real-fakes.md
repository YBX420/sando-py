---
name: sando-py-sim2real-fakes
description: sando-py 是 sim2real 主线;当前 demo 三大占位债(class写死if / 匀速预测 / EKF降维丢信息)按优先级要换成真感知。塔菲大人 2026-06-01 明确要记。
metadata: 
  node_type: memory
  type: project
  originSessionId: 19193e4c-df5f-40d8-a717-7ec143270399
---

sando-py 是 **sim2real 主线**(最终上真无人机 + 真 RGBD 感知,墙一定是任意占据体素、不可能是规整盒子)。规划内核(MINCO + per-class 人硬墙软)已验到机器精度、是真的;要还的债全在「感知 → 规划」之间那段喂数据的接口上。三大占位债(按该先修顺序,均已核代码,塔菲大人 2026-06-01 点名要记):

1. **class 源写死 if**(planner.py:730-731):靠 `id∈[200,300)` 判人/墙。**per-class 整个卖点(人硬墙软)的地基就是这一行 if**;类别判错,硬约束就套到错的物体上。审稿人一眼看穿「你的分类器是个 if」。**头号,必换真分类器 / RGBD 语义标签。**
2. **匀速直线预测**(obstacles.py:67,`predict(t)=centre0+vel*t`):是「在正确时刻躲人」(指标①)的物理前提,真人会变向拐弯。要升级成消费 tracker 多项式 / 带不确定度的预测管。
3. **EKF 接口降维丢信息**(planner.py:733 只取 `traj.velocity()` 一个瞬时速度塞进球):agent 说 obstacle_tracker.py 有完整 9D CA EKF(算了加速度 + 5 次多项式),接口处只喂瞬时速度 = 自己写了真东西又在接口扔了。(733 只取瞬时 vel 已亲核;tracker 到底多丰富待核。)边际成本最低、收益直接,建议和第 2 一起做。

依赖关系:**class 源是地基**(分错后人/墙两条流都接到错物体)。顺序 = class 源 → 人预测/EKF → 墙体素表示。墙体素表示的决策见 [[sando-py-esdf-sfc-decision]]。本线方法论见 [[feedback-rootcause-first]],权威进度 [[sando-rgbd-plan]]。
