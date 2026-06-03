---
name: sando-py-esdf-sfc-decision
description: 墙避障表示决策:SFC 不用;ESDF 不是现在(局部墙现是解析盒非体素);sim2real 接体素后选 ESDF vs ESDF-free。2026-06-01 workflow 5路对抗 + 核代码。
metadata: 
  node_type: memory
  type: project
  originSessionId: 19193e4c-df5f-40d8-a717-7ec143270399
---

墙(静态环境)避障表示的决策(2026-06-01,19→5 路对抗 workflow + 亲核代码):

**认知纠正(重要)**:局部优化器躲的墙现在是 **DynTraj 解析盒** `AABBObstacle`(planner.py:1184),**不是 RGBD 体素**;体素(voxel_map cmap)只喂全局 heat-A*,两条独立数据链中间没桥。所以「给局部软墙喂体素 ESDF」现在根本没接口——盒子是 sim2real 占位债(见 [[sando-py-sim2real-fakes]])。我一度顺着「墙是 RGBD」预判要上 ESDF,被红队 + 核代码推翻。

**SFC(凸走廊):不用。** 但否定理由要换——**别用「整数不可微」**(那只否得掉 MIQP,否不掉走廊定好后凸可微的纯 SFC,审稿人会当场拆)。真理由:SFC 把墙变硬 → 砸掉 per-class 软场一极 → 退化成「软硬 = weight 大小」。paper 要正面回应「SFC 至今仍是静态墙 SOTA 主流(GCOPTER/FASTER/GCS),你凭啥不用」。

**ESDF:墙真接体素后用 ESDF(不是 ESDF-free)——2026-06-01 workflow `wf_9649c419` 定论(6 agent:web 调研 + 本机实测 + 红队,高置信)。** 现在别动(局部墙还是解析盒、AABB 盒外已是精确欧氏距离;墙若 demo 抖最小修法是 soft-min 解析光滑化,不是 ESDF)。墙真接 RGBD 体素后:
- **选 ESDF 不选 ESDF-free**:① ESDF 只换 `_signed_dist_and_grad_batch`(local_opt.py:740)里「d 从哪来」这一格,φ=(d_safe-d)^3 + MINCO adjoint 字节级不变 → 统一框架卖点保住;ESDF-free 把墙变成 (P-anchor)·v 时变平面代理,墙连真距离都没了、砸统一一半、正中上轮红队攻击。② 可解释性:ESDF 的 d/∇d 是真距离场、和人侧 λ 讲同一「距离/裕度」故事 + 白送距离场可视化(缓解「墙是占位盒」痛点);ESDF-free 的 anchor 是内部时变构造,审稿人追问稳定性,减分。
- **★ 实测修正了乐观数(完全真实)**:scipy `distance_transform_edt` 约 0.1µs/voxel 线性,150^3 实测 **394ms(带符号两次 edt 778ms)**,不是之前传的 82ms → **研究阶段必须硬锁体素 <= ~20万**(60^3 或 100x100x20,~20-40ms 可 20-30Hz);一过 20 万掉到 2-3Hz、动态反应性自爆。
- **分阶段**:研究 = scipy edt 每拍全量(零增量、和每拍重建地图零摩擦);上机 = 有 NVIDIA Orin→nvblox(0.3-8ms,带动态,torch 接口)/ 纯 CPU→FIESTA C++ 增量(~4ms,但要重写指针链表,搬运债比 ESDF-free 重)。
- **第二钉子**:三线性 ∇d 跨体素面 C0 不连续(实测跳变 0.1-0.2),软场 grad 门禁要补 `_field_smooth_config` 排「采样点距格面 < eps」(类比 alm 排 z~0),否则 check_grad FD 跨面偶发 rel-err ~1e-2 假性失败。
- **落地(未来债,内核一行不动)**:voxel_map 加 `occupancy_field()`(cmap→两次 edt→带符号场,reshape (dimZ,dimY,dimX) 照抄 voxel_map.py:502 先例)、obstacles 加 `VoxelFieldObstacle`(单点三线性 signed_dist)、local_opt:740 `_signed_dist_and_grad_batch` 加三线性第三支(插值梯度对 FD 已验 3e-10)、planner:1199 墙分支 AABBObstacle→VoxelFieldObstacle。
- **sim2real 洞察升级成 paper 论点**:不是「两套距离表示」,是**同一个 signed-distance oracle 的两个 backend**(解析 backend=人闭式 d/∇d、栅格 backend=墙插值 d/∇d),统一性在 φ + adjoint 层、不在 d 来源层 → 红队「第二套表示砸统一」反转成卖点物理依据。limitation 诚实写:解析精确 / 栅格量化近似,精度档位(精确 vs 近似)与安全档位(硬撞致命 vs 软可蹭)对齐 = 有意设计。

**sim2real 关键洞察**:真实感知栈天然两条流——人 = 检测/分割的「物体」(解析球/凸包 + 硬约束 + 时空预测)、墙 = RGBD 稠密「环境」(体素 → 场 + 软)。所以「墙吃体素、人吃解析」两套表示**有语义依据**(物体 vs 环境本来就两种东西),正好长在 per-class(人硬墙软)上,化解了「第二套距离表示砸统一框架」的红队攻击,反而是卖点的物理依据。

参考 [[sando-rgbd-plan]] 权威进度。
