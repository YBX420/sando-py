---
name: sando-py-defense-map
description: 顶会防御地图(2026-06-01 两 workflow:算法novelty + roast 12 弱点):命脉=零 benchmark 对走廊对手 + 确定性证书假装预测准;三波行动(管嘴→做表→上机);含 per-point 高速假阴性真 bug。
metadata: 
  node_type: memory
  type: project
  originSessionId: 19193e4c-df5f-40d8-a717-7ec143270399
---

> ⚠️ **2026-06-11 已 pivot —— 本文是 pivot 前 planner 论文的防御地图,framing 已过时。** 安全层主论文的竞争地图 / 写作防雷见 `docs/safety-layer-spec.md` §7、open risks 见 §8。本文里的技术结论(per-point 高速假阴性 bug ✅ 已修、relative-trajectory Bernstein 凸包修法等)仍有效。

2026-06-01 两个深度 workflow(算法 novelty `ww8qldyt6` + roast 防御地图 `wqt10g6s8`)收敛结论。**总判:方法数学自洽(密采无泄漏、grad 1e-9~1e-13、三臂可分辨已验),但"自己证给自己看、没证给世界看"。要命的是 claim 吹太大 + 零 benchmark。顺序:管嘴 → 做表 → 上机。**

**两存亡伤(无法话术绕,必须补实验)**:
- **corridor-rival**:base 就是 SANDO(走廊+MIQP+真机),Safe-Interval 也走廊+95%+真机。你说"无走廊更好"零对比数字,且自己实测 per-point≈segment-single 反向削弱核心机制。
- **no-experiments**:零 baseline / 不实时(Python 188ms)/ 全 mock。
→ 必补合成 Monte-Carlo benchmark + 2 对手(EGO 软场 + SIPP 时空走廊穷人版,同一 MINCO 求解器换避障表示求公平)。这是整篇立足点。

**三波行动**:
1. **第一波(一下午,零代码,管嘴,免费消大半攻击)**:guarantee→conditional certificate;全程硬→committed-horizon 硬+窗外软(标注 tau_trust);连续时间凸包→借的已知工具(别自称贡献,删 minco.py:69 / local_opt.py:1266 自证注释);可解释性→安全审计副产品;learnable→future work(删 5-10x);改 avoid_config 自拆台"软硬只是 weight"错注释(它正好证人硬墙软是 ALM vs EGO 两套真机制)。
2. **第二波(投稿前必修,纯 Python 2-4 天)**:
   - ① **★证书高速人假阴性 BUG ✅ 已修 2026-06-01**:旧法整段各向同性球膨胀 R=r+d_safe+||v||T_i+0.5||a||T_i²(local_opt.py:1323),人朝单一方向快走时各向同性放大严重过保守 → 高速人 cert -5.7 但真实 +4.0(假阴性废掉证书)。**注意:roast agent + 我最初提的"逐控制点 s_k 紧膨胀"是错的(实测没用,min 由段末 k=5 主导、s_5=T_i 退化成整段)——先验证不盲信救了一刀**。真修法 = **relative-trajectory Bernstein supporting halfspace**:把人局部 CA 运动 c(s)=c0+v_eff·s+0.5a·s²(v_eff=velocity_at(t0))升次成五次 Bernstein 控制点 c_k,与轨迹控制点做差 D=P-c_k,对 D 做一道支撑半空间(无膨胀)。soundness:P(s)-c(s)=Σ B_k(s)D_k 凸组合。实测 fast 假阴性 -5.7→+2.1 回正且 sound;S4.5 证书更紧(worst false-safe -1.339→-0.127)仍 sound;全套回归绿(perclass_grad 17/17 等);常驻回归 test/stage3_minco_cert_fast.py 9/9。**顺手把确定性证书做紧了(回应"数学浅"那刀),但仍确定性、不替代 conformal 升级(②)。**
   - ② **R 加预测不确定度项(=算法 novelty 解药,见 [[sando-py-core-idea]])**:R += k_sigma·sigma_pred(T),只改 R 公式 2 行、grad dg/dP=-a 不变;但核心是证"逐点 coverage 抬成整段 coverage"的 conformal 连续时间定理,非 κσ padding。
   - ③ class 误分类 fail-safe(不确定即按人 hard)。
   - ④ stop-cost 三态记账(SUCCESS / STOP-DEADLOCK / COLLISION,别把 STOP 偷换成功)出三条曲线(成功率 vs 难度、绕路比 vs 软方法、STOP 率 vs 保守度)。
3. **第三波(deploy,后置)**:C++ 实时、真 RGBD/语义、真机、外部开源 EGO/MADER head-to-head。

**fast-replan 稀释证书(致命度 4)的防御**:区分"连续时间=段内空间 inter-sample 无泄漏"(真贡献,不被高频重规划稀释,S4.6 已验未膨胀漏 +1.21)vs"0.75s 长时窗预测"(被快重规划稀释=真伤,realtime 测试 predict=False 仍 PASS 是反例)。重定位卖点到前者;长时窗用 persistent-feasibility(急停可行性)论证 + 重规划率 sweep 实验锚到"算力受限/真机 5Hz"。

参考 [[sando-py-core-idea]](novelty 升级)、[[sando-py-sim2real-fakes]](占位债)、[[sando-rgbd-plan]](进度)。
