# DynTraj label-set ABI 设计稿（v1，2026-06-17）

> W1 任务（plan §3）。目标：让 C++ 核心的 DynTraj 携带**类别**信息，追平 Python 参考实现
> （planner.py:733-740 已经优先读真标签），并**预留 conformal 分类集合**（spec §3.2 的核心贡献：
> 分类集合门控约束硬度，human ∈ 集合 → 硬）。
> **铁律：向后兼容、不破 golden 19/19。** 改 C++ 后必重编 `cpp/capi/sando_capi.so`。

## 1. 现状（为什么要做）
- 下游唯一判据：`obst_class[i]=="wall"`→软，否则→硬（planner.hpp:938/1060）。
- C++ 的 class 来源 = `id>=200 → wall` 启发式（planner.hpp:496-499）。DynTraj **无 label 字段**
  （types.hpp:346）；`traj_create`（capi:122）+ bridge（sando_cpp_bridge.py:233）**不传类别**。
- ⇒ 走 C++/Isaac/bridge 路径时真标签丢失，永远用 id 启发式（spec §4 修正 2 点名）。
- Python 参考（planner.py:737-740）已对：优先 `traj.obst_class`，空才退 id 启发式，判不出→人/硬。

## 2. 设计原则
1. **集合优先,从第一天就传集合**：ABI 传 `label_set`(整型类码数组),不传单标签——这样 Phase 2
   上真分类器(conformal 集合 + latch)时**零 ABI 改动**。Phase 1 planner 逻辑只用「human∈集合?」。
2. **空集合 = 无分类器信息 → 退回 id 启发式**(legacy)。⇒ 所有现有 golden 测试不传集合、行为
   完全不变、**不触发重基线**。
3. **类码与 Mondrian 3 类对齐**(spec §2):`HUMAN=0, VEHICLE_LIKE=1, OTHER=2`。
4. **latch 规则(一旦 human 永远 human)不在 DynTraj 里**——它是 per-track 状态,由 Python/tracker
   侧维护;传进来的集合已反映 latch。DynTraj 只如实携带当前帧集合。

## 3. 改动清单（4 处，全部加法、不动既有签名）

### 3a. `types.hpp` DynTraj 加字段（~365 行附近）
```cpp
// conformal classification set (Mondrian class codes: 0=HUMAN,1=VEHICLE_LIKE,2=OTHER).
// EMPTY = no classifier info -> caller falls back to the legacy id heuristic.
std::vector<int> label_set;
bool human_in_set() const {                       // human (code 0) ∈ set?
  return std::find(label_set.begin(), label_set.end(), 0) != label_set.end();
}
```

### 3b. `sando_capi.cpp` 新增 setter（不改 traj_create 的 10 参签名）
```cpp
// set the conformal label-set on an existing DynTraj handle. codes: Mondrian class ids.
// n==0 clears the set (-> legacy id-heuristic fallback downstream). Call after traj_create.
SANDO_API void traj_set_label_set(void* h, const int* codes, int n) {
  try {
    auto* d = static_cast<DynTraj*>(h);
    d->label_set.assign(codes, codes + (n > 0 ? n : 0));
  } catch (...) {}
}
```

### 3c. `planner.hpp` class 派生（496-499 替换）
```cpp
// class source: ① conformal label-set (real classifier) -> human∈set?"human"(hard):"wall"(soft);
//               ② empty set -> LEGACY id heuristic (id>=200 -> wall, else human).
std::string cls;
if (!t.label_set.empty()) cls = t.human_in_set() ? "human" : "wall";
else                      cls = (tid >= 200) ? "wall" : "human";
oclass.push_back(cls);
```
> v1 保持 human/wall 二元(对齐 Python 参考与下游)。Phase 2 再把非 human 集合细分到 avoid_config
> 的 vehicle-like/other,并接 unknown→hard fail-safe(spec §5)。

### 3d. `sando_cpp_bridge.py` 编组（DynTraj + sig）
```python
# sig (顶部 _traj_create 附近):
_traj_set_label_set = _sig("traj_set_label_set", None, C.c_void_p, C.POINTER(C.c_int), C.c_int)

# DynTraj.__init__: self.label_set = []          # Mondrian 类码列表;空=退回 id 启发式
# DynTraj.compile_analytic() 末尾(traj_create 之后):
if self.label_set:
    arr = (C.c_int * len(self.label_set))(*[int(x) for x in self.label_set])
    _traj_set_label_set(self._handle, arr, len(self.label_set))
```

## 4. 不做（划界，避免 scope creep）
- 不接真分类器(Phase 2,spec 估 5-8 天)。
- 不细分 vehicle-like/other 的软硬(v1 二元;非 human 一律软)。
- 不动 per-时间步 tube(独立任务,spec §5)。
- latch / 首检帧 / Mondrian 分格在标定工具链里做,不在本 ABI。

## 5. 验收
1. `cd cpp && cmake --build build -j && (cd build && ctest)` → **19/19 全绿**(没传集合,行为不变)。
2. 重编 `g++ ... -o capi/sando_capi.so ...`。
3. 新增微测(python/test):建一个 id<200 的 traj、显式塞 `label_set=[1]`(vehicle-like,无 human)
   → 经 bridge 跑 plan,验证它被当**软**(wall)处理,而非 id 启发式的硬(human)→ 证明 ABI 通了。
4. 反向:`label_set=[0]`(human)→ 硬。空集合 → 退回 id 启发式。
