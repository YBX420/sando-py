# 移植指南：Ubuntu 22.04 笔记本（独立操作版）

> 2026-06-11 在 WSL2 Ubuntu **22.04.5**（= 笔记本同款系统）上逐项验证过：
> C++ golden 测试 **19/19**、`sando_capi.so` 重编+ctypes 冒烟 **5/5 replan 成功**、
> Python 算法层 `stage3_minco_perclass.py` **38/38**、
> `sando_viz.sh` **D435i 闭环全链路跑通**（0→16m 绕开两个障碍盒，PNG 渲染正常）。
> 所以照本文做完，笔记本上的预期结果与上面一致。

---

## 0. 系统总共两块东西

| 块 | 来源 | 作用 |
|---|---|---|
| **sando-py 仓库** | `https://github.com/YBX420/sando-py.git`，分支 **main** | 规划器本体（C++ 核心 `cpp/` + Python 层 `python/` + ROS 桥 `ros2_bridge/`） |
| **sando_ws 工作区** | `https://github.com/mit-acl/sando.git`，**tag v0.0.3** + submodules | 基线/仿真设施：Gazebo、fake_sim、D435 无人机模型、dynus_interfaces 消息、RViz 配置 |

不在 git 里、需要单独处理的只有：
- **Gurobi license**（`~/gurobi.lic`）：**只有跑原版 SANDO 基线才需要**；sando-py 用 LBFGSpp，**完全不需要 Gurobi license**。要跑基线就在新机器上重新 `grbgetkey`（license 锁机器，旧文件拷过去没用）。
- `~/.bashrc` 里的几行环境（见第 4 步，照抄即可）。

WSL 里 sando_ws 对上游唯一的本地改动：**删掉了 `deps/acl-mapping` submodule**（私有 GitLab 仓库，拉不下来也用不上）。下面第 2 步会重现这一改动。

---

## 1. 系统准备（全新 Ubuntu 22.04）

```bash
# ROS 2 Humble（desktop 含 RViz）
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt install -y ros-humble-desktop

# 构建工具 + 本项目实际用到的包
sudo apt install -y build-essential g++ cmake git python3-pip python3-colcon-common-extensions \
  ros-humble-xacro ros-humble-robot-state-publisher ros-humble-tf2-ros \
  python3-numpy python3-scipy python3-matplotlib python3-yaml \
  gazebo libgazebo-dev tmux tmuxp
```

> 注意：**不要**单独 `apt install ros-humble-gazebo-ros-pkgs`——sando 自带补丁版 fork，
> `setup.sh` 构建时用 `--allow-overriding` 覆盖；先装了也行（WSL 里就装了），但别在之后手动升级它。
> 22.04 自带 numpy 1.21.5 / scipy 1.8.0 / Python 3.10.12，全部验证兼容，不用 pip 升级。

---

## 2. 装 sando_ws 基线（一次性，最耗时的一步）

```bash
mkdir -p ~/code/sando_ws/src && cd ~/code/sando_ws/src
git clone https://github.com/mit-acl/sando.git
cd sando && git checkout v0.0.3

# 重现 WSL 的本地改动：去掉拉不下来的私有 submodule
git rm --cached deps/acl-mapping 2>/dev/null || true
git config -f .gitmodules --remove-section submodule.deps/acl-mapping 2>/dev/null || true

./setup.sh -j 8        # 自动:装系统依赖+Gurobi11.0.3、拉其余 submodules、建 decomp_ws/livox_ws/sando_ws
```

- `setup.sh` 幂等，中断了重跑即可。模板重的文件单个 `cc1plus` 可吃 3–4 GB 内存，内存小就 `-j 4`。
- 构建**不需要** Gurobi license（运行原版基线才需要）。
- 完成标志：`~/code/sando_ws/install/sando/share/sando/` 存在，
  `~/code/sando_ws/install/realsense_gazebo_plugin/lib/librealsense_gazebo_plugin.so` 存在。

---

## 3. 装 sando-py（很快）

```bash
cd ~/code
git clone https://github.com/YBX420/sando-py.git    # 分支 main 即可
cd sando-py

# 3.1 编 C++ 核心 .so（~20 秒）
cd cpp
g++ -O3 -shared -fPIC -std=c++17 -o capi/sando_capi.so capi/sando_capi.cpp \
    -Iinclude -Ithird_party/eigen -Ithird_party

# 3.2 C++ golden 测试（预期 19/19）
cmake -S . -B build && cmake --build build -j && (cd build && ctest)

# 3.3 Python 冒烟（预期最后一行 38/38 passed）
cd ../python && python3 test/stage3_minco_perclass.py
```

> **加载优先级（防"旧库盖新码"的坑）**：`python/sando_cpp_bridge.py` 在 Linux 上按
> `cpp/capi/sando_capi.so → cpp/build/sando_capi.so → python/sando_capi.so` 的顺序找库。
> 重编时永远输出到 `cpp/capi/sando_capi.so`（上面 3.1 的命令就是），不需要再拷贝。
> 仓库里的 `python/sando_capi.dll` 是 Windows 用的，Linux 下自动忽略。

---

## 4. ~/.bashrc 追加（照抄）

```bash
source /opt/ros/humble/setup.bash
source $HOME/code/decomp_ws/install/setup.bash
source $HOME/code/sando_ws/install/setup.bash
export ROS_DOMAIN_ID=20
# 下面三行只有跑原版 Gurobi 基线才需要
export GUROBI_HOME="/opt/gurobi1103/linux64"
export PATH="$PATH:$GUROBI_HOME/bin"
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
```

（`setup.sh` 可能已写入类似几行，重复 source 无害；`ROS_DOMAIN_ID=20` 保持和原环境一致即可。）

---

## 5. 跑 D435i 闭环 demo

脚本已改成**位置无关**：从脚本自身定位仓库，工作区默认 `~/code/sando_ws`
（不同位置用 `SANDO_WS=/path/to/ws bash ...` 覆盖）。**不需要**再把脚本拷到 sando_ws 根目录。

```bash
cd ~/code/sando-py

# 无头版:飞一趟 → 出 PNG(约 50 秒;预期 ~1500 odom 采样、PNG 在 /tmp 和 ~ 各一份)
bash ros2_bridge/sando_viz.sh

# 实时版:Gazebo + RViz 实时看(Ctrl-C 全部退出)
bash ros2_bridge/sando_live.sh
```

排障日志都在 `/tmp/`：`gz.log`（Gazebo）、`spawn.log`、`fakesim.log`、`mapper.log`（depth_to_occupancy）、`bridge.log`（规划器桥）。

进程清理（异常退出后）：
```bash
pkill -9 gzserver gzclient rviz2; pkill -9 -f fake_sim; pkill -9 -f depth_to_occupancy; pkill -9 -f sando_py_bridge; pkill -9 -f robot_state_publisher
```

---

## 6. ROS 拓扑（已核对，namespace 统一 /NX01）

```
Gazebo(D435 插件) ─ /NX01/d435/depth/color/points (PointCloud2, BEST_EFFORT)
        ↓
depth_to_occupancy.py ─ 发 /NX01/occupancy_grid + /NX01/unknown_grid(暂为空) (PointCloud2, BEST_EFFORT)
        ↓                                  TF: map ← 相机光学帧(由 robot_state_publisher + fake_sim 提供)
sando_py_bridge.py ── 订 occupancy_grid(BEST_EFFORT) + term_goal(PoseStamped, RELIABLE) + state(RELIABLE)
        │             10Hz replan / 50Hz 控制流;ctypes → cpp/capi/sando_capi.so
        └─ 发 /NX01/goal (dynus_interfaces/Goal, RELIABLE)
        ↓
fake_sim(sando 包) ── 订 goal → 积分运动 → 发 /NX01/state + 推位姿给 Gazebo(send_state_to_gazebo)
        └──────────────── state 闭环回 bridge ────────────────┘
```

QoS 设计核对结论：传感器流 BEST_EFFORT（订阅端 BEST_EFFORT 兼容任何发布端）、控制/状态流 RELIABLE 两端一致，**无不匹配**。
无人机模型用的是 sando 自带 `urdf/quadrotor.urdf.xacro`（内含 D435 + realsense_gazebo_plugin）；
`ros2_bridge/urdf/drone.urdf.xacro` 是实验备份，当前没有脚本引用它。

---

## 7. 已知坑速查

| 坑 | 处理 |
|---|---|
| 改了 `cpp/` 头文件后 Python 跑的还是旧行为 | 必须重编 `cpp/capi/sando_capi.so`（3.1 的命令）；它是第一加载优先级 |
| `ros2 topic list` 看不到话题 | 检查 `ROS_DOMAIN_ID`（本环境统一 20）；不同终端要一致 |
| gzserver 起不来/黑屏 | 看 `/tmp/gz.log`；world 里已带 `gazebo_ros_state`，且 `GAZEBO_MODEL_DATABASE_URI=` 已置空（否则首启会卡在联网拉模型） |
| RViz/Gazebo GUI 在真机上 | 真机有显示器不需要任何 DISPLAY 设置；脚本里 `DISPLAY=${DISPLAY:-:0}` 只是 WSLg 兜底 |
| `unknown_grid` 是空的 | 设计如此（M3 视锥 free-carve 还没做），不是 bug |
| 原版基线 `run_sim.py` 跑不了 | 那是 Gurobi 路线，需要 license（`grbgetkey` 重新取）；sando-py demo 不受影响 |
| colcon 工作区报 overriding 错 | sando 完整构建需要 `--allow-overriding gazebo_dev gazebo_msgs gazebo_ros gazebo_ros_pkgs gazebo_plugins`（setup.sh 自带） |

---

## 8. 备选路线：直接搬 WSL（不推荐，但最快）

如果明天来不及全新构建，可以把 WSL 整个导出再导入笔记本的 WSL/或 rsync 到真机：

```powershell
# Windows 上导出(得到 ~20GB 文件)
wsl --export Ubuntu-22.04 D:\wsl-u2204.tar
```
```bash
# 真机上(用户名必须也是 boxuan,否则 install/ 里的绝对路径全失效,必须重新 colcon build)
sudo tar -xpf wsl-u2204.tar -C / --exclude=etc/fstab --exclude=etc/resolv.conf   # 不建议整根覆盖,谨慎
# 更稳妥:只搬 ~/code,然后 cd ~/code/sando_ws && rm -rf build install log && 重跑 setup.sh 的 colcon 步骤
```

要点：**colcon 的 `install/` 与 `build/` 充满绝对路径**，换用户名/换路径必须删掉重建；
源码（git 仓库）怎么搬都行。所以推荐还是第 2、3 步的干净安装。
