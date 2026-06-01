# 中文说明:这是 ROS 2 / Python 包的安装脚本(ament_python 风格)。它告诉 colcon:
#   有哪些 Python 子包要安装、哪些资源文件(launch/*.py、config 里的 yaml 和 rviz)要拷到
#   share 目录,以及把哪些函数注册成可执行节点(下面的 console_scripts,就是 `ros2 run` /
#   launch 里 executable= 能用的那些名字)。改了节点入口或新增 launch/config,通常要回来动这里。
from setuptools import setup
from glob import glob
import os

package_name = "sando_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.hgp",
        f"{package_name}.local",
        f"{package_name}.nodes",
    ],
    # 中文:要随包安装到 share 目录的非代码文件——ament 索引登记、package.xml、
    #       所有 launch 脚本、以及 config 下的 yaml 配置和 rviz 布局。
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="boxuanye",
    maintainer_email="sy2023@ic.ac.uk",
    description="Python reproduction of SANDO (planner core + ROS I/O).",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    # 中文:把这些 "可执行名 = 模块:函数" 注册成命令行入口。这就是 launch 文件里
    #       executable="sando_node"/"fake_sim"/"perclass_obstacle_pub" 等能对应上的来源。
    entry_points={
        "console_scripts": [
            "sando_node = sando_py.nodes.sando_node:main",
            "convert_goal_to_cmd_vel = sando_py.nodes.convert_goal_to_cmd_vel:main",
            "convert_odom_to_state = sando_py.nodes.convert_odom_to_state:main",
            "convert_vicon_to_state = sando_py.nodes.convert_vicon_to_state:main",
            "odom_to_global_state = sando_py.nodes.odom_to_global_state:main",
            "obstacle_tracker_node = sando_py.nodes.obstacle_tracker:main",
            "fake_sim = sando_py.nodes.fake_sim:main",
            "perclass_obstacle_pub = sando_py.nodes.perclass_obstacle_pub:main",
            "auto_goal_pub = sando_py.nodes.auto_goal_pub:main",
        ],
    },
)
