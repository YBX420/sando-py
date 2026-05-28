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
        f"{package_name}.nodes",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="boxuanye",
    maintainer_email="sy2023@ic.ac.uk",
    description="Python reproduction of SANDO (planner core + ROS I/O).",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sando_node = sando_py.nodes.sando_node:main",
            "convert_goal_to_cmd_vel = sando_py.nodes.convert_goal_to_cmd_vel:main",
            "convert_odom_to_state = sando_py.nodes.convert_odom_to_state:main",
            "convert_vicon_to_state = sando_py.nodes.convert_vicon_to_state:main",
            "odom_to_global_state = sando_py.nodes.odom_to_global_state:main",
            "obstacle_tracker_node = sando_py.nodes.obstacle_tracker:main",
            "fake_sim = sando_py.nodes.fake_sim:main",
        ],
    },
)
