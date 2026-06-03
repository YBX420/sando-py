# perclass_demo_cpp.launch.py — full C++ port demo:
#   sando_node (planner.hpp) + fake_sim (loop-closer) + perclass_obstacle_pub + auto_goal_pub.
# Mirrors sando_py/launch/perclass_demo.launch.py but every node is the C++ build.
# perclass_obstacle_pub emits `predicted_trajs`; we remap it to `trajs` (what sando_node subs),
# i.e. the trivial identity stand-in for the Python obstacle_tracker pass-through.
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='sando_cpp', executable='sando_node', name='sando_node', output='screen'),
        Node(package='sando_cpp', executable='fake_sim', name='fake_sim', output='screen',
             parameters=[{'start_pos': [0.0, 0.0, 2.0], 'start_yaw': 0.0}]),
        Node(package='sando_cpp', executable='perclass_obstacle_pub', name='perclass_obstacle_pub',
             output='screen', remappings=[('predicted_trajs', 'trajs')]),
        Node(package='sando_cpp', executable='auto_goal_pub', name='auto_goal_pub', output='screen'),
    ])
