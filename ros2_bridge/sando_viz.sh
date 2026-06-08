#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source /home/boxuan/code/sando_ws/install/setup.bash
export DISPLAY=:0
export GAZEBO_MODEL_DATABASE_URI=
export ROS_DOMAIN_ID=20
export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$HOME/code/sando_ws/install/sando/share/sando/models"
cd /home/boxuan/code/sando_ws
pkill -9 gzserver 2>/dev/null; pkill -9 -f fake_sim 2>/dev/null; pkill -9 -f depth_to_occupancy 2>/dev/null
pkill -9 -f sando_py_bridge 2>/dev/null; pkill -9 -f robot_state_publisher 2>/dev/null; sleep 2

python3 -c "import matplotlib" 2>/dev/null || { echo "installing matplotlib..."; echo 1358 | sudo -S apt-get install -y python3-matplotlib >/dev/null 2>&1; }

echo "=== bring up sim + drone + fake_sim + boxes + mapper + bridge ==="
gzserver --verbose /home/boxuan/code/sando_ws/minimal_state.world \
  -s libgazebo_ros_init.so -s libgazebo_ros_factory.so > /tmp/gz.log 2>&1 &
sleep 7
xacro /home/boxuan/code/sando_ws/src/sando/urdf/quadrotor.urdf.xacro namespace:=NX01 > /tmp/quad.urdf 2>/dev/null
ros2 run gazebo_ros spawn_entity.py -entity NX01 -file /tmp/quad.urdf -x 0 -y 0 -z 2.0 >/tmp/spawn.log 2>&1
python3 - <<'PY'
import yaml
u = open('/tmp/quad.urdf').read()
yaml.safe_dump({'/**': {'ros__parameters': {'frame_prefix': 'NX01/', 'use_sim_time': False, 'robot_description': u}}},
              open('/tmp/rsp.yaml','w'), default_style='|', allow_unicode=True)
PY
ros2 run robot_state_publisher robot_state_publisher --ros-args -r __ns:=/NX01 --params-file /tmp/rsp.yaml >/tmp/rsp.log 2>&1 &
ros2 run sando fake_sim --ros-args -r __ns:=/NX01 \
  -p start_pos:="[0.0,0.0,2.0]" -p start_yaw:=0.0 -p send_state_to_gazebo:=true \
  -p visual_level:=2 -p publish_odom:=true -p odom_topic:=visual_slam/odom \
  -p odom_frame_id:=map -p base_frame_id:=NX01/base_link >/tmp/fakesim.log 2>&1 &
sleep 5
cat > /tmp/box.sdf <<'SDF'
<?xml version="1.0"?>
<sdf version="1.6"><model name="b"><static>true</static><link name="l">
<collision name="c"><geometry><box><size>1 1 3</size></box></geometry></collision>
<visual name="v"><geometry><box><size>1 1 3</size></box></geometry></visual>
</link></model></sdf>
SDF
ros2 run gazebo_ros spawn_entity.py -entity obs1 -file /tmp/box.sdf -x 6 -y -0.6 -z 1.5 >/dev/null 2>&1
ros2 run gazebo_ros spawn_entity.py -entity obs2 -file /tmp/box.sdf -x 10 -y 0.8 -z 1.5 >/dev/null 2>&1
python3 /home/boxuan/code/sando_ws/depth_to_occupancy.py >/tmp/mapper.log 2>&1 &
python3 /home/boxuan/code/sando_ws/sando_py_bridge.py --ros-args -r __ns:=/NX01 -p v_max:=2.0 -p a_max:=6.0 >/tmp/bridge.log 2>&1 &
sleep 6

echo "=== send goal + RECORD 15s ==="
ros2 topic pub --once /NX01/term_goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 16.0, y: 0.0, z: 2.0}}}" >/dev/null 2>&1
python3 /home/boxuan/code/sando_ws/record_and_plot.py 15.0

echo "=== copy PNG to Windows ==="
cp -f /tmp/sando_flight.png /mnt/c/Users/29505/Downloads/sando_flight.png && echo "copied to C:\\Users\\29505\\Downloads\\sando_flight.png"
ls -la /tmp/sando_flight.png

pkill -9 gzserver 2>/dev/null; pkill -9 -f fake_sim 2>/dev/null; pkill -9 -f depth_to_occupancy 2>/dev/null
pkill -9 -f sando_py_bridge 2>/dev/null; pkill -9 -f robot_state_publisher 2>/dev/null
echo "=== DONE ==="
