#!/usr/bin/env bash
# Live RViz view of the D435i closed loop. Run in an interactive terminal
# (WSLg or a real Ubuntu desktop shows the GUI):
#   bash <repo>/ros2_bridge/sando_live.sh
# Ctrl-C in this terminal stops everything.
#
# Paths are derived from this script's location; the sando_ws workspace
# defaults to ~/code/sando_ws (override: SANDO_WS=/path/to/ws bash sando_live.sh).
set -m
BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDO_WS="${SANDO_WS:-$HOME/code/sando_ws}"
source /opt/ros/humble/setup.bash
source "$SANDO_WS/install/setup.bash"
export DISPLAY=${DISPLAY:-:0}
export GAZEBO_MODEL_DATABASE_URI=
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-20}
export GAZEBO_MODEL_PATH="$GAZEBO_MODEL_PATH:$SANDO_WS/install/sando/share/sando/models"
cd "$SANDO_WS"

cleanup() { pkill -9 gzserver gzclient rviz2 2>/dev/null; pkill -9 -f fake_sim 2>/dev/null; \
            pkill -9 -f depth_to_occupancy 2>/dev/null; pkill -9 -f sando_py_bridge 2>/dev/null; \
            pkill -9 -f robot_state_publisher 2>/dev/null; pkill -9 -f term_goal 2>/dev/null; }
trap cleanup EXIT INT TERM
cleanup; sleep 2

echo ">> Gazebo server (headless) + GUI (gzclient)"
gzserver --verbose "$BRIDGE_DIR/minimal_state.world" \
  -s libgazebo_ros_init.so -s libgazebo_ros_factory.so > /tmp/gz.log 2>&1 &
sleep 7
gzclient > /tmp/gzclient.log 2>&1 &     # 3D view of the drone + boxes (optional; close it freely)

echo ">> spawn drone + TF + fake_sim"
xacro "$SANDO_WS/src/sando/urdf/quadrotor.urdf.xacro" namespace:=NX01 > /tmp/quad.urdf 2>/dev/null
ros2 run gazebo_ros spawn_entity.py -entity NX01 -file /tmp/quad.urdf -x 0 -y 0 -z 2.0 >/tmp/spawn.log 2>&1
python3 - <<'PY'
import yaml
u=open('/tmp/quad.urdf').read()
yaml.safe_dump({'/**':{'ros__parameters':{'frame_prefix':'NX01/','use_sim_time':False,'robot_description':u}}},
              open('/tmp/rsp.yaml','w'), default_style='|', allow_unicode=True)
PY
ros2 run robot_state_publisher robot_state_publisher --ros-args -r __ns:=/NX01 --params-file /tmp/rsp.yaml >/tmp/rsp.log 2>&1 &
ros2 run sando fake_sim --ros-args -r __ns:=/NX01 -p start_pos:="[0.0,0.0,2.0]" -p start_yaw:=0.0 \
  -p send_state_to_gazebo:=true -p visual_level:=2 -p publish_odom:=true -p odom_topic:=visual_slam/odom \
  -p odom_frame_id:=map -p base_frame_id:=NX01/base_link >/tmp/fakesim.log 2>&1 &
sleep 4

echo ">> obstacle boxes"
cat > /tmp/box.sdf <<'SDF'
<?xml version="1.0"?>
<sdf version="1.6"><model name="b"><static>true</static><link name="l">
<collision name="c"><geometry><box><size>1 1 3</size></box></geometry></collision>
<visual name="v"><geometry><box><size>1 1 3</size></box></geometry></visual></link></model></sdf>
SDF
ros2 run gazebo_ros spawn_entity.py -entity obs1 -file /tmp/box.sdf -x 6 -y -0.6 -z 1.5 >/dev/null 2>&1
ros2 run gazebo_ros spawn_entity.py -entity obs2 -file /tmp/box.sdf -x 10 -y 0.8 -z 1.5 >/dev/null 2>&1

echo ">> mapper + your sando-py bridge"
python3 "$BRIDGE_DIR/depth_to_occupancy.py" >/tmp/mapper.log 2>&1 &
python3 "$BRIDGE_DIR/sando_py_bridge.py" --ros-args -r __ns:=/NX01 -p v_max:=2.0 -p a_max:=6.0 >/tmp/bridge.log 2>&1 &
sleep 5

echo ">> goal ping-pong (16,0,2) <-> (0,0,2) every 12s"
( while true; do
    ros2 topic pub --once /NX01/term_goal geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: map}, pose: {position: {x: 16.0, y: 0.0, z: 2.0}}}" >/dev/null 2>&1; sleep 12
    ros2 topic pub --once /NX01/term_goal geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: map}, pose: {position: {x: 0.0, y: 0.0, z: 2.0}}}" >/dev/null 2>&1; sleep 12
  done ) &

echo ">> RViz (close it or Ctrl-C here to stop everything)"
RVIZ="$SANDO_WS/install/sando/share/sando/rviz/sando.rviz"
if [ -f "$RVIZ" ]; then rviz2 -d "$RVIZ"; else rviz2; fi
