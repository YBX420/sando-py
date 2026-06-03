// convert_vicon_to_state.cpp — C++ ROS2 port of sando_py/nodes/convert_vicon_to_state.py.
// Vicon PoseStamped (`world`) + TwistStamped (`twist`) -> dynus_interfaces/State (`state`).
// Python re-implements message_filters ApproximateTime by hand (nearest twist within 30ms
// slop of the latest pose); we reproduce that exact match logic.
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <dynus_interfaces/msg/state.hpp>
#include <deque>
#include <cmath>

static double stamp_seconds(const builtin_interfaces::msg::Time& s) {
  return static_cast<double>(s.sec) + static_cast<double>(s.nanosec) * 1e-9;
}

class PoseTwistToState : public rclcpp::Node {
 public:
  PoseTwistToState() : rclcpp::Node("pose_twist_to_state_node") {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.best_effort().transient_local();  // matches the Python QoS exactly
    sub_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "world", qos, std::bind(&PoseTwistToState::pose_cb, this, std::placeholders::_1));
    sub_twist_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "twist", qos, std::bind(&PoseTwistToState::twist_cb, this, std::placeholders::_1));
    pub_ = create_publisher<dynus_interfaces::msg::State>("state", 10);
  }

 private:
  void pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr m) {
    if (poses_.size() >= 20) poses_.pop_front();
    poses_.push_back(*m);
    try_pair();
  }
  void twist_cb(const geometry_msgs::msg::TwistStamped::SharedPtr m) {
    if (twists_.size() >= 20) twists_.pop_front();
    twists_.push_back(*m);
    try_pair();
  }
  void try_pair() {
    if (poses_.empty() || twists_.empty()) return;
    const auto& latest_pose = poses_.back();
    double t_pose = stamp_seconds(latest_pose.header.stamp);
    double best_dt = std::numeric_limits<double>::infinity();
    const geometry_msgs::msg::TwistStamped* best = nullptr;
    for (const auto& tw : twists_) {
      double dt = std::abs(stamp_seconds(tw.header.stamp) - t_pose);
      if (dt < best_dt) { best_dt = dt; best = &tw; }
    }
    if (best == nullptr || best_dt > slop_) return;
    dynus_interfaces::msg::State s;
    s.header.stamp = now();
    s.header.frame_id = latest_pose.header.frame_id;
    s.pos.x = latest_pose.pose.position.x;
    s.pos.y = latest_pose.pose.position.y;
    s.pos.z = latest_pose.pose.position.z;
    s.vel.x = best->twist.linear.x;
    s.vel.y = best->twist.linear.y;
    s.vel.z = best->twist.linear.z;
    s.quat.x = latest_pose.pose.orientation.x;
    s.quat.y = latest_pose.pose.orientation.y;
    s.quat.z = latest_pose.pose.orientation.z;
    s.quat.w = latest_pose.pose.orientation.w;
    pub_->publish(s);
  }
  std::deque<geometry_msgs::msg::PoseStamped> poses_;
  std::deque<geometry_msgs::msg::TwistStamped> twists_;
  double slop_ = 0.03;  // 30 ms sync slop
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_twist_;
  rclcpp::Publisher<dynus_interfaces::msg::State>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseTwistToState>());
  rclcpp::shutdown();
  return 0;
}
