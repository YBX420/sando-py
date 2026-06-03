// fake_sim.cpp — C++ ROS2 port of sando_py/nodes/fake_sim.py (kinematic loop-closer).
// Subscribes Goal, ideal-follows it (state := goal), publishes State (+ TF + optional Odom)
// at 100Hz with a constant-accel extrapolation. Hopf-fibration attitude (matches the C++ baseline).
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <dynus_interfaces/msg/state.hpp>
#include <dynus_interfaces/msg/goal.hpp>

#include <array>
#include <cmath>
#include <mutex>
#include <chrono>
using namespace std::chrono_literals;

static std::array<double, 4> quat_mul(const std::array<double, 4>& a, const std::array<double, 4>& b) {
  double ax = a[0], ay = a[1], az = a[2], aw = a[3];
  double bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return {aw * bx + ax * bw + ay * bz - az * by,
          aw * by - ax * bz + ay * bw + az * bx,
          aw * bz + ax * by - ay * bx + az * bw,
          aw * bw - ax * bx - ay * by - az * bz};
}
static std::array<double, 4> hopf_quat(double tx, double ty, double tz, double yaw) {
  double gx = tx, gy = ty, gz = tz + 9.81;
  double n = std::sqrt(gx * gx + gy * gy + gz * gz);
  if (n < 1e-9) return {0.0, 0.0, std::sin(yaw / 2), std::cos(yaw / 2)};
  double a = gx / n, b = gy / n, c = gz / n;
  double tmp = 1.0 / std::sqrt(2.0 * (1.0 + c));
  std::array<double, 4> qabc = {-b * tmp, a * tmp, 0.0, tmp * (1.0 + c)};
  std::array<double, 4> qpsi = {0.0, 0.0, std::sin(yaw / 2), std::cos(yaw / 2)};
  return quat_mul(qabc, qpsi);
}

class FakeSim : public rclcpp::Node {
 public:
  FakeSim() : rclcpp::Node("fake_sim") {
    auto sp = declare_parameter<std::vector<double>>("start_pos", {0.0, 0.0, 4.0});
    double start_yaw = declare_parameter<double>("start_yaw", -1.57);
    publish_odom_ = declare_parameter<bool>("publish_odom", false);
    odom_topic_ = declare_parameter<std::string>("odom_topic", "odom");
    std::string ns = get_namespace(); if (!ns.empty() && ns[0] == '/') ns = ns.substr(1);
    target_frame_ = ns + "/base_link";

    state_pos_ = {sp[0], sp[1], sp[2]}; state_vel_ = {0, 0, 0};
    state_quat_ = hopf_quat(0, 0, 0, start_yaw);
    goal_pos_ = {0, 0, 0}; goal_vel_ = {0, 0, 0}; goal_acc_ = {0, 0, 0};
    goal_quat_ = {0, 0, 0, 1};

    pub_state_ = create_publisher<dynus_interfaces::msg::State>("state", 10);
    if (publish_odom_) pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    sub_goal_ = create_subscription<dynus_interfaces::msg::Goal>(
        "goal", 10, std::bind(&FakeSim::goal_cb, this, std::placeholders::_1));
    br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    timer_ = create_wall_timer(10ms, std::bind(&FakeSim::tick, this));
    RCLCPP_INFO(get_logger(), "FakeSim (C++ port) initialized");
  }

 private:
  void goal_cb(const dynus_interfaces::msg::Goal::SharedPtr msg) {
    auto q = hopf_quat(msg->a.x, msg->a.y, msg->a.z, msg->yaw);
    std::lock_guard<std::mutex> lk(mtx_);
    goal_stamp_ = now();
    goal_pos_ = {msg->p.x, msg->p.y, msg->p.z};
    goal_vel_ = {msg->v.x, msg->v.y, msg->v.z};
    goal_acc_ = {msg->a.x, msg->a.y, msg->a.z};
    goal_quat_ = q;
    goal_received_ = true;
    state_pos_ = goal_pos_; state_vel_ = goal_vel_; state_quat_ = q;
  }

  void tick() {
    rclcpp::Time t = now();
    std::array<double, 3> p; std::array<double, 4> quat;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (goal_received_) {
        double dt = std::max(0.0, std::min(0.1, (t - goal_stamp_).seconds()));
        for (int i = 0; i < 3; ++i) p[i] = goal_pos_[i] + goal_vel_[i] * dt + 0.5 * goal_acc_[i] * dt * dt;
        quat = goal_quat_;
      } else { p = state_pos_; quat = state_quat_; }
    }
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = t; tf.header.frame_id = "map"; tf.child_frame_id = target_frame_;
    tf.transform.translation.x = p[0]; tf.transform.translation.y = p[1]; tf.transform.translation.z = p[2];
    tf.transform.rotation.x = quat[0]; tf.transform.rotation.y = quat[1];
    tf.transform.rotation.z = quat[2]; tf.transform.rotation.w = quat[3];
    br_->sendTransform(tf);

    dynus_interfaces::msg::State s;
    s.header.stamp = t; s.header.frame_id = "map";
    s.pos.x = p[0]; s.pos.y = p[1]; s.pos.z = p[2];
    s.vel.x = state_vel_[0]; s.vel.y = state_vel_[1]; s.vel.z = state_vel_[2];
    s.quat.x = quat[0]; s.quat.y = quat[1]; s.quat.z = quat[2]; s.quat.w = quat[3];
    pub_state_->publish(s);

    if (pub_odom_) {
      nav_msgs::msg::Odometry odom;
      odom.header.stamp = t; odom.header.frame_id = "map"; odom.child_frame_id = target_frame_;
      odom.pose.pose.position.x = p[0]; odom.pose.pose.position.y = p[1]; odom.pose.pose.position.z = p[2];
      odom.pose.pose.orientation.x = quat[0]; odom.pose.pose.orientation.y = quat[1];
      odom.pose.pose.orientation.z = quat[2]; odom.pose.pose.orientation.w = quat[3];
      odom.twist.twist.linear.x = state_vel_[0]; odom.twist.twist.linear.y = state_vel_[1];
      odom.twist.twist.linear.z = state_vel_[2];
      pub_odom_->publish(odom);
    }
  }

  std::mutex mtx_;
  bool goal_received_ = false, publish_odom_ = false;
  std::string odom_topic_, target_frame_;
  std::array<double, 3> state_pos_, state_vel_, goal_pos_, goal_vel_, goal_acc_;
  std::array<double, 4> state_quat_, goal_quat_;
  rclcpp::Time goal_stamp_;
  rclcpp::Publisher<dynus_interfaces::msg::State>::SharedPtr pub_state_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Subscription<dynus_interfaces::msg::Goal>::SharedPtr sub_goal_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> br_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FakeSim>());
  rclcpp::shutdown();
  return 0;
}
