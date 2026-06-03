// odom_to_global_state.cpp — C++ ROS2 port of sando_py/nodes/odom_to_global_state.py.
// Looks up TF world -> <veh>/init_pose once, caches (t_init, q_init, R_init), then maps every
// local Odometry into the global `world` frame:  p_g = R_init*p_l + t_init,  v_g = R_init*v_l,
// q_g = q_init (x) q_l (normalized). Publishes State (`state`) + PoseStamped (`global_pose`).
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <dynus_interfaces/msg/state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
using namespace std::chrono_literals;

static Eigen::Vector4d quat_mul(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
  double ax = a[0], ay = a[1], az = a[2], aw = a[3];
  double bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return {aw * bx + ax * bw + ay * bz - az * by,
          aw * by - ax * bz + ay * bw + az * bx,
          aw * bz + ax * by - ay * bx + az * bw,
          aw * bw - ax * bx - ay * by - az * bz};
}
static Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q) {
  double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  Eigen::Matrix3d R;
  R << 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
       2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
       2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy);
  return R;
}

class OdomToGlobalState : public rclcpp::Node {
 public:
  OdomToGlobalState() : rclcpp::Node("odom_to_global_state") {
    std::string ns = get_namespace(); if (ns.empty()) ns = "/";
    while (!ns.empty() && ns.back() == '/') ns.pop_back();
    auto p = ns.find_last_of('/');
    veh_name_ = (p == std::string::npos) ? ns : ns.substr(p + 1);
    if (veh_name_.empty()) veh_name_ = "PX04";

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pub_state_ = create_publisher<dynus_interfaces::msg::State>("state", 10);
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>("global_pose", 10);
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "odom", 10, std::bind(&OdomToGlobalState::odom_cb, this, std::placeholders::_1));
    tf_timer_ = create_wall_timer(100ms, std::bind(&OdomToGlobalState::poll_tf, this));
    RCLCPP_INFO(get_logger(), "Waiting for TF: world -> %s/init_pose", veh_name_.c_str());
  }

 private:
  void poll_tf() {
    std::string source = veh_name_ + "/init_pose";
    if (!tf_buffer_->canTransform("world", source, tf2::TimePointZero)) return;
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform("world", source, tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF lookup failed: %s", ex.what());
      return;
    }
    t_init_ = Eigen::Vector3d(tf.transform.translation.x, tf.transform.translation.y,
                              tf.transform.translation.z);
    q_init_ = Eigen::Vector4d(tf.transform.rotation.x, tf.transform.rotation.y,
                              tf.transform.rotation.z, tf.transform.rotation.w);
    R_init_ = quat_to_rot(q_init_);
    tf_acquired_ = true;
    tf_timer_->cancel();
    double yaw = std::atan2(R_init_(1, 0), R_init_(0, 0));
    RCLCPP_INFO(get_logger(), "TF acquired: t=[%.3f, %.3f, %.3f], yaw=%.3f rad",
                t_init_[0], t_init_[1], t_init_[2], yaw);
  }

  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    if (!tf_acquired_) return;
    Eigen::Vector3d lp(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Vector3d lv(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    Eigen::Vector4d lq(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                       msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    Eigen::Vector3d gp = R_init_ * lp + t_init_;
    Eigen::Vector3d gv = R_init_ * lv;
    Eigen::Vector4d gq = quat_mul(q_init_, lq);
    double n = gq.norm();
    if (n > 1e-9) gq /= n;

    dynus_interfaces::msg::State s;
    s.header.stamp = msg->header.stamp; s.header.frame_id = "world";
    s.pos.x = gp[0]; s.pos.y = gp[1]; s.pos.z = gp[2];
    s.vel.x = gv[0]; s.vel.y = gv[1]; s.vel.z = gv[2];
    s.quat.x = gq[0]; s.quat.y = gq[1]; s.quat.z = gq[2]; s.quat.w = gq[3];
    pub_state_->publish(s);

    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = msg->header.stamp; ps.header.frame_id = "world";
    ps.pose.position.x = gp[0]; ps.pose.position.y = gp[1]; ps.pose.position.z = gp[2];
    ps.pose.orientation.x = gq[0]; ps.pose.orientation.y = gq[1];
    ps.pose.orientation.z = gq[2]; ps.pose.orientation.w = gq[3];
    pub_pose_->publish(ps);
  }

  std::string veh_name_;
  bool tf_acquired_ = false;
  Eigen::Vector3d t_init_ = Eigen::Vector3d::Zero();
  Eigen::Vector4d q_init_ = Eigen::Vector4d(0, 0, 0, 1);
  Eigen::Matrix3d R_init_ = Eigen::Matrix3d::Identity();
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<dynus_interfaces::msg::State>::SharedPtr pub_state_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomToGlobalState>());
  rclcpp::shutdown();
  return 0;
}
