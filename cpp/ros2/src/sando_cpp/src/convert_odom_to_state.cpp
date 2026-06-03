// convert_odom_to_state.cpp — C++ ROS2 port of sando_py/nodes/convert_odom_to_state.py.
// Bridge: nav_msgs/Odometry -> dynus_interfaces/State. Pure field copy, no transform.
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <dynus_interfaces/msg/state.hpp>

class OdometryToState : public rclcpp::Node {
 public:
  OdometryToState() : rclcpp::Node("odometry_to_state_node") {
    pub_ = create_publisher<dynus_interfaces::msg::State>("state", 10);
    sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "odom", 10, std::bind(&OdometryToState::cb, this, std::placeholders::_1));
  }

 private:
  void cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    dynus_interfaces::msg::State s;
    s.header.stamp = msg->header.stamp;
    s.header.frame_id = msg->header.frame_id;
    s.pos.x = msg->pose.pose.position.x;
    s.pos.y = msg->pose.pose.position.y;
    s.pos.z = msg->pose.pose.position.z;
    s.vel.x = msg->twist.twist.linear.x;
    s.vel.y = msg->twist.twist.linear.y;
    s.vel.z = msg->twist.twist.linear.z;
    s.quat.x = msg->pose.pose.orientation.x;
    s.quat.y = msg->pose.pose.orientation.y;
    s.quat.z = msg->pose.pose.orientation.z;
    s.quat.w = msg->pose.pose.orientation.w;
    pub_->publish(s);
  }
  rclcpp::Publisher<dynus_interfaces::msg::State>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdometryToState>());
  rclcpp::shutdown();
  return 0;
}
