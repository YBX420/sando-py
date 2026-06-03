// auto_goal_pub.cpp — C++ ROS2 port of sando_py/nodes/auto_goal_pub.py.
// After `delay_s`, publishes a PoseStamped on `term_goal` (latched, 3x to beat races),
// then flips between end A / end B every `switch_period_s`.
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <array>
#include <chrono>
using namespace std::chrono_literals;

class AutoGoalPub : public rclcpp::Node {
 public:
  AutoGoalPub() : rclcpp::Node("auto_goal_pub") {
    frame_ = declare_parameter<std::string>("frame_id", "map");
    targets_[0] = {declare_parameter<double>("goal_x", 10.0), declare_parameter<double>("goal_y", 0.0),
                   declare_parameter<double>("goal_z", 2.0)};
    targets_[1] = {declare_parameter<double>("goal_x2", 0.0), declare_parameter<double>("goal_y2", 0.0),
                   declare_parameter<double>("goal_z2", 2.0)};
    delay_ = declare_parameter<double>("delay_s", 3.0);
    switch_period_ = declare_parameter<double>("switch_period_s", 7.0);

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable().transient_local();  // latched
    pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("term_goal", qos);
    delay_timer_ = create_wall_timer(std::chrono::duration<double>(delay_), std::bind(&AutoGoalPub::start, this));
  }

 private:
  geometry_msgs::msg::PoseStamped msg(const std::array<double, 3>& g) {
    geometry_msgs::msg::PoseStamped m;
    m.header.stamp = now(); m.header.frame_id = frame_;
    m.pose.position.x = g[0]; m.pose.position.y = g[1]; m.pose.position.z = g[2];
    m.pose.orientation.w = 1.0;
    return m;
  }
  void send() {
    auto g = targets_[idx_];
    for (int i = 0; i < 3; ++i) pub_->publish(msg(g));
    RCLCPP_INFO(get_logger(), "term_goal -> [%.1f, %.1f, %.1f]", g[0], g[1], g[2]);
  }
  void start() {
    delay_timer_->cancel();
    send();
    switch_timer_ = create_wall_timer(std::chrono::duration<double>(switch_period_),
                                      std::bind(&AutoGoalPub::flip, this));
  }
  void flip() { idx_ = (idx_ + 1) % 2; send(); }

  std::string frame_;
  std::array<std::array<double, 3>, 2> targets_;
  double delay_, switch_period_;
  int idx_ = 0;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr delay_timer_, switch_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AutoGoalPub>());
  rclcpp::shutdown();
  return 0;
}
