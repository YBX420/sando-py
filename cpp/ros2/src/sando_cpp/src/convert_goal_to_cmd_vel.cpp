// convert_goal_to_cmd_vel.cpp — C++ ROS2 port of sando_py/nodes/convert_goal_to_cmd_vel.py.
// Lyapunov tracking controller turning a flat Goal into a ground-robot Twist (cmd_vel):
//   v_cmd    = v_d*cos(eyaw) - kx*ex
//   yawd_cmd = dyaw_d - v_d*(ky*ey + sin(eyaw))/sqrt(ey^2+eps^2) - kyaw*eyaw
// where (ex,ey) is the position error rotated into the desired-yaw frame. Control law
// identical to the Python (and the original C++ baseline).
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <dynus_interfaces/msg/goal.hpp>
#include <dynus_interfaces/msg/state.hpp>
#include <array>
#include <cmath>
#include <chrono>
using namespace std::chrono_literals;

static inline double yaw_from_quat(double qx, double qy, double qz, double qw) {
  double siny = 2.0 * (qw * qz + qx * qy);
  double cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny, cosy);
}

class GoalToCmdVel : public rclcpp::Node {
 public:
  GoalToCmdVel() : rclcpp::Node("goal_to_cmd_vel") {
    state_pos_ = {declare_parameter<double>("x", 0.0), declare_parameter<double>("y", 0.0),
                  declare_parameter<double>("z", 0.0)};
    state_yaw_ = declare_parameter<double>("yaw", 0.0);
    std::string cmd_topic = declare_parameter<std::string>("cmd_vel_topic_name", "cmd_vel");
    kx_ = declare_parameter<double>("ground_robot_kx", 1.0);
    ky_ = declare_parameter<double>("ground_robot_ky", 1.0);
    kyaw_ = declare_parameter<double>("ground_robot_kyaw", 1.0);
    eps_ = declare_parameter<double>("ground_robot_eps", 1e-2);
    RCLCPP_INFO(get_logger(), "kx=%g, ky=%g, kyaw=%g, eps=%g", kx_, ky_, kyaw_, eps_);

    pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic, 10);
    sub_goal_ = create_subscription<dynus_interfaces::msg::Goal>(
        "goal", 10, std::bind(&GoalToCmdVel::goal_cb, this, std::placeholders::_1));
    sub_state_ = create_subscription<dynus_interfaces::msg::State>(
        "state", 10, std::bind(&GoalToCmdVel::state_cb, this, std::placeholders::_1));
    timer_ = create_wall_timer(10ms, std::bind(&GoalToCmdVel::tick, this));
  }

 private:
  void state_cb(const dynus_interfaces::msg::State::SharedPtr m) {
    state_pos_ = {m->pos.x, m->pos.y, m->pos.z};
    state_yaw_ = yaw_from_quat(m->quat.x, m->quat.y, m->quat.z, m->quat.w);
    state_init_ = true;
  }
  void goal_cb(const dynus_interfaces::msg::Goal::SharedPtr m) {
    goal_p_ = {m->p.x, m->p.y, m->p.z};
    goal_v_ = {m->v.x, m->v.y, m->v.z};
    goal_yaw_ = m->yaw; goal_dyaw_ = m->dyaw;
    goal_init_ = true;
  }
  void tick() {
    if (!state_init_ || !goal_init_) return;
    double x_d = goal_p_[0], y_d = goal_p_[1];
    double v_d = std::hypot(goal_v_[0], goal_v_[1]);
    double sx = state_pos_[0], sy = state_pos_[1];
    double ex = std::cos(goal_yaw_) * (sx - x_d) + std::sin(goal_yaw_) * (sy - y_d);
    double ey = -std::sin(goal_yaw_) * (sx - x_d) + std::cos(goal_yaw_) * (sy - y_d);
    double eyaw = state_yaw_ - goal_yaw_;
    double v_cmd = v_d * std::cos(eyaw) - kx_ * ex;
    double denom = std::sqrt(ey * ey + eps_ * eps_);
    double yawd_cmd = goal_dyaw_ - v_d * (ky_ * ey + std::sin(eyaw)) / denom - kyaw_ * eyaw;
    geometry_msgs::msg::Twist t;
    t.linear.x = v_cmd; t.angular.z = yawd_cmd;
    pub_->publish(t);
  }
  std::array<double, 3> state_pos_{0, 0, 0}, goal_p_{0, 0, 0}, goal_v_{0, 0, 0};
  double state_yaw_ = 0, goal_yaw_ = 0, goal_dyaw_ = 0;
  double kx_, ky_, kyaw_, eps_;
  bool state_init_ = false, goal_init_ = false;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<dynus_interfaces::msg::Goal>::SharedPtr sub_goal_;
  rclcpp::Subscription<dynus_interfaces::msg::State>::SharedPtr sub_state_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GoalToCmdVel>());
  rclcpp::shutdown();
  return 0;
}
