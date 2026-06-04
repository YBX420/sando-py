// perclass_obstacle_pub.cpp — C++ ROS2 port of sando_py/nodes/perclass_obstacle_pub.py.
// Publishes a HARD human (id 100, sin y-oscillation, analytic) + SOFT wall (id 200, static)
// as DynTraj at 10Hz on `predicted_trajs`. (RViz markers omitted — cosmetic.)
#include <rclcpp/rclcpp.hpp>
#include <dynus_interfaces/msg/dyn_traj.hpp>
#include <cmath>
#include <string>
#include <cstdio>
#include <chrono>
using namespace std::chrono_literals;

static std::string num(double v) { char b[64]; std::snprintf(b, sizeof(b), "%.12g", v); return b; }

class PerClassObstaclePub : public rclcpp::Node {
 public:
  PerClassObstaclePub() : rclcpp::Node("perclass_obstacle_pub") {
    frame_ = declare_parameter<std::string>("frame_id", "map");
    hx_ = declare_parameter<double>("human_x", 5.0);
    amp_ = declare_parameter<double>("human_amp", 2.5);
    period_ = declare_parameter<double>("human_period", 6.0);
    hz_ = declare_parameter<double>("human_z", 1.0);
    wc_ = declare_parameter<std::vector<double>>("wall_center", {5.0, 1.2, 2.0});
    we_ = declare_parameter<std::vector<double>>("wall_extents", {2.0, 0.4, 3.0});
    w_ = 2.0 * M_PI / std::max(period_, 1e-3);
    t0_ = now().seconds();
    pub_ = create_publisher<dynus_interfaces::msg::DynTraj>("predicted_trajs", 10);
    timer_ = create_wall_timer(100ms, std::bind(&PerClassObstaclePub::tick, this));
    RCLCPP_INFO(get_logger(), "perclass_obstacle_pub (C++ port) up");
  }

 private:
  dynus_interfaces::msg::DynTraj make(int id, const std::vector<double>& bbox,
                                      const std::string& fx, const std::string& fy, const std::string& fz,
                                      const std::string& vx, const std::string& vy, const std::string& vz,
                                      double px, double py, double pz) {
    dynus_interfaces::msg::DynTraj m;
    m.header.stamp = now(); m.header.frame_id = frame_;
    m.id = id; m.is_agent = false;
    m.bbox = bbox;
    m.function = {fx, fy, fz};
    m.velocity = {vx, vy, vz};
    m.pos.x = px; m.pos.y = py; m.pos.z = pz;
    return m;
  }
  void tick() {
    double t = now().seconds();
    double hpy = amp_ * std::sin(w_ * (t - t0_));
    std::string arg = "(t - " + num(t0_) + ")";
    auto human = make(100, {0.6, 0.6, 1.8},
                      num(hx_), num(amp_) + "*sin(" + num(w_) + "*" + arg + ")", num(hz_),
                      "0.0", num(amp_ * w_) + "*cos(" + num(w_) + "*" + arg + ")", "0.0",
                      hx_, hpy, hz_);
    auto wall = make(200, we_, num(wc_[0]), num(wc_[1]), num(wc_[2]),
                     "0.0", "0.0", "0.0", wc_[0], wc_[1], wc_[2]);
    pub_->publish(human);
    pub_->publish(wall);
  }
  std::string frame_;
  double hx_, amp_, period_, hz_, w_, t0_;
  std::vector<double> wc_, we_;
  rclcpp::Publisher<dynus_interfaces::msg::DynTraj>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PerClassObstaclePub>());
  rclcpp::shutdown();
  return 0;
}
