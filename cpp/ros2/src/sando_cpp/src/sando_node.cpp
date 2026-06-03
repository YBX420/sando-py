// sando_node.cpp — C++ ROS2 node wrapping the golden-verified C++ port of the SANDO
// MINCO per-class planner (sando_cpp/planner.hpp). Faithful to sando_py/nodes/sando_node.py
// (MINCO / rviz_only path): State -> update_state, term_goal -> set_terminal_goal,
// DynTraj -> add_traj, replan timer -> replan(), goal timer -> get_next_goal() -> Goal.
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <dynus_interfaces/msg/state.hpp>
#include <dynus_interfaces/msg/goal.hpp>
#include <dynus_interfaces/msg/dyn_traj.hpp>

#include "sando_cpp/types.hpp"
#include "sando_cpp/planner.hpp"

#include <chrono>
#include <memory>
#include <cmath>

using namespace std::chrono_literals;

// yaw from quaternion (ZYX) — local copy of utils.py::yaw_from_quat to avoid pulling
// utils_math.hpp (which redefines project_point_to_sphere already present in planner.hpp).
static inline double node_yaw_from_quat(double qx, double qy, double qz, double qw) {
  double siny = 2.0 * (qw * qz + qx * qy);
  double cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny, cosy);
}

class SandoNode : public rclcpp::Node {
 public:
  SandoNode() : rclcpp::Node("sando_node") {
    // rviz_only + MINCO demo config (mirrors perclass_demo.launch.py overrides)
    par_.sim_env = "rviz_only";
    par_.local_solver = "minco";
    par_.global_planner = "astar_heat";
    par_.skip_initial_yawing = true;
    par_.vehicle_type = "uav";
    par_.force_goal_z = true;
    par_.default_goal_z = 2.0;
    planner_ = std::make_unique<sando::SANDO>(par_);

    sub_state_ = create_subscription<dynus_interfaces::msg::State>(
        "state", 10, std::bind(&SandoNode::state_cb, this, std::placeholders::_1));
    sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "term_goal", 10, std::bind(&SandoNode::term_goal_cb, this, std::placeholders::_1));
    sub_trajs_ = create_subscription<dynus_interfaces::msg::DynTraj>(
        "trajs", 50, std::bind(&SandoNode::trajs_cb, this, std::placeholders::_1));
    pub_goal_ = create_publisher<dynus_interfaces::msg::Goal>("goal", 10);

    planner_->update_occupancy_map_ptr({});  // rviz_only map pre-seed -> map_initialized

    timer_replan_ = create_wall_timer(10ms, std::bind(&SandoNode::replan_cb, this));
    timer_goal_ = create_wall_timer(std::chrono::duration<double>(par_.dc),
                                    std::bind(&SandoNode::publish_goal, this));
    RCLCPP_INFO(get_logger(), "sando_node (C++ port: planner.hpp) up — rviz_only + minco");
  }

 private:
  double now_s() { return this->get_clock()->now().seconds(); }

  void state_cb(const dynus_interfaces::msg::State::SharedPtr msg) {
    sando::RobotState st;
    st.pos = Eigen::Vector3d(msg->pos.x, msg->pos.y, msg->pos.z);
    st.vel = Eigen::Vector3d(msg->vel.x, msg->vel.y, msg->vel.z);
    st.yaw = node_yaw_from_quat(msg->quat.x, msg->quat.y, msg->quat.z, msg->quat.w);
    planner_->update_state(st);
  }

  void term_goal_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    sando::RobotState G;
    double gz = par_.force_goal_z ? par_.default_goal_z : msg->pose.position.z;
    G.pos = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, gz);
    planner_->set_terminal_goal(G);
  }

  void trajs_cb(const dynus_interfaces::msg::DynTraj::SharedPtr msg) {
    sando::DynTraj dt;
    dt.id = msg->id;
    dt.is_agent = msg->is_agent;
    if (msg->bbox.size() >= 3) dt.bbox = Eigen::Vector3d(msg->bbox[0], msg->bbox[1], msg->bbox[2]);
    dt.mode = "Analytic";
    if (msg->function.size() == 3) {
      dt.traj_x = msg->function[0]; dt.traj_y = msg->function[1]; dt.traj_z = msg->function[2];
    }
    if (msg->velocity.size() == 3) {
      dt.traj_vx = msg->velocity[0]; dt.traj_vy = msg->velocity[1]; dt.traj_vz = msg->velocity[2];
    }
    dt.compile_analytic();
    planner_->add_traj(dt, now_s());
  }

  void replan_cb() {
    auto t0 = std::chrono::steady_clock::now();
    planner_->replan(last_rt_, now_s());
    last_rt_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  void publish_goal() {
    if (planner_->get_drone_status() == static_cast<int>(sando::DroneStatus::GOAL_REACHED)) return;
    auto ng = planner_->get_next_goal();
    if (!ng.ok) return;
    dynus_interfaces::msg::Goal g;
    g.header.stamp = now();
    g.p.x = ng.goal.pos[0]; g.p.y = ng.goal.pos[1]; g.p.z = ng.goal.pos[2];
    g.v.x = ng.goal.vel[0]; g.v.y = ng.goal.vel[1]; g.v.z = ng.goal.vel[2];
    g.a.x = ng.goal.accel[0]; g.a.y = ng.goal.accel[1]; g.a.z = ng.goal.accel[2];
    g.j.x = ng.goal.jerk[0]; g.j.y = ng.goal.jerk[1]; g.j.z = ng.goal.jerk[2];
    g.yaw = ng.goal.yaw; g.dyaw = ng.goal.dyaw; g.power = true; g.mode_xy = 0; g.mode_z = 0;
    pub_goal_->publish(g);
  }

  sando::Parameters par_;
  std::unique_ptr<sando::SANDO> planner_;
  double last_rt_ = 0.0;
  rclcpp::Subscription<dynus_interfaces::msg::State>::SharedPtr sub_state_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::Subscription<dynus_interfaces::msg::DynTraj>::SharedPtr sub_trajs_;
  rclcpp::Publisher<dynus_interfaces::msg::Goal>::SharedPtr pub_goal_;
  rclcpp::TimerBase::SharedPtr timer_replan_, timer_goal_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SandoNode>());
  rclcpp::shutdown();
  return 0;
}
