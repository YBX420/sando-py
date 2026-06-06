// obstacle_tracker.cpp — C++ ROS2 port of sando_py/nodes/obstacle_tracker.py.
// PointCloud2 -> voxel downsample -> z passthrough -> connected-components clustering ->
// per-cluster adaptive EKF (9-D const-accel) -> clipped extrapolation -> cubic+quintic polyfit
// -> DynTraj on `predicted_trajs`. All numeric kernels are golden-verified against Python
// (see cpp/test/test_obstacle_tracker.cpp). RViz markers included (cosmetic, color is arbitrary).
//
// Clustering note: matches the Python's cKDTree+union-find FALLBACK path (sklearn absent);
// a real sklearn-DBSCAN run would differ on adversarial inputs. Documented in cpp/README.md.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <dynus_interfaces/msg/dyn_traj.hpp>

#include "sando_cpp/obstacle_tracker_kernels.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

using namespace sando::obstacle_tracker;

class ObstacleTracker : public rclcpp::Node {
 public:
  ObstacleTracker() : rclcpp::Node("obstacle_tracker_node") {
    visual_level_ = declare_parameter<int>("visual_level", 1);
    use_adaptive_ = declare_parameter<bool>("use_adaptive_kf", true);
    alpha_ = declare_parameter<double>("adaptive_kf_alpha", 0.98);
    kf_dt_ = declare_parameter<double>("adaptive_kf_dt", 0.1);
    cluster_tol_ = declare_parameter<double>("cluster_tolerance", 2.0);
    min_size_ = declare_parameter<int>("min_cluster_size", 10);
    max_size_ = declare_parameter<int>("max_cluster_size", 2000);
    horizon_ = declare_parameter<double>("prediction_horizon", 2.0);
    pred_dt_ = declare_parameter<double>("prediction_dt", 0.1);
    delete_old_ = declare_parameter<double>("time_to_delete_old_obstacles", 10.0);
    bbox_cutoff_ = declare_parameter<double>("cluster_bbox_cutoff_size", 5.0);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "point_cloud", 10, std::bind(&ObstacleTracker::cloud_cb, this, std::placeholders::_1));
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("tracked_obstacles", 10);
    pub_bboxes_ = create_publisher<visualization_msgs::msg::MarkerArray>("cluster_bounding_boxes", 10);
    pub_traj_ = create_publisher<dynus_interfaces::msg::DynTraj>("predicted_trajs", 10);
    RCLCPP_INFO(get_logger(), "Obstacle Tracker Node (C++ port) Initialized");
  }

 private:
  double ros_now() { return static_cast<double>(get_clock()->now().nanoseconds()) * 1e-9; }

  void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(msg->width * msg->height);
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y"), iz(*msg, "z");
    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      float x = *ix, y = *iy, z = *iz;
      if (std::isnan(x) || std::isnan(y) || std::isnan(z)) continue;
      points.emplace_back(x, y, z);
    }
    if (points.empty()) { RCLCPP_WARN(get_logger(), "Received empty point cloud!"); return; }

    points = voxel_downsample(points, 0.2);
    std::vector<Eigen::Vector3d> filt;
    for (auto& p : points) if (p[2] >= 0.5 && p[2] <= 6.0) filt.push_back(p);

    auto clusters = connected_components(filt, cluster_tol_, min_size_, max_size_);
    std::vector<Eigen::Vector3d> centroids, bboxes;
    for (auto& c : clusters) {
      Eigen::Vector3d pmin = filt[c[0]], pmax = filt[c[0]];
      for (int idx : c) { pmin = pmin.cwiseMin(filt[idx]); pmax = pmax.cwiseMax(filt[idx]); }
      centroids.push_back((pmin + pmax) * 0.5);
      bboxes.push_back(pmax - pmin);
    }

    double now = ros_now();
    states_.erase(std::remove_if(states_.begin(), states_.end(),
                  [&](const EKFState& s) { return now - s.time_updated > delete_old_; }), states_.end());

    // record cluster->track as indices into states_ (pointers would dangle on reallocation)
    std::vector<int> records;
    for (size_t i = 0; i < centroids.size(); ++i) {
      if (bboxes[i].norm() > bbox_cutoff_) continue;
      int idx = associate_cluster(centroids[i], states_, cluster_tol_);
      if (idx >= 0) {
        ekf_predict(states_[idx], kf_dt_);
        aekf_update(states_[idx], centroids[i], alpha_, now, bboxes[i], use_adaptive_);
        records.push_back(idx);
      } else {
        states_.push_back(make_state(now, bboxes[i], centroids[i]));
        records.push_back(static_cast<int>(states_.size()) - 1);
      }
    }
    publish_predictions(records);
  }

  EKFState make_state(double now, const Eigen::Vector3d& bbox, const Eigen::Vector3d& centroid) {
    EKFState s;
    average_q_r(s.Q, s.R);
    s.time_updated = now; s.bbox = bbox; s.id = ekf_id_++;
    s.x.setZero(); s.x.head<3>() = centroid;
    return s;
  }
  void average_q_r(Eigen::Matrix<double, 9, 9>& Q, Eigen::Matrix3d& R) {
    if (states_.empty()) { Q = Eigen::Matrix<double, 9, 9>::Identity() * 0.01; R = Eigen::Matrix3d::Identity() * 0.01; return; }
    Q.setZero(); R.setZero();
    for (auto& s : states_) { Q += s.Q; R += s.R; }
    Q /= static_cast<double>(states_.size()); R /= static_cast<double>(states_.size());
  }

  void publish_predictions(const std::vector<int>& records) {
    int num_steps = static_cast<int>(horizon_ / pred_dt_);
    visualization_msgs::msg::MarkerArray ma;
    {
      visualization_msgs::msg::Marker clear;
      clear.action = visualization_msgs::msg::Marker::DELETEALL;
      clear.header.frame_id = frame_id_; clear.header.stamp = now();
      ma.markers.push_back(clear);
    }
    int marker_id = 0;
    for (int ridx : records) {
      EKFState& s = states_[ridx];
      auto samp = predict_samples(s.x, num_steps, pred_dt_);
      double cutoff = 0.1;
      if (std::abs(samp.x.front() - samp.x.back()) < cutoff &&
          std::abs(samp.y.front() - samp.y.back()) < cutoff &&
          std::abs(samp.z.front() - samp.z.back()) < cutoff) continue;

      Eigen::VectorXd t = Eigen::Map<Eigen::VectorXd>(samp.t.data(), samp.t.size());
      Eigen::VectorXd xx = Eigen::Map<Eigen::VectorXd>(samp.x.data(), samp.x.size());
      Eigen::VectorXd yy = Eigen::Map<Eigen::VectorXd>(samp.y.data(), samp.y.size());
      Eigen::VectorXd zz = Eigen::Map<Eigen::VectorXd>(samp.z.data(), samp.z.size());
      Eigen::VectorXd bx = polyfit(t, xx, 3), by = polyfit(t, yy, 3), bz = polyfit(t, zz, 3);
      double vx = calculate_variance(t, xx, bx, 3), vy = calculate_variance(t, yy, by, 3),
             vz = calculate_variance(t, zz, bz, 3);
      Eigen::VectorXd qx = polyfit(t, xx, 5), qy = polyfit(t, yy, 5), qz = polyfit(t, zz, 5);

      double t0 = ros_now();
      // samples only reach t=(num_steps-1)*dt (= samp.t.back()), one dt short of horizon_;
      // declare the window to the actual last sample so the poly is never extrapolated past fit data.
      double t_end_rel = samp.t.back();
      dynus_interfaces::msg::DynTraj m;
      m.header.stamp = now(); m.header.frame_id = frame_id_;
      m.id = s.id;
      m.bbox = {s.bbox[0], s.bbox[1], s.bbox[2]};
      m.pwp.times = {t0, t0 + t_end_rel};
      auto cp = [](const Eigen::VectorXd& b) {
        dynus_interfaces::msg::CoeffPoly3 c; c.a = b[0]; c.b = b[1]; c.c = b[2]; c.d = b[3]; return c;
      };
      m.pwp.coeff_x = {cp(bx)}; m.pwp.coeff_y = {cp(by)}; m.pwp.coeff_z = {cp(bz)};
      m.ekf_cov_p = {s.P(0, 0), s.P(1, 1), s.P(2, 2)};
      m.ekf_cov_q = {s.Q(0, 0), s.Q(1, 1), s.Q(2, 2)};
      m.ekf_cov_r = {s.R(0, 0), s.R(1, 1), s.R(2, 2)};
      m.poly_cov = {vx, vy, vz};
      m.poly_coeffs_x.assign(qx.data(), qx.data() + qx.size());
      m.poly_coeffs_y.assign(qy.data(), qy.data() + qy.size());
      m.poly_coeffs_z.assign(qz.data(), qz.data() + qz.size());
      m.poly_start_time = t0; m.poly_end_time = t0 + t_end_rel;   // window trimmed to fit-data end, no extrapolation
      m.pos.x = s.x[0]; m.pos.y = s.x[1]; m.pos.z = s.x[2];
      m.is_agent = false;
      pub_traj_->publish(m);

      if (visual_level_ >= 0) {
        Eigen::Vector3d pos = s.x.head<3>();
        for (int k = 1; k < (int)samp.x.size(); ++k) {
          visualization_msgs::msg::Marker mk;
          mk.header.frame_id = frame_id_; mk.header.stamp = now();
          mk.id = marker_id++; mk.type = visualization_msgs::msg::Marker::ARROW;
          mk.action = visualization_msgs::msg::Marker::ADD;
          geometry_msgs::msg::Point p0, p1;
          p0.x = samp.x[k - 1]; p0.y = samp.y[k - 1]; p0.z = samp.z[k - 1];
          p1.x = samp.x[k]; p1.y = samp.y[k]; p1.z = samp.z[k];
          mk.points = {p0, p1};
          mk.scale.x = 0.1; mk.scale.y = 0.2;
          mk.color.r = 0.2f; mk.color.g = 0.8f; mk.color.b = 0.2f; mk.color.a = 0.6f;
          ma.markers.push_back(mk);
        }
      }
    }
    if (visual_level_ >= 0) pub_markers_->publish(ma);
  }

  int visual_level_, min_size_, max_size_, ekf_id_ = 0;
  bool use_adaptive_;
  double alpha_, kf_dt_, cluster_tol_, horizon_, pred_dt_, delete_old_, bbox_cutoff_;
  std::string frame_id_ = "map";
  std::vector<EKFState> states_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_, pub_bboxes_;
  rclcpp::Publisher<dynus_interfaces::msg::DynTraj>::SharedPtr pub_traj_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleTracker>());
  rclcpp::shutdown();
  return 0;
}
