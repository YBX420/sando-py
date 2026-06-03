// types.hpp — faithful C++ port of the NUMERIC core of sando_py/types.py.
//
// 不许欺骗、必须还原: every formula copied 1:1 from types.py, no approximation.
// Ports: PieceWisePol (piecewise cubic eval/velocity/acceleration/get_end_time/times),
//        _AnalyticExpr (safe recursive-descent evaluator of a string expr in variable t
//        with whitelist sin/cos/tan/asin/acos/atan/atan2/sqrt/exp/log/pow/abs + pi/e and
//        + - * / ** () and unary minus — matching Python eval semantics),
//        DynTraj ("Analytic" via _AnalyticExpr with numeric central-difference fallback
//        for velocity/accel exactly as Python; "Piecewise" via PieceWisePol),
//        plus data structs RobotState, Parameters and the DroneStatus enum.
//
// Verified against gen_types_golden.py (dumps the real Python module outputs).
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstddef>
#include <memory>

namespace sando {

// ===========================================================================
// DroneStatus (IntEnum)
// ===========================================================================
enum class DroneStatus : int {
  YAWING = 0,
  TRAVELING = 1,
  GOAL_SEEN = 2,
  GOAL_REACHED = 3,
  HOVER_AVOIDING = 4,
};

// ===========================================================================
// _poly3_deriv — cubic a u^3 + b u^2 + cc u + d, order-th derivative at u
// (1:1 with types.py::_poly3_deriv)
// ===========================================================================
inline double poly3_deriv(double a, double b, double cc, double d, double u, int order) {
  if (order == 0) return a * u * u * u + b * u * u + cc * u + d;
  if (order == 1) return 3.0 * a * u * u + 2.0 * b * u + cc;
  if (order == 2) return 6.0 * a * u + 2.0 * b;
  if (order == 3) return 6.0 * a;
  return 0.0;
}

// ===========================================================================
// PieceWisePol — piecewise cubic. Segment i spans [times[i], times[i+1]),
// local parameter u = t - times[i]; coeffs (a, b, c, d): p(u)=a u^3+b u^2+c u+d.
// ===========================================================================
class PieceWisePol {
 public:
  std::vector<double> times;
  // each coeff entry is a length-4 array {a, b, c, d}
  std::vector<Eigen::Vector4d> coeff_x;
  std::vector<Eigen::Vector4d> coeff_y;
  std::vector<Eigen::Vector4d> coeff_z;

  void clear() {
    times.clear();
    coeff_x.clear();
    coeff_y.clear();
    coeff_z.clear();
  }

  double get_duration() const {
    if (times.empty()) return 0.0;
    return times.back() - times.front();
  }

  double get_end_time() const { return times.empty() ? 0.0 : times.back(); }

  // order: 0 pos, 1 vel, 2 accel. Boundary handling mirrors Python exactly.
  double eval_axis(const std::vector<Eigen::Vector4d>& coeffs, double t, int order) const {
    if (times.empty() || coeffs.empty()) return 0.0;
    const std::size_t n = times.size();
    // t at/after the trajectory end: use last segment, u = full length of that segment
    if (t >= times[n - 1]) {
      double u = times[n - 1] - times[n - 2];
      const Eigen::Vector4d& c = coeffs.back();
      return poly3_deriv(c(0), c(1), c(2), c(3), u, order);
    }
    if (t < times[0]) {
      const Eigen::Vector4d& c = coeffs[0];
      return poly3_deriv(c(0), c(1), c(2), c(3), 0.0, order);
    }
    for (std::size_t i = 0; i + 1 < n; ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        const Eigen::Vector4d& c = coeffs[i];
        return poly3_deriv(c(0), c(1), c(2), c(3), u, order);
      }
    }
    return 0.0;
  }

  Eigen::Vector3d eval(double t) const {
    return Eigen::Vector3d(eval_axis(coeff_x, t, 0), eval_axis(coeff_y, t, 0),
                           eval_axis(coeff_z, t, 0));
  }
  Eigen::Vector3d velocity(double t) const {
    return Eigen::Vector3d(eval_axis(coeff_x, t, 1), eval_axis(coeff_y, t, 1),
                           eval_axis(coeff_z, t, 1));
  }
  Eigen::Vector3d acceleration(double t) const {
    return Eigen::Vector3d(eval_axis(coeff_x, t, 2), eval_axis(coeff_y, t, 2),
                           eval_axis(coeff_z, t, 2));
  }
};

// ===========================================================================
// _AnalyticExpr — safe recursive-descent parser/evaluator of a string
// expression in the single variable t.
//
// Replicates Python's compile/eval over the _ALLOWED_NAMES whitelist:
//   functions: sin cos tan asin acos atan atan2 sqrt exp log pow abs
//   constants: pi e
//   operators: + - * / **  (ExprTk-style ^ is rewritten to ** like Python does)
//   unary +/-, parentheses, scientific-notation numeric literals.
//
// Python precedence (and what we mirror):
//   expr   := term (('+'|'-') term)*
//   term   := factor (('*'|'/') factor)*
//   factor := ('+'|'-') factor | power           (unary binds looser than ** on its right)
//   power  := atom ['**' factor]                 (** is right-associative; RHS is a factor,
//                                                  so 2**-1 and -2**2==-4 match Python)
//   atom   := number | name | name'(' args ')' | '(' expr ')'
//
// Empty source -> callable returning 0.0 (matches __call__ with _fn is None).
// ===========================================================================
class AnalyticExpr {
 public:
  AnalyticExpr() : empty_(true) {}
  explicit AnalyticExpr(const std::string& src) : src_(src) {
    if (src.empty()) {
      empty_ = true;
      return;
    }
    empty_ = false;
    // ExprTk-style ^ -> ** (Python: src.replace("^", "**"))
    for (char ch : src) {
      if (ch == '^') {
        py_src_.push_back('*');
        py_src_.push_back('*');
      } else {
        py_src_.push_back(ch);
      }
    }
  }

  bool is_empty() const { return empty_; }
  const std::string& src() const { return src_; }

  double operator()(double t) const {
    if (empty_) return 0.0;
    Parser p(py_src_, t);
    double v = p.parse_expr();
    p.skip_ws();
    if (!p.at_end()) throw std::runtime_error("AnalyticExpr: trailing characters in '" + src_ + "'");
    return v;
  }

 private:
  std::string src_;
  std::string py_src_;
  bool empty_;

  struct Parser {
    const std::string& s;
    std::size_t i = 0;
    double tval;
    Parser(const std::string& str, double t) : s(str), tval(t) {}

    bool at_end() const { return i >= s.size(); }
    void skip_ws() {
      while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }

    double parse_expr() {
      double v = parse_term();
      for (;;) {
        skip_ws();
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
          char op = s[i++];
          double rhs = parse_term();
          v = (op == '+') ? v + rhs : v - rhs;
        } else {
          break;
        }
      }
      return v;
    }

    double parse_term() {
      double v = parse_factor();
      for (;;) {
        skip_ws();
        if (i < s.size() && (s[i] == '*' || s[i] == '/')) {
          // do not consume '**' here — that belongs to parse_power
          if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*') break;
          char op = s[i++];
          double rhs = parse_factor();
          v = (op == '*') ? v * rhs : v / rhs;
        } else {
          break;
        }
      }
      return v;
    }

    // unary +/- (Python: looser than ** on the right operand)
    double parse_factor() {
      skip_ws();
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        char op = s[i++];
        double v = parse_factor();
        return (op == '-') ? -v : v;
      }
      return parse_power();
    }

    // power: atom ['**' factor]  (right-associative via factor recursion)
    double parse_power() {
      double base = parse_atom();
      skip_ws();
      if (i + 1 < s.size() && s[i] == '*' && s[i + 1] == '*') {
        i += 2;
        double exp = parse_factor();
        return std::pow(base, exp);
      }
      return base;
    }

    double parse_atom() {
      skip_ws();
      if (at_end()) throw std::runtime_error("AnalyticExpr: unexpected end of expression");
      char c = s[i];
      if (c == '(') {
        ++i;
        double v = parse_expr();
        skip_ws();
        if (i >= s.size() || s[i] != ')') throw std::runtime_error("AnalyticExpr: missing ')'");
        ++i;
        return v;
      }
      if (is_num_start(c)) return parse_number();
      if (is_name_start(c)) return parse_name();
      throw std::runtime_error(std::string("AnalyticExpr: unexpected char '") + c + "'");
    }

    static bool is_num_start(char c) { return (c >= '0' && c <= '9') || c == '.'; }
    static bool is_name_start(char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }
    static bool is_name_char(char c) { return is_name_start(c) || (c >= '0' && c <= '9'); }

    double parse_number() {
      std::size_t start = i;
      while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.')) ++i;
      // exponent part: e/E [+/-] digits
      if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        std::size_t save = i;
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        if (i < s.size() && (s[i] >= '0' && s[i] <= '9')) {
          while (i < s.size() && (s[i] >= '0' && s[i] <= '9')) ++i;
        } else {
          i = save;  // not an exponent (e.g. a bare 'e' constant follows multiplication)
        }
      }
      return std::stod(s.substr(start, i - start));
    }

    double parse_name() {
      std::size_t start = i;
      while (i < s.size() && is_name_char(s[i])) ++i;
      std::string name = s.substr(start, i - start);
      skip_ws();
      if (i < s.size() && s[i] == '(') {
        // function call
        ++i;
        std::vector<double> args;
        skip_ws();
        if (i < s.size() && s[i] == ')') {
          ++i;
        } else {
          for (;;) {
            double a = parse_expr();
            args.push_back(a);
            skip_ws();
            if (i < s.size() && s[i] == ',') {
              ++i;
              continue;
            }
            break;
          }
          skip_ws();
          if (i >= s.size() || s[i] != ')')
            throw std::runtime_error("AnalyticExpr: missing ')' after args of " + name);
          ++i;
        }
        return call_func(name, args);
      }
      return lookup_const(name);
    }

    double lookup_const(const std::string& name) const {
      // Match Python's float(np.pi) / float(np.e) bit-for-bit (M_PI/M_E not portable on MinGW).
      static constexpr double kPi = 3.141592653589793;
      static constexpr double kE = 2.718281828459045;
      if (name == "t") return tval;
      if (name == "pi") return kPi;
      if (name == "e") return kE;
      throw std::runtime_error("AnalyticExpr: unknown name '" + name + "'");
    }

    static double call_func(const std::string& name, const std::vector<double>& a) {
      auto need = [&](std::size_t n) {
        if (a.size() != n)
          throw std::runtime_error("AnalyticExpr: wrong arg count for " + name);
      };
      if (name == "sin") { need(1); return std::sin(a[0]); }
      if (name == "cos") { need(1); return std::cos(a[0]); }
      if (name == "tan") { need(1); return std::tan(a[0]); }
      if (name == "asin") { need(1); return std::asin(a[0]); }
      if (name == "acos") { need(1); return std::acos(a[0]); }
      if (name == "atan") { need(1); return std::atan(a[0]); }
      if (name == "atan2") { need(2); return std::atan2(a[0], a[1]); }
      if (name == "sqrt") { need(1); return std::sqrt(a[0]); }
      if (name == "exp") { need(1); return std::exp(a[0]); }
      if (name == "log") { need(1); return std::log(a[0]); }       // np.log == natural log
      if (name == "pow") { need(2); return std::pow(a[0], a[1]); }
      if (name == "abs") { need(1); return std::fabs(a[0]); }
      throw std::runtime_error("AnalyticExpr: unknown function '" + name + "'");
    }
  };
};

// ===========================================================================
// DynTraj — dynamic obstacle / agent trajectory.
//   mode "Piecewise": evaluate via PieceWisePol.
//   mode "Analytic":  evaluate traj_x/y/z via AnalyticExpr; velocity uses
//                     traj_vx/vy/vz if all present, else central difference;
//                     accel always central-differences velocity. Exactly as Python.
// ===========================================================================
class DynTraj {
 public:
  std::string mode = "Analytic";  // "Piecewise" | "Analytic"
  PieceWisePol pwp;

  std::string traj_x, traj_y, traj_z;
  std::string traj_vx, traj_vy, traj_vz;

  bool analytic_compiled = false;

  // metadata fields (defaults mirror types.py)
  Eigen::Vector3d ekf_cov_p = Eigen::Vector3d::Zero();
  Eigen::Vector3d ekf_cov_q = Eigen::Vector3d::Zero();
  Eigen::Vector3d poly_cov = Eigen::Vector3d::Zero();
  std::vector<Eigen::Matrix<double, 3, 4>> control_points;  // 3x4 each
  Eigen::Vector3d bbox = Eigen::Vector3d(0.5, 0.5, 0.5);
  Eigen::Vector3d goal = Eigen::Vector3d::Zero();
  Eigen::Vector3d current_pos = Eigen::Vector3d::Zero();
  bool is_agent = false;
  int id = -1;
  double time_received = 0.0;
  double tracking_utility = 0.0;
  double communication_delay = 0.0;

  void set_piecewise(const PieceWisePol& p) {
    mode = "Piecewise";
    pwp = p;
  }

  // Compile analytic strings (lazy compile mirrors Python). Returns false on parse error.
  bool compile_analytic() {
    try {
      expr_x_ = AnalyticExpr(traj_x);
      expr_y_ = AnalyticExpr(traj_y);
      expr_z_ = AnalyticExpr(traj_z);
      expr_vx_ = traj_vx.empty() ? AnalyticExpr() : AnalyticExpr(traj_vx);
      expr_vy_ = traj_vy.empty() ? AnalyticExpr() : AnalyticExpr(traj_vy);
      expr_vz_ = traj_vz.empty() ? AnalyticExpr() : AnalyticExpr(traj_vz);
      have_v_ = !traj_vx.empty() && !traj_vy.empty() && !traj_vz.empty();
      analytic_compiled = true;
      return true;
    } catch (const std::exception&) {
      analytic_compiled = false;
      return false;
    }
  }

  Eigen::Vector3d eval(double t) {
    if (mode == "Piecewise") return pwp.eval(t);
    if (!analytic_compiled && !traj_x.empty()) compile_analytic();
    if (!analytic_compiled) return Eigen::Vector3d::Zero();
    return Eigen::Vector3d(expr_x_(t), expr_y_(t), expr_z_(t));
  }

  Eigen::Vector3d velocity(double t) {
    if (mode == "Piecewise") return pwp.velocity(t);
    if (!analytic_compiled && !traj_x.empty()) compile_analytic();
    if (have_v_) return Eigen::Vector3d(expr_vx_(t), expr_vy_(t), expr_vz_(t));
    // Numerical central-difference fallback: (p(t+dt)-p(t-dt))/(2 dt)
    const double dt = 1e-3;
    return (eval(t + dt) - eval(t - dt)) / (2.0 * dt);
  }

  Eigen::Vector3d accel(double t) {
    if (mode == "Piecewise") return pwp.acceleration(t);
    const double dt = 1e-3;
    return (velocity(t + dt) - velocity(t - dt)) / (2.0 * dt);
  }

 private:
  AnalyticExpr expr_x_, expr_y_, expr_z_;
  AnalyticExpr expr_vx_, expr_vy_, expr_vz_;
  bool have_v_ = false;
};

// ===========================================================================
// RobotState — full self-state of the drone.
// ===========================================================================
class RobotState {
 public:
  double t = 0.0;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d jerk = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double dyaw = 0.0;
  bool use_tracking_yaw = false;

  void set_zero() {
    pos.setZero();
    vel.setZero();
    accel.setZero();
    jerk.setZero();
    yaw = 0.0;
    dyaw = 0.0;
  }

  RobotState clone() const { return *this; }

  void set_pos(const Eigen::Vector3d& v) { pos = v; }
  void set_pos(double x, double y, double z) { pos = Eigen::Vector3d(x, y, z); }
  void set_vel(const Eigen::Vector3d& v) { vel = v; }
  void set_vel(double x, double y, double z) { vel = Eigen::Vector3d(x, y, z); }
  void set_accel(const Eigen::Vector3d& v) { accel = v; }
  void set_accel(double x, double y, double z) { accel = Eigen::Vector3d(x, y, z); }
  void set_jerk(const Eigen::Vector3d& v) { jerk = v; }
  void set_jerk(double x, double y, double z) { jerk = Eigen::Vector3d(x, y, z); }
};

// ===========================================================================
// Parameters — all tunable planner parameters; defaults copied verbatim from
// types.py (the dataclass field defaults).
// ===========================================================================
struct Parameters {
  // Sim environment
  std::string sim_env = "gazebo";
  bool use_global_pc = true;

  // Vehicle
  std::string vehicle_type = "uav";
  bool provide_goal_in_global_frame = false;
  bool state_already_in_global_frame = false;
  bool use_hardware = false;

  // Flight
  std::string flight_mode = "terminal_goal";
  int visual_level = 2;

  // Global planner
  std::string global_planner = "astar_heat";
  bool global_planner_verbose = false;
  double global_planner_heuristic_weight = 2.0;
  double factor_hgp = 1.0;
  double inflation_hgp = 0.45;
  double x_min = -200.0;
  double x_max = 200.0;
  double y_min = -200.0;
  double y_max = 200.0;
  double z_min = 0.5;
  double z_max = 6.0;
  double drone_radius = 0.2;
  int hgp_timeout_duration_ms = 1000;
  int max_num_expansion = 100000;
  bool use_free_start = true;
  double free_start_factor = 2.0;
  bool use_free_goal = false;
  double free_goal_factor = 2.0;

  // LoS post processing
  int los_cells = 0;
  double min_len = 1.0;
  double min_turn = 0.0;

  // Path push visualization
  bool use_state_update = true;
  bool use_random_color_for_global_path = false;
  bool use_path_push_for_visualization = false;

  // Local solver selection
  std::string local_solver = "minco";

  // Decomposition
  std::string environment_assumption = "dynamic";
  std::vector<double> sfc_size = {3.0, 3.0, 3.0};
  double min_dist_from_agent_to_traj = 10.0;
  bool use_shrinked_box = false;
  double shrinked_box_size = 0.0;

  // Map
  double map_buffer = 1.0;
  double center_shift_factor = 0.5;
  double initial_wdx = 15.0;
  double initial_wdy = 15.0;
  double initial_wdz = 4.0;
  double min_wdx = 15.0;
  double min_wdy = 15.0;
  double min_wdz = 4.0;
  double res = 0.3;

  // Communication delay
  bool use_comm_delay_inflation = true;
  double comm_delay_inflation_alpha = 0.2;
  double comm_delay_inflation_max = 0.1;
  double comm_delay_filter_alpha = 0.9;

  // Sim
  double depth_camera_depth_max = 10.0;
  double fov_visual_depth = 10.0;
  double fov_visual_x_deg = 76.0;
  double fov_visual_y_deg = 47.0;

  // Segments
  double max_dist_vertexes = 1.0;
  double w_unknown = 0.0;
  double w_align = 0.0;
  double decay_len_cells = 100.0;
  double w_side = 0.0;
  double heat_weight = 10.0;

  // Heat
  bool use_heat_map = true;
  bool dynamic_heat_enabled = true;
  bool dynamic_as_occupied_current = true;
  bool dynamic_as_occupied_future = false;
  bool use_only_curr_pos_for_dynamic_obst = false;
  double heat_alpha0 = 0.2;
  double heat_alpha1 = 1.0;
  int heat_p = 2;
  int heat_q = 2;
  double heat_tau_ratio = 0.5;
  double heat_gamma = 0.0;
  double heat_Hmax = 2.0;
  double dyn_base_inflation_m = 0.1;
  double dyn_heat_tube_radius_m = 0.5;
  int heat_num_samples = 15;

  bool static_heat_enabled = true;
  double static_heat_alpha = 1.0;
  int static_heat_p = 2;
  double static_heat_Hmax = 5.0;
  double static_heat_rmax_m = 1.0;
  double static_heat_default_radius_m = 0.5;
  bool static_heat_boundary_only = true;
  bool static_heat_apply_on_unknown = false;
  bool static_heat_exclude_dynamic = true;

  // Soft-cost
  bool use_soft_cost_obstacles = true;
  double obstacle_soft_cost = 5.0;

  // Optimization
  double horizon = 15.0;
  double dc = 0.01;
  std::string dynamic_constraint_type = "Linf";
  double v_max = 10.0;
  double a_max = 20.0;
  double j_max = 100.0;
  std::vector<double> drone_bbox = {0.2, 0.2, 0.2};
  double goal_radius = 0.5;
  double goal_seen_radius = 2.0;

  // SANDO
  int num_P = 3;
  int num_N = 5;
  bool use_dynamic_factor = true;
  double dynamic_factor_k_radius = 0.4;
  double dynamic_factor_initial_mean = 1.5;
  double factor_initial = 1.0;
  double factor_final = 2.5;
  double factor_constant_step_size = 0.1;
  double obst_max_vel = 0.5;
  double obst_position_error = 0.0;
  bool inflate_unknown_boundary = true;
  double max_gurobi_comp_time_sec = 1.0;
  double jerk_smooth_weight = 10.0;
  bool using_variable_elimination = true;

  // Dynamic obstacles
  double traj_lifetime = 7.0;

  // Dynamic k_value
  int num_replanning_before_adapt = 10;
  int default_k_value = 50;
  double alpha_k_value_filtering = 0.9;
  double k_value_factor = 5.0;

  // Yaw
  double alpha_filter_dyaw = 0.8;
  double w_max = 1.0;
  double w_max_yawing = 0.5;
  bool skip_initial_yawing = false;
  int yaw_spinning_threshold = 10000;
  double yaw_spinning_dyaw = 1.0;

  // Sim env / goal
  bool force_goal_z = true;
  double default_goal_z = 2.0;

  // Debug
  bool debug_verbose = false;

  // Hover avoidance
  bool ignore_other_trajs = false;
  bool hover_avoidance_enabled = false;
  bool hover_avoidance_2d = true;
  double hover_avoidance_d_trigger = 4.0;
  double hover_avoidance_h = 3.0;
  double hover_avoidance_min_repulsion_norm = 0.01;
};

}  // namespace sando
