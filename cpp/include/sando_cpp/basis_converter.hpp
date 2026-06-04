// BasisConverter — faithful C++ port of sando_py/types.py:BasisConverter.
// B-spline <-> MINVO / Bezier basis conversion matrices (consumed by the Gurobi baseline + viz).
// Pure matrix math (no Gurobi). Golden-verified against Python.
#pragma once
#include <Eigen/Dense>
#include <vector>

namespace sando {

inline Eigen::Matrix4d _A_pos_mv_rest() {
  Eigen::Matrix4d M;
  M << -3.4416308968564117698463178385282, 6.9895481477801393310755884158425, -4.4622887507045296828778191411402, 0.91437149978080234369315348885721,
       6.6792587327074839365081970754545, -11.845989901556746914934592496138, 5.2523596690684613008670567069203, 0.0,
       -6.6792587327074839365081970754545, 8.1917862965657040064115790301003, -1.5981560640774179482548333908198, 0.085628500219197656306846511142794,
       3.4416308968564117698463178385282, -3.3353445427890959784633650997421, 0.80808514571348655231020075007109, -8.4567769453869345852581318467855e-18;
  return M;
}
inline Eigen::Matrix3d _A_vel_mv_rest() {
  Eigen::Matrix3d M;
  M << 1.4999999992328318931811281800037, -2.3660254034601951866889635311964, 0.9330127021136816189983420599674,
       -2.9999999984656637863622563600074, 2.9999999984656637863622563600074, 0.0,
       1.4999999992328318931811281800037, -0.6339745950054685996732928288111, 0.066987297886318325490506708774774;
  return M;
}
inline Eigen::Matrix2d _A_accel_mv_rest() { Eigen::Matrix2d M; M << -1.0, 1.0, 1.0, 0.0; return M; }
inline Eigen::Matrix4d _A_pos_be_rest() {
  Eigen::Matrix4d M; M << -1,3,-3,1, 3,-6,3,0, -3,3,0,0, 1,0,0,0; return M;
}
inline Eigen::Matrix4d _A_pos_bs_seg0() {
  Eigen::Matrix4d M; M << -1.0,3.0,-3.0,1.0, 1.75,-4.5,3.0,0.0, -0.9167,1.5,0.0,0.0, 0.1667,0.0,0.0,0.0; return M;
}
inline Eigen::Matrix4d _A_pos_bs_seg1() {
  Eigen::Matrix4d M; M << -0.25,0.75,-0.75,0.25, 0.5833,-1.25,0.25,0.5833, -0.5,0.5,0.5,0.1667, 0.1667,0.0,0.0,0.0; return M;
}
inline Eigen::Matrix4d _A_pos_bs_rest() {
  Eigen::Matrix4d M; M << -0.1667,0.5,-0.5,0.1667, 0.5,-1.0,0.0,0.6667, -0.5,0.5,0.5,0.1667, 0.1667,0.0,0.0,0.0; return M;
}
inline Eigen::Matrix4d _A_pos_bs_seg_last2() {
  Eigen::Matrix4d M; M << -0.1667,0.5,-0.5,0.1667, 0.5,-1.0,0.0,0.6667, -0.5833,0.5,0.5,0.1667, 0.25,0.0,0.0,0.0; return M;
}
inline Eigen::Matrix4d _A_pos_bs_seg_last() {
  Eigen::Matrix4d M; M << -0.1667,0.5,-0.5,0.1667, 0.9167,-1.25,-0.25,0.5833, -1.75,0.75,0.75,0.25, 1.0,0.0,0.0,0.0; return M;
}

class BasisConverter {
 public:
  Eigen::Matrix4d A_pos_mv_rest, A_pos_mv_rest_inv;
  Eigen::Matrix3d A_vel_mv_rest, A_vel_mv_rest_inv;
  Eigen::Matrix2d A_accel_mv_rest, A_accel_mv_rest_inv;
  Eigen::Matrix4d A_pos_be_rest;
  Eigen::Matrix4d A_pos_bs_seg0, A_pos_bs_seg1, A_pos_bs_rest, A_pos_bs_seg_last2, A_pos_bs_seg_last;

  BasisConverter() {
    A_pos_mv_rest = _A_pos_mv_rest(); A_pos_mv_rest_inv = A_pos_mv_rest.inverse();
    A_vel_mv_rest = _A_vel_mv_rest(); A_vel_mv_rest_inv = A_vel_mv_rest.inverse();
    A_accel_mv_rest = _A_accel_mv_rest(); A_accel_mv_rest_inv = A_accel_mv_rest.inverse();
    A_pos_be_rest = _A_pos_be_rest();
    A_pos_bs_seg0 = _A_pos_bs_seg0(); A_pos_bs_seg1 = _A_pos_bs_seg1();
    A_pos_bs_rest = _A_pos_bs_rest();
    A_pos_bs_seg_last2 = _A_pos_bs_seg_last2(); A_pos_bs_seg_last = _A_pos_bs_seg_last();
  }

  std::vector<Eigen::Matrix4d> get_A_bspline(int num_pol) const {
    if (num_pol < 4) return std::vector<Eigen::Matrix4d>(num_pol, A_pos_bs_rest);
    std::vector<Eigen::Matrix4d> out = {A_pos_bs_seg0, A_pos_bs_seg1};
    for (int i = 0; i < num_pol - 4; ++i) out.push_back(A_pos_bs_rest);
    out.push_back(A_pos_bs_seg_last2);
    out.push_back(A_pos_bs_seg_last);
    return out;
  }
  std::vector<Eigen::Matrix4d> get_A_minvo(int num_pol) const { return std::vector<Eigen::Matrix4d>(num_pol, A_pos_mv_rest); }
  std::vector<Eigen::Matrix4d> get_A_bezier(int num_pol) const { return std::vector<Eigen::Matrix4d>(num_pol, A_pos_be_rest); }

  std::vector<Eigen::Matrix4d> get_minvo_pos_converters(int num_pol) const {
    auto As = get_A_bspline(num_pol);
    std::vector<Eigen::Matrix4d> out;
    for (auto& A : As) out.push_back(A_pos_mv_rest_inv * A);
    return out;
  }
  std::vector<Eigen::Matrix4d> get_bezier_pos_converters(int num_pol) const {
    Eigen::Matrix4d Ainv_be = A_pos_be_rest.inverse();
    auto As = get_A_bspline(num_pol);
    std::vector<Eigen::Matrix4d> out;
    for (auto& A : As) out.push_back(Ainv_be * A);
    return out;
  }
  std::vector<Eigen::Matrix3d> get_minvo_vel_converters(int num_pol) const { return std::vector<Eigen::Matrix3d>(num_pol, A_vel_mv_rest_inv); }
  std::vector<Eigen::Matrix2d> get_minvo_accel_converters(int num_pol) const { return std::vector<Eigen::Matrix2d>(num_pol, A_accel_mv_rest_inv); }
};

}  // namespace sando
