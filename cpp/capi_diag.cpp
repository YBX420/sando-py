// capi_diag — call the planner directly (no CAPI silent catch) and print exception what()
#include <cstdio>
#include <vector>
#include "sando_cpp/types.hpp"
#include "sando_cpp/planner.hpp"
using namespace sando;

#define TRY(label, code) \
  try { code; printf("%-26s OK\n", label); } \
  catch (const std::exception& e) { printf("%-26s THREW std::exception: %s\n", label, e.what()); } \
  catch (...) { printf("%-26s THREW (non-std)\n", label); }

int main() {
  Parameters par;
  par.replan_dt = 0.1;
  SANDO s(par);

  RobotState st; st.pos = Eigen::Vector3d(0,0,2); st.vel = Eigen::Vector3d::Zero(); st.accel = Eigen::Vector3d::Zero();
  TRY("update_state", s.update_state(st));

  RobotState G; G.pos = Eigen::Vector3d(20,0,2);
  TRY("set_terminal_goal", s.set_terminal_goal(G));

  std::vector<Eigen::Vector3d> empty;
  TRY("update_occupancy(empty)", s.update_occupancy_map_ptr(empty));

  std::pair<bool,bool> r{false,false};
  TRY("replan", r = s.replan(0.0, 0.0));
  printf("replan -> ok=%d att=%d  drone_status=%d\n", (int)r.first, (int)r.second, (int)s.get_drone_status());

  // also try a non-empty map
  std::vector<Eigen::Vector3d> wall;
  for (double y=-1; y<=1.001; y+=0.25) for (double z=1.5; z<=2.5001; z+=0.25) wall.push_back(Eigen::Vector3d(8,y,z));
  SANDO s2(par);
  TRY("s2 update_state", s2.update_state(st));
  TRY("s2 set_terminal_goal", s2.set_terminal_goal(G));
  TRY("s2 update_occupancy(wall)", s2.update_occupancy_map_ptr(wall));
  std::pair<bool,bool> r2{false,false};
  TRY("s2 replan", r2 = s2.replan(0.0, 0.0));
  printf("s2 replan -> ok=%d att=%d\n", (int)r2.first, (int)r2.second);
  return 0;
}
