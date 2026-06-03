// Golden verification: C++ planner.hpp (SANDO orchestrator, MINCO path) must
// reproduce the Python sando_py.planner.SANDO deterministic methods, and the
// end-to-end replan wiring.
//
// Reads cpp/golden/planner_cases.txt (gen_planner_golden.py). Case kinds:
//   COMPUTE_G  : compute_G                        -> EXACT (abs tol)
//   FINDA      : find_A_and_Atime (k_value logic) -> EXACT
//   SAFESUB    : find_safe_sub_goal               -> EXACT
//   OBSTTIME   : _compute_obst_pos_and_traj_max_time -> EXACT (Th + snapshot + preds)
//   OBSTSNAP   : _obstacles_from_snapshot         -> EXACT (per-obstacle geometry)
//   MJ2PWP     : _minjerk_to_pwp eval-equivalence -> EXACT
//   APPEND     : append_to_plan queue splice      -> EXACT
//   NEEDREPLAN : need_replan state machine         -> EXACT
//   GLOBALPATH : generate_global_path (solve_hgp deterministic) -> EXACT path
//   REPLAN     : full replan end-to-end           -> WIRING (ATTEMPTED) EXACT;
//                local trajectory inherits plan_minco's optimiser non-determinism
//                (LBFGSpp != scipy on nonconvex ALM) -> sampled-position error and
//                PLANNED_OK match are REPORTED honestly (NOT gated).
//
// 不许欺骗、必须还原: the deterministic methods must match exactly or FAIL.
//
// Build (standalone):
//   g++ -std=c++17 -O2 -I cpp/third_party/eigen -I cpp/third_party -I cpp/include \
//       cpp/test/test_planner.cpp -o /tmp/test_planner && \
//   /tmp/test_planner cpp/golden/planner_cases.txt
#include "sando_cpp/planner.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <cmath>
#include <limits>
#include <Eigen/Dense>

using std::string;
using std::vector;

static vector<double> nums(const string& rest) {
  vector<double> v; std::istringstream is(rest); double x;
  while (is >> x) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "golden/planner_cases.txt";
  std::ifstream f(path);
  if (!f) { std::printf("CANNOT OPEN %s\n", path); return 2; }

  string line;
  std::map<string, vector<double>> num;
  std::map<string, string> str;
  // traj buckets keyed by prefix index ("TRAJ<i>_...")
  std::map<int, std::map<string, string>> traj_str;
  std::map<int, std::map<string, vector<double>>> traj_num;
  // obstsnap input buckets ("IN<i>_..." / "OUT<i>_...")
  std::map<int, std::map<string, string>> in_str, out_str;
  std::map<int, std::map<string, vector<double>>> in_num, out_num;
  // predicted samples ("PRED<i>")
  std::map<int, vector<double>> pred;

  int ncase = 0, nfail = 0;
  double global_max_err = 0.0;
  const double TOL = 1e-7;

  // REPLAN aggregate stats (informational)
  int replan_n = 0, replan_attempt_match = 0, replan_plannedok_match = 0;
  double replan_pos_err_max = 0.0;
  string replan_worst;

  auto build_traj = [&](int i) -> sando::DynTraj {
    sando::DynTraj t;
    auto& sm = traj_str[i];
    auto& nm = traj_num[i];
    t.id = (int)nm["ID"][0];
    t.mode = sm.count("MODE") ? sm["MODE"] : "Analytic";
    t.traj_x = sm.count("TX") ? sm["TX"] : "";
    t.traj_y = sm.count("TY") ? sm["TY"] : "";
    t.traj_z = sm.count("TZ") ? sm["TZ"] : "";
    t.traj_vx = (sm.count("TVX") && sm["TVX"] != "_") ? sm["TVX"] : "";
    t.traj_vy = (sm.count("TVY") && sm["TVY"] != "_") ? sm["TVY"] : "";
    t.traj_vz = (sm.count("TVZ") && sm["TVZ"] != "_") ? sm["TVZ"] : "";
    if (nm.count("BBOX")) t.bbox = Eigen::Vector3d(nm["BBOX"][0], nm["BBOX"][1], nm["BBOX"][2]);
    if (nm.count("TIME_RECEIVED")) t.time_received = nm["TIME_RECEIVED"][0];
    return t;
  };

  auto v3 = [&](const char* k) {
    return Eigen::Vector3d(num[k][0], num[k][1], num[k][2]);
  };

  auto process = [&]() {
    const string kind = str["KIND"];
    const string name = str.count("NAME") ? str["NAME"] : "?";
    double e = 0.0;

    if (kind == "COMPUTE_G") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      par.horizon = num["HORIZON"][0];
      sando::SANDO s(par);
      sando::RobotState A, Gt;
      A.pos = v3("A_POS"); Gt.pos = v3("GTERM_POS");
      s.compute_G(A, Gt, num["HORIZON"][0]);
      sando::RobotState G = s.get_G();
      for (int j = 0; j < 3; ++j) e = std::max(e, std::fabs(G.pos(j) - num["G_POS"][j]));
      e = std::max(e, std::fabs(G.yaw - num["G_YAW"][0]));

    } else if (kind == "FINDA") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      sando::SANDO s(par);
      int nplan = (int)num["NPLAN"][0];
      s.plan.clear();
      for (int i = 0; i < nplan; ++i) {
        sando::RobotState st;
        st.pos = Eigen::Vector3d(num["PLAN"][i * 3], num["PLAN"][i * 3 + 1], num["PLAN"][i * 3 + 2]);
        s.plan.push_back(st);
      }
      s.use_adapt_k_value = (num["USE_ADAPT"][0] != 0.0);
      s.num_replanning = (int)num["NUM_REPLANNING"][0];
      s.est_comp_time = num["EST_COMP"][0];
      auto r = s.find_A_and_Atime(num["CURRENT_TIME"][0], num["LAST_COMP"][0]);
      e = std::max(e, std::fabs((r.ok ? 1.0 : 0.0) - num["OK"][0]));
      for (int j = 0; j < 3; ++j) e = std::max(e, std::fabs(r.A.pos(j) - num["A_POS"][j]));
      e = std::max(e, std::fabs(r.A_time - num["A_TIME"][0]));
      e = std::max(e, std::fabs((double)s.k_value - num["K_VALUE"][0]));

    } else if (kind == "SAFESUB") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      par.drone_radius = num["DRONE_RADIUS"][0];
      par.obst_max_vel = num["OBST_MAX_VEL"][0];
      sando::SANDO s(par);
      s.traj_max_time = num["TRAJ_MAX_TIME"][0];
      int nunk = (int)num["NUNK"][0];
      s.pclptr_unk.clear();
      for (int i = 0; i < nunk; ++i)
        s.pclptr_unk.push_back(Eigen::Vector3d(num["UNK"][i * 3], num["UNK"][i * 3 + 1], num["UNK"][i * 3 + 2]));
      int npath = (int)num["NPATH"][0];
      vector<Eigen::Vector3d> pathv;
      for (int i = 0; i < npath; ++i)
        pathv.push_back(Eigen::Vector3d(num["PATH"][i * 3], num["PATH"][i * 3 + 1], num["PATH"][i * 3 + 2]));
      auto out = s.find_safe_sub_goal(pathv);
      int nout = (int)num["NOUT"][0];
      e = std::max(e, std::fabs((double)out.size() - nout));
      int nchk = std::min((int)out.size(), nout);
      for (int i = 0; i < nchk; ++i)
        for (int j = 0; j < 3; ++j)
          e = std::max(e, std::fabs(out[i](j) - num["OUT"][i * 3 + j]));

    } else if (kind == "OBSTTIME") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      par.horizon = num["HORIZON"][0];
      sando::SANDO s(par);
      s.state.pos = v3("STATE_POS");
      s.worst_traj_time = num["WORST_TRAJ_TIME"][0];
      if (num["MAP_INIT"][0] != 0.0) {
        s.map_size_initialized = true; s.map_seen = true;
        s.map_center = v3("MAP_CENTER");
        s.wdx = num["WDX"][0]; s.wdy = num["WDY"][0]; s.wdz = num["WDZ"][0];
      }
      int ntraj = (int)num["NTRAJ"][0];
      s.trajs.clear();
      for (int i = 0; i < ntraj; ++i) s.trajs.push_back(build_traj(i));
      vector<Eigen::Vector3d> opos, obbox;
      vector<vector<Eigen::Vector3d>> preds;
      vector<double> ptimes;
      double Th = s.compute_obst_pos_and_traj_max_time(opos, obbox, preds, ptimes,
                                                       num["CURRENT_TIME"][0]);
      e = std::max(e, std::fabs(Th - num["TH"][0]));
      int nsel = (int)num["NSEL"][0];
      e = std::max(e, std::fabs((double)opos.size() - nsel));
      int nchk = std::min((int)opos.size(), nsel);
      for (int i = 0; i < nchk; ++i)
        for (int j = 0; j < 3; ++j) {
          e = std::max(e, std::fabs(opos[i](j) - num["OPOS"][i * 3 + j]));
          e = std::max(e, std::fabs(obbox[i](j) - num["OBBOX"][i * 3 + j]));
          e = std::max(e, std::fabs(s.obst_vel[i](j) - num["OVEL"][i * 3 + j]));
          e = std::max(e, std::fabs(s.obst_accel[i](j) - num["OACCEL"][i * 3 + j]));
        }
      // class tags
      {
        std::istringstream cs(str.count("OCLASS") ? str["OCLASS"] : "");
        vector<string> gclass; string tok;
        while (cs >> tok) gclass.push_back(tok);
        for (int i = 0; i < nchk; ++i)
          if (i < (int)gclass.size() && i < (int)s.obst_class.size())
            if (gclass[i] != s.obst_class[i]) e = std::max(e, 1.0);
      }
      int nptimes = (int)num["NPTIMES"][0];
      e = std::max(e, std::fabs((double)ptimes.size() - nptimes));
      int ntchk = std::min((int)ptimes.size(), nptimes);
      for (int i = 0; i < ntchk; ++i)
        e = std::max(e, std::fabs(ptimes[i] - num["PTIMES"][i]));
      int npred = (int)num["NPRED"][0];
      e = std::max(e, std::fabs((double)preds.size() - npred));
      int npchk = std::min((int)preds.size(), npred);
      for (int i = 0; i < npchk; ++i) {
        const auto& pk = preds[i];
        const auto& gpk = pred[i];
        for (int s2 = 0; s2 < (int)pk.size() && s2 * 3 + 2 < (int)gpk.size(); ++s2)
          for (int j = 0; j < 3; ++j)
            e = std::max(e, std::fabs(pk[s2](j) - gpk[s2 * 3 + j]));
      }

    } else if (kind == "OBSTSNAP") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      sando::SANDO s(par);
      int n = (int)num["N"][0];
      vector<Eigen::Vector3d> opos, obbox, ovel, oaccel;
      vector<string> oclass;
      for (int i = 0; i < n; ++i) {
        opos.push_back(Eigen::Vector3d(in_num[i]["POS"][0], in_num[i]["POS"][1], in_num[i]["POS"][2]));
        obbox.push_back(Eigen::Vector3d(in_num[i]["BBOX"][0], in_num[i]["BBOX"][1], in_num[i]["BBOX"][2]));
        ovel.push_back(Eigen::Vector3d(in_num[i]["VEL"][0], in_num[i]["VEL"][1], in_num[i]["VEL"][2]));
        oaccel.push_back(Eigen::Vector3d(in_num[i]["ACCEL"][0], in_num[i]["ACCEL"][1], in_num[i]["ACCEL"][2]));
        oclass.push_back(in_str[i].count("CLASS") && in_str[i]["CLASS"] != "_" ? in_str[i]["CLASS"] : "");
      }
      // Mirror Python: missing class entries -> list shorter than n (handled inside).
      // Build a trimmed class list that drops empty tags at the tail (Python passes []).
      vector<string> class_in;
      for (auto& c : oclass) if (!c.empty()) class_in.push_back(c);
      std::vector<std::shared_ptr<sando::Obstacle>> owned;
      vector<const sando::Obstacle*> obstacles;
      std::map<string, sando::AvoidParams> cfg;
      s.obstacles_from_snapshot(opos, obbox, class_in, ovel, oaccel, owned, obstacles, cfg);
      int nout = (int)num["NOUT"][0];
      e = std::max(e, std::fabs((double)obstacles.size() - nout));
      int nchk = std::min((int)obstacles.size(), nout);
      for (int i = 0; i < nchk; ++i) {
        const string kk = out_str[i]["KIND"];
        if (kk == "sphere") {
          auto sp = dynamic_cast<const sando::SphereObstacle*>(obstacles[i]);
          if (!sp) { e = std::max(e, 1.0); continue; }
          for (int j = 0; j < 3; ++j) {
            e = std::max(e, std::fabs(sp->centre0(j) - out_num[i]["C0"][j]));
            e = std::max(e, std::fabs(sp->vel(j) - out_num[i]["VEL"][j]));
            e = std::max(e, std::fabs(sp->accel(j) - out_num[i]["ACC"][j]));
          }
          e = std::max(e, std::fabs(sp->radius - out_num[i]["R"][0]));
          if (sp->class_name != out_str[i]["CLASS"]) e = std::max(e, 1.0);
        } else {
          auto bb = dynamic_cast<const sando::AABBObstacle*>(obstacles[i]);
          if (!bb) { e = std::max(e, 1.0); continue; }
          for (int j = 0; j < 3; ++j) {
            e = std::max(e, std::fabs(bb->lo(j) - out_num[i]["LO"][j]));
            e = std::max(e, std::fabs(bb->hi(j) - out_num[i]["HI"][j]));
          }
          if (bb->class_name != out_str[i]["CLASS"]) e = std::max(e, 1.0);
        }
      }

    } else if (kind == "MJ2PWP") {
      int M = (int)num["M"][0];
      Eigen::MatrixXd wp(M + 1, 3);
      for (int i = 0; i < M + 1; ++i) for (int j = 0; j < 3; ++j) wp(i, j) = num["WP"][i * 3 + j];
      Eigen::VectorXd T = Eigen::Map<Eigen::VectorXd>(num["T"].data(), M);
      sando::MinjerkTraj tr(wp, T, v3("V0"), v3("A0"), v3("VF"), v3("AF"));
      sando::QuinticPieceWisePol pwp = sando::minjerk_to_pwp(tr, num["A_TIME"][0]);
      // boundary times
      int ntimes = (int)num["NTIMES"][0];
      e = std::max(e, std::fabs((double)pwp.times.size() - ntimes));
      for (int i = 0; i < (int)pwp.times.size() && i < ntimes; ++i)
        e = std::max(e, std::fabs(pwp.times[i] - num["TIMES"][i]));
      // coeffs
      for (int i = 0; i < M; ++i)
        for (int j = 0; j < 6; ++j) {
          e = std::max(e, std::fabs(pwp.qcoeff_x[i](j) - num["COEFF_X"][i * 6 + j]));
          e = std::max(e, std::fabs(pwp.qcoeff_y[i](j) - num["COEFF_Y"][i * 6 + j]));
          e = std::max(e, std::fabs(pwp.qcoeff_z[i](j) - num["COEFF_Z"][i * 6 + j]));
        }
      // sampled pos/vel/acc at absolute times
      int nsamp = (int)num["NSAMP"][0];
      for (int s2 = 0; s2 < nsamp; ++s2) {
        double t = num["SAMP_T"][s2];
        Eigen::Vector3d p = pwp.eval_q(t);
        Eigen::Vector3d vv = pwp.velocity_q(t);
        Eigen::Vector3d aa = pwp.acceleration_q(t);
        for (int j = 0; j < 3; ++j) {
          e = std::max(e, std::fabs(p(j) - num["PWP_POS"][s2 * 3 + j]));
          e = std::max(e, std::fabs(vv(j) - num["PWP_VEL"][s2 * 3 + j]));
          e = std::max(e, std::fabs(aa(j) - num["PWP_ACC"][s2 * 3 + j]));
        }
      }

    } else if (kind == "APPEND") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      sando::SANDO s(par);
      int nplan = (int)num["NPLAN"][0];
      s.plan.clear();
      for (int i = 0; i < nplan; ++i) {
        sando::RobotState st;
        st.pos = Eigen::Vector3d(num["PLAN"][i * 3], num["PLAN"][i * 3 + 1], num["PLAN"][i * 3 + 2]);
        s.plan.push_back(st);
      }
      int nset = (int)num["NSET"][0];
      s.goal_setpoints.clear();
      for (int i = 0; i < nset; ++i) {
        sando::RobotState st;
        st.pos = Eigen::Vector3d(num["SET"][i * 3], num["SET"][i * 3 + 1], num["SET"][i * 3 + 2]);
        s.goal_setpoints.push_back(st);
      }
      s.k_value = (int)num["K_VALUE_IN"][0];
      s.got_enough_replanning = true;
      bool ok = s.append_to_plan();
      e = std::max(e, std::fabs((ok ? 1.0 : 0.0) - num["OK"][0]));
      e = std::max(e, std::fabs((double)s.k_value - num["K_VALUE_OUT"][0]));
      int nfinal = (int)num["NFINAL"][0];
      e = std::max(e, std::fabs((double)s.plan.size() - nfinal));
      int idx = 0;
      for (auto& st : s.plan) {
        if (idx >= nfinal) break;
        for (int j = 0; j < 3; ++j)
          e = std::max(e, std::fabs(st.pos(j) - num["FINAL"][idx * 3 + j]));
        ++idx;
      }

    } else if (kind == "NEEDREPLAN") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      par.hover_avoidance_enabled = (num["HOVER_ENABLED"][0] != 0.0);
      par.goal_radius = num["GOAL_RADIUS"][0];
      par.goal_seen_radius = num["GOAL_SEEN_RADIUS"][0];
      sando::SANDO s(par);
      s.drone_status_ = (sando::DroneStatus)(int)num["STATUS_BEFORE"][0];
      sando::RobotState ls, gt, lps;
      ls.pos = v3("STATE_POS"); ls.vel = v3("STATE_VEL");
      gt.pos = v3("GTERM_POS"); lps.pos = v3("LAST_PLAN_POS");
      bool res = s.need_replan(ls, gt, lps);
      e = std::max(e, std::fabs((res ? 1.0 : 0.0) - num["RESULT"][0]));
      e = std::max(e, std::fabs((double)(int)s.drone_status_ - num["STATUS_AFTER"][0]));

    } else if (kind == "GLOBALPATH") {
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      sando::SANDO s(par);
      s.worst_traj_time = num["WORST_TRAJ_TIME"][0];
      sando::RobotState st0; st0.pos = v3("STATE_POS");
      s.update_state(st0);
      int nocc = (int)num["NOCC"][0];
      vector<Eigen::Vector3d> occ;
      for (int i = 0; i < nocc; ++i)
        occ.push_back(Eigen::Vector3d(num["OCC"][i * 3], num["OCC"][i * 3 + 1], num["OCC"][i * 3 + 2]));
      s.update_occupancy_map_ptr(occ);
      sando::RobotState gt; gt.pos = v3("GTERM_POS");
      s.set_terminal_goal(gt);
      s.change_drone_status(sando::DroneStatus::TRAVELING);
      vector<Eigen::Vector3d> gp;
      bool ok = s.generate_global_path(num["CURRENT_TIME"][0], 0.0, gp);
      e = std::max(e, std::fabs((ok ? 1.0 : 0.0) - num["OK"][0]));
      int ngp = (int)num["NGP"][0];
      e = std::max(e, std::fabs((double)gp.size() - ngp));
      int nchk = std::min((int)gp.size(), ngp);
      for (int i = 0; i < nchk; ++i)
        for (int j = 0; j < 3; ++j)
          e = std::max(e, std::fabs(gp[i](j) - num["GP"][i * 3 + j]));
      for (int j = 0; j < 3; ++j) {
        e = std::max(e, std::fabs(s.get_A().pos(j) - num["A_POS"][j]));
        e = std::max(e, std::fabs(s.get_G().pos(j) - num["G_POS"][j]));
      }
      e = std::max(e, std::fabs(s.get_A_time() - num["A_TIME"][0]));
      e = std::max(e, std::fabs(s.final_g - num["FINAL_G"][0]));

    } else if (kind == "REPLAN") {
      // informational + WIRING (ATTEMPTED exact). Not gated on local trajectory.
      sando::Parameters par; par.sim_env = "rviz_only"; par.local_solver = "minco";
      sando::SANDO s(par);
      s.worst_traj_time = num["WORST_TRAJ_TIME"][0];
      sando::RobotState st0; st0.pos = v3("STATE_POS");
      s.update_state(st0);
      int nocc = (int)num["NOCC"][0];
      vector<Eigen::Vector3d> occ;
      for (int i = 0; i < nocc; ++i)
        occ.push_back(Eigen::Vector3d(num["OCC"][i * 3], num["OCC"][i * 3 + 1], num["OCC"][i * 3 + 2]));
      s.update_occupancy_map_ptr(occ);
      sando::RobotState gt; gt.pos = v3("GTERM_POS");
      s.set_terminal_goal(gt);
      s.change_drone_status(sando::DroneStatus::TRAVELING);
      int ntraj = (int)num["NTRAJ"][0];
      for (int i = 0; i < ntraj; ++i) s.add_traj(build_traj(i), 0.0);
      auto pr = s.replan(0.0, 0.0);
      bool planned_ok = pr.first, attempted = pr.second;
      bool attempt_match = ((attempted ? 1.0 : 0.0) == num["ATTEMPTED"][0]);
      bool plannedok_match = ((planned_ok ? 1.0 : 0.0) == num["PLANNED_OK"][0]);
      // sampled position error vs Python's committed setpoints (only if both planned)
      double pos_err = 0.0;
      int nsamp = (int)num["NSAMP"][0];
      auto setpts = s.retrieve_goal_setpoints();
      if (planned_ok && num["PLANNED_OK"][0] != 0.0 && nsamp > 0 && !setpts.empty()) {
        // compare like-fraction samples (both committed plans parameterise s in [0,1])
        for (int s2 = 0; s2 < nsamp; ++s2) {
          double frac = (nsamp == 1) ? 0.0 : double(s2) / double(nsamp - 1);
          int ci = (int)std::lround(frac * (setpts.size() - 1));
          for (int j = 0; j < 3; ++j)
            pos_err = std::max(pos_err, std::fabs(setpts[ci].pos(j) - num["SET_POS"][s2 * 3 + j]));
        }
      }
      replan_n++;
      if (attempt_match) replan_attempt_match++;
      if (plannedok_match) replan_plannedok_match++;
      if (pos_err > replan_pos_err_max) { replan_pos_err_max = pos_err; replan_worst = name; }
      std::printf("  REPLAN %-20s : attempt_match=%d plannedok_match=%d "
                  "(cpp_ok=%d py_ok=%d) pos_err=%.3e nplan_after_cpp=%d\n",
                  name.c_str(), attempt_match ? 1 : 0, plannedok_match ? 1 : 0,
                  planned_ok ? 1 : 0, (int)num["PLANNED_OK"][0], pos_err,
                  (int)s.plan.size());
      // GATE only on ATTEMPTED (deterministic wiring up to the local solve).
      if (!attempt_match) { nfail++; std::printf("    -> ATTEMPTED mismatch FAIL\n"); }
      ncase++;
      return;
    }

    global_max_err = std::max(global_max_err, e);
    bool ok = e < TOL;
    if (!ok) nfail++;
    std::printf("  %-10s %-20s : max_err=%.3e  %s\n", kind.c_str(), name.c_str(), e,
                ok ? "PASS" : "FAIL");
    ncase++;
  };

  while (std::getline(f, line)) {
    if (line == "CASE") {
      num.clear(); str.clear();
      traj_str.clear(); traj_num.clear();
      in_str.clear(); in_num.clear(); out_str.clear(); out_num.clear();
      pred.clear();
      continue;
    }
    if (line == "END") { process(); continue; }
    std::istringstream is(line);
    string key; is >> key;
    string rest; std::getline(is, rest);
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);

    // string-valued keys
    if (key == "KIND" || key == "NAME" || key == "OCLASS") { str[key] = rest; continue; }
    auto digit_after = [&](const string& k, size_t prefix_len) -> bool {
      return k.size() > prefix_len && k[prefix_len] >= '0' && k[prefix_len] <= '9';
    };
    // traj keys: TRAJ<i>_FIELD  (require a digit right after the prefix)
    if (key.rfind("TRAJ", 0) == 0 && digit_after(key, 4) && key.find('_') != string::npos) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(4, us - 4));
      string field = key.substr(us + 1);
      if (field == "MODE" || field == "TX" || field == "TY" || field == "TZ" ||
          field == "TVX" || field == "TVY" || field == "TVZ")
        traj_str[i][field] = rest;
      else
        traj_num[i][field] = nums(rest);
      continue;
    }
    // obstsnap input keys: IN<i>_FIELD
    if (key.rfind("IN", 0) == 0 && digit_after(key, 2) && key.find('_') != string::npos) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(2, us - 2));
      string field = key.substr(us + 1);
      if (field == "CLASS") in_str[i][field] = rest;
      else in_num[i][field] = nums(rest);
      continue;
    }
    // obstsnap output keys: OUT<i>_FIELD
    if (key.rfind("OUT", 0) == 0 && digit_after(key, 3) && key.find('_') != string::npos) {
      size_t us = key.find('_');
      int i = std::stoi(key.substr(3, us - 3));
      string field = key.substr(us + 1);
      if (field == "KIND" || field == "CLASS") out_str[i][field] = rest;
      else out_num[i][field] = nums(rest);
      continue;
    }
    // predicted samples: PRED<i>
    if (key.rfind("PRED", 0) == 0 && digit_after(key, 4) && key.find('_') == string::npos) {
      int i = std::stoi(key.substr(4));
      pred[i] = nums(rest);
      continue;
    }
    num[key] = nums(rest);
  }

  std::printf("\nplanner golden (EXACT methods): %d fail, global_max_err=%.3e (tol=%.0e)\n",
              nfail, global_max_err, TOL);
  std::printf("planner REPLAN (informational, local NOT bit-exact): %d cases, "
              "attempt_match=%d/%d, plannedok_match=%d/%d, pos_err_max=%.3e (worst=%s)\n",
              replan_n, replan_attempt_match, replan_n, replan_plannedok_match, replan_n,
              replan_pos_err_max, replan_worst.c_str());
  std::printf("%s\n", nfail == 0
              ? "ALL EXACT METHODS PASS (C++ reproduces Python SANDO deterministic path)"
              : "FAILED (an EXACT method case did not match Python)");
  return nfail == 0 ? 0 : 1;
}
