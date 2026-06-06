// avoid_config — faithful C++ port of sando_py/local/avoid_config.py.
// Per-class avoidance params (mode/d_safe/weight) + default 2-class config + resolve_mode.
#pragma once
#include <string>
#include <map>

namespace sando {

struct AvoidParams {
  std::string name;
  std::string mode;   // "soft" | "hard"
  double d_safe;      // metres
  double weight;
};

inline std::map<std::string, AvoidParams> default_config() {
  return {
      {"wall", AvoidParams{"wall", "soft", 0.4, 1.0e1}},
      {"human", AvoidParams{"human", "hard", 0.8, 1.0e4}},
      // dynamic-agent: a MOVING obstacle (not static structure) -> hard continuous-time certificate,
      // tighter d_safe than a human so a dense crowd is not inflated to infeasibility. The per-class
      // axis is static-structure(soft) vs dynamic-agent(hard), keyed on motion (dynamic_speed_thresh).
      {"dynamic", AvoidParams{"dynamic", "hard", 0.5, 1.0e4}},
  };
}

// Resolve an obstacle's avoidance mode -> "hard" | "soft".
// override "" = per-class; "soft"/"hard" = global force. FAIL-SAFE: unknown/malformed -> "hard".
inline std::string resolve_mode(const std::string& class_name,
                                const std::map<std::string, AvoidParams>& cfg,
                                const std::string& override_ = "") {
  if (override_ == "soft" || override_ == "hard") return override_;
  auto it = cfg.find(class_name);
  if (it == cfg.end()) return "hard";
  const std::string& m = it->second.mode;
  if (m != "soft" && m != "hard") return "hard";
  return m;
}

}  // namespace sando
