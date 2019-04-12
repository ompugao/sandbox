#include "parameters.h"
#include <iostream>

Parameters::Parameters() {
  // R.diagonal() << 0.1, 0.1, 0.1, 0.1;
  // Rd.diagonal() << 10, 10, 10, 10;
  // Q.diagonal() << 1, 1, 0.1, 10.0, 10.0, 3.0;
  // Qf = Q;
}

void Parameters::load_json(picojson::value& v) {
  auto& obj = v.get<picojson::object>();
  if (obj.find("R") != obj.end()) {
    picojson::array arr = obj["R"].get<picojson::array>();
    for (int i = 0; i < R.cols(); i++) {
      R(i, i) = arr[i].get<double>();
    }
  }
  if (obj.find("Rd") != obj.end()) {
    picojson::array arr = obj["Rd"].get<picojson::array>();
    for (int i = 0; i < Rd.cols(); i++) {
      Rd(i, i) = arr[i].get<double>();
    }
  }
  if (obj.find("Q") != obj.end()) {
    picojson::array arr = obj["Q"].get<picojson::array>();
    for (int i = 0; i < Q.cols(); i++) {
      Q(i, i) = arr[i].get<double>();
    }
  }
  if (obj.find("Qf") != obj.end()) {
    picojson::array arr = obj["Qf"].get<picojson::array>();
    for (int i = 0; i < Qf.cols(); i++) {
      Qf(i, i) = arr[i].get<double>();
    }
  }
  if (obj.find("goal_distance_tolerance") != obj.end()) {
    goal_distance_tolerance = obj["goal_distance_tolerance"].get<double>();
  }
  if (obj.find("stop_speed") != obj.end()) {
    stop_speed = obj["stop_speed"].get<double>();
  }
  if (obj.find("max_time") != obj.end()) {
    max_time = obj["max_time"].get<double>();
  }
  if (obj.find("target_speed") != obj.end()) {
    target_speed = obj["target_speed"].get<double>();
  }
  if (obj.find("n_indices_search") != obj.end()) {
    n_indices_search = obj["n_indices_search"].get<double>();
  }
  if (obj.find("dl") != obj.end()) {
    dl = obj["dl"].get<double>();
  }
  if (obj.find("dt") != obj.end()) {
    dt = obj["dt"].get<double>();
  }
  if (obj.find("ht") != obj.end()) {
    ht = obj["ht"].get<double>();
  }
  if (obj.find("horizon") != obj.end()) {
    horizon = obj["horizon"].get<double>();
  }
  if (obj.find("course_path") != obj.end()) {
    course_path = obj["course_path"].get<std::string>();
  }
  if (obj.find("max_iterations") != obj.end()) {
    max_iterations = obj["max_iterations"].get<double>();
  }
  if (obj.find("du_th") != obj.end()) {
    du_th = obj["du_th"].get<double>();
  }
  if (obj.find("circumscribed_area_cost") != obj.end()) {
    circumscribed_area_cost = obj["circumscribed_area_cost"].get<double>();
  }
  if (obj.find("inflation_radius") != obj.end()) {
    inflation_radius = obj["inflation_radius"].get<double>();
  }
  if (obj.find("obstacles") != obj.end()) {
    picojson::array arr = obj["obstacles"].get<picojson::array>();
    obstacles.clear();
    for (auto&& v : arr) {
      auto& a = v.get<picojson::array>();
      obstacles.emplace_back(Obstacle(a[0].get<double>(), a[1].get<double>(), a[2].get<double>(),
                                      circumscribed_area_cost, inflation_radius));
    }
  }
  if (obj.find("local_area") != obj.end()) {
    picojson::array arr = obj["local_area"].get<picojson::array>();
    local_area_x = arr[0].get<double>();
    local_area_y = arr[1].get<double>();
  }
  if (obj.find("alpha") != obj.end()) {
    alpha = obj["alpha"].get<double>();
  }
  if (obj.find("zeta") != obj.end()) {
    zeta = obj["zeta"].get<double>();
  }
  if (obj.find("tf") != obj.end()) {
    tf = obj["tf"].get<double>();
  }
  if (obj.find("iteration_threshold") != obj.end()) {
    iteration_threshold = obj["iteration_threshold"].get<double>();
  }
  if (obj.find("visualize") != obj.end()) {
    visualize = obj["visualize"].get<bool>();
  }
}
Parameters::~Parameters() {}
void Parameters::print(std::ostream& os) {
  os << "R:" << std::endl;
  os << R << std::endl;
  os << "Rd:" << std::endl;
  os << Rd << std::endl;
  os << "Q:" << std::endl;
  os << Q << std::endl;
  os << "Qf:" << std::endl;
  os << Qf << std::endl;
  os << "goal_distance_tolerance: ";
  os << goal_distance_tolerance << std::endl;
  os << "stop_speed: ";
  os << stop_speed << std::endl;
  os << "max_time: ";
  os << max_time << std::endl;
  os << "target_speed: ";
  os << target_speed << std::endl;
  os << "n_indices_search: ";
  os << n_indices_search << std::endl;
  os << "dl: ";
  os << dl << std::endl;
  os << "dt: ";
  os << dt << std::endl;
  os << "ht: ";
  os << ht << std::endl;
  os << "horizon: ";
  os << horizon << std::endl;
  os << "course_path: ";
  os << course_path << std::endl;
  os << "max_iterations: ";
  os << max_iterations << std::endl;
  os << "du_th: ";
  os << du_th << std::endl;
  os << "circumscribed_area_cost: ";
  os << circumscribed_area_cost << std::endl;
  os << "inflation_radius: ";
  os << inflation_radius << std::endl;
  os << "local_area: ";
  os << local_area_x << ", " << local_area_y << std::endl;
  os << "alpha: ";
  os << alpha << std::endl;
  os << "zeta: ";
  os << zeta << std::endl;
  os << "tf: ";
  os << tf << std::endl;
  os << "iteration_threshold: ";
  os << iteration_threshold << std::endl;
}

