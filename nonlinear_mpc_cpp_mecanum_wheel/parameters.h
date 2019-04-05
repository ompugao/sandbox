#pragma once
#include "picojson.h"
#include <Eigen/Core>
#include "obstacle.h"

class Parameters {
public:
  Parameters();
  void load_json(picojson::value& v);
  virtual ~Parameters();
  void print(std::ostream& os = std::cout);

  Eigen::Matrix<double, 4, 4> R;
  Eigen::Matrix<double, 4, 4> Rd;
  Eigen::Matrix<double, 6, 6> Q;
  Eigen::Matrix<double, 6, 6> Qf;

  double goal_distance_tolerance = 1.0;
  double stop_speed = 0.2;
  double max_time = 100;

  double target_speed = 3.0;
  int n_indices_search = 10;
  double dt = 0.2; // シミュレーションdelta時間
  double ht = 0.02; // 差分近似の幅

  double dl = 0.2;  // course tick

  int horizon = 6;  // horizon length
  std::string course_path;

  int max_iterations = 50;
  double du_th = 0.001;

  double circumscribed_area_cost = 10;
  double inflation_radius = 5;
  std::vector<Obstacle> obstacles;

  double local_area_x = -1.0;
  double local_area_y = -1.0;

  double alpha = 0.5;  //時間上昇ゲイン
  int N = 10;          // 分割数
  double tf = 1;       // 最終予測時間 (time finite)
  double zeta = 100.0;
  double iteration_threshold = 1e-6;

  bool visualize = true;
};
