#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#include <fstream>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#include "csv.h"

#include "picojson.h"

class ScopedTime {
public:
  ScopedTime(const std::string& msg = "") : msg_(msg) {
    start_ = std::chrono::system_clock::now();
  }
  double lap() {
    const auto duration_time = std::chrono::system_clock::now() - start_;
    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_time);
    const double duration_ms = duration_ns.count()/1000.0/1000.0;
    return duration_ms;
  }
  virtual ~ScopedTime() {
    const auto duration_ms = this->lap();
    std::cerr << "[" << duration_ms << " ms] " << msg_ << std::endl;
  }

private:
  std::chrono::system_clock::time_point start_;
  std::string msg_;
};

double normalize_angle_positive(double angle) {
  return std::fmod(std::fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
}

double normalize_angle(double angle) {
  double a = normalize_angle_positive(angle);
  if (a > M_PI) {
    a -= 2.0 * M_PI;
  }
  return a;
}

class Obstacle {
public:
  Obstacle(double x, double y, double r, double circumscribed_area_cost, double inflation_radius) {
    x_ = x;
    y_ = y;
    r_ = r;
    // TODO get radius of vehicle
    // double infl = r - inflation_radius;
    // if (std::abs(infl) < 1) {
    //  LOG(WARNING) << "inflation radius is too small! r: " << r << ", inflation radius: " << inflation_radius;
    //  infl = 1;
    //}
    a_ = circumscribed_area_cost / inflation_radius;
  }
  virtual ~Obstacle() {}

  const double& x() {
    return x_;
  }
  const double& y() {
    return y_;
  }
  const double& r() {
    return r_;
  }
  const double& a() {
    return a_;
  }

  void step(double dt) {}
  double x_, y_, r_;
  double a_;
};

class Parameters {
public:
  Parameters() {
    R.diagonal() << 0.1, 0.1, 0.1, 0.1;
    Rd.diagonal() << 10, 10, 10, 10;
    Q.diagonal() << 1, 1, 0.1, 10.0, 10.0, 3.0;
    Qf = Q;
  }

  void load_json(picojson::value& v) {
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
  }
  virtual ~Parameters() {}
  void print(std::ostream& os = std::cout) {
    os << "R:" << std::endl;
    os << R << std::endl;
    os << "Rd:" << std::endl;
    os << Rd << std::endl;
    os << "Q:" << std::endl;
    os << Q << std::endl;
    os << "Qf:" << std::endl;
    os << Qf << std::endl;
  }

  Eigen::Matrix<double, 4, 4> R;
  Eigen::Matrix<double, 4, 4> Rd;
  Eigen::Matrix<double, 6, 6> Q;
  Eigen::Matrix<double, 6, 6> Qf;

  double goal_distance_tolerance = 1.0;
  double stop_speed = 0.2;
  double max_time = 100;

  double target_speed = 3.0;
  int n_indices_search = 10;
  double dt = 0.2;

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
};

class VehicleParameters {
public:
  VehicleParameters() {}
  virtual ~VehicleParameters() {}

  const double lf = 0.5;
  const double lb = 0.5;

  const double width = 2.0;
  // const double wheel_len = 0.3;
  // const double wheel_width = 0.2;
  const double wheel_radius = 0.152;
  const double wheel_mass = 0.3;
  const double Iw = 1.0 / 2 * wheel_mass * wheel_radius * wheel_radius;
  const double mass = 15.0; //40.0;
  const double Iv = 1.0 / 2 * mass * 0.5*0.5;//0.3 * 0.3;

  const double input_gain = 1.0;//0.7;
  const double damping_gain = 0.01;

  const double max_speed = 5.0;
  const double min_speed = -5.0;

  const double max_motor_torque = 1.2;//0.270 * 12; // motor torque * gear ratio
  const double max_motor_torque_vel = 0.500 * 12;
};

class WayPoint {
public:
  WayPoint(double _x, double _y, double _theta, double _speed = 0) : x(_x), y(_y), theta(_theta), speed(_speed) {}
  virtual ~WayPoint() {}
  double x;
  double y;
  double theta;
  double speed;
};

class Course {
public:
  Course() {}
  virtual ~Course() {}
  size_t size() const {
    return waypoints.size();
  }
  void load_csv(const std::string& path) {
    waypoints.clear();
    xs.clear();
    ys.clear();

    io::CSVReader<4> in(path);
    in.read_header(io::ignore_extra_column, "cx", "cy", "ctheta", "ck");
    double cx, cy, ctheta, ck;
    while (in.read_row(cx, cy, ctheta, ck)) {
      waypoints.emplace_back(WayPoint(cx, cy, ctheta));
      xs.emplace_back(cx);
      ys.emplace_back(cy);
    }
  }

  void set_speed_profile(const double target_v) {
    for (int i = 0; i < waypoints.size() - 1; i++) {
      double dx = waypoints[i + 1].x - waypoints[i].x;
      double dy = waypoints[i + 1].y - waypoints[i].y;
      double movedir = std::atan2(dy, dx);
      double dir = 0.0;
      if (dx != 0.0 && dy != 0.0) {
        double d_angle = std::abs(normalize_angle(movedir - waypoints[i].theta));
        if (d_angle > M_PI / 4.0) {
          dir = -1.0;
        } else {
          dir = 1.0;
        }
        waypoints[i].speed = dir * target_v;
      }
    }
    waypoints[waypoints.size() - 1].speed = 0;
  }

  void smooth_theta() {
    for (int i = 0; i < waypoints.size() - 1; i++) {
      WayPoint& w_n = waypoints[i + 1];
      WayPoint& w = waypoints[i];
      double dtheta = w_n.theta - w.theta;
      while (dtheta >= M_PI / 2.0) {
        w_n.theta -= M_PI * 2.0;
        dtheta = w_n.theta - w.theta;
      }

      while (dtheta <= -M_PI / 2.0) {
        w_n.theta += M_PI * 2.0;
        dtheta = w_n.theta - w.theta;
      }
    }
  }

  void plot() {
    plt::plot(xs, ys, "-r");
  }

  std::vector<WayPoint> waypoints;

  // for visualization
  std::vector<double> xs, ys;
};

class Vehicle {
public:
  Vehicle(const std::shared_ptr<Parameters>& env_params) {
    params_ = env_params;
    state_ = Eigen::VectorXd(num_state_);

    A_ = Eigen::MatrixXd::Identity(num_state_, num_state_);
    B_ = Eigen::MatrixXd::Zero(num_state_, num_input_);
    C_ = Eigen::VectorXd::Zero(num_state_);

    const double& dt = params_->dt;
    const double& r = vparams_.wheel_radius;
    const double& Iv = vparams_.Iv;
    const double& Iw = vparams_.Iw;
    const double& lb = vparams_.lb;
    const double& lf = vparams_.lf;
    const double& c = vparams_.damping_gain;
    const double& k = vparams_.input_gain;
    const double& M = vparams_.mass;
    const double w = vparams_.width / 2;
    const double B = (lb + w) * (lb + w) + (lf + w) * (lf + w);
    const double A = r * r * Iv + B * Iw;

    // theta
    A_(2, 5) += dt;

    const double coeff1 = r * r * M + 2 * Iw;
    A_(3, 3) += dt * (-2 * c / coeff1);
    B_(3, 0) += dt * (k * r / 2.0 * (-1)) / coeff1;
    B_(3, 1) += dt * (k * r / 2.0 * (-1)) / coeff1;
    B_(3, 2) += dt * (k * r / 2.0 * (+1)) / coeff1;
    B_(3, 3) += dt * (k * r / 2.0 * (+1)) / coeff1;

    const double coeff2 = coeff1 - Iw * Iw * (lb - lf) * (lb - lf) / A;

    A_(4, 4) += dt * (Iw * (lb - lf) * (lb - lf) / A - 2) * c / coeff2;
    A_(4, 5) += dt * (Iw * B / A + 1) * c * (lb - lf) / coeff2;
    B_(4, 0) += dt * ((k * r / 2.0 * Iw / A * (lb - lf) * (lf + w)) + k * r / 2) / coeff2;
    B_(4, 1) += dt * ((k * r / 2.0 * Iw / A * (lb - lf) * (lb + w)) - k * r / 2) / coeff2;
    B_(4, 2) += dt * ((k * r / 2.0 * Iw / A * (lb - lf) * (lb + w)) - k * r / 2) / coeff2;
    B_(4, 3) += dt * ((k * r / 2.0 * Iw / A * (lb - lf) * (lf + w)) + k * r / 2) / coeff2;

    const double coeff3 = A - Iw * Iw * (lb - lf) * (lb - lf) / coeff1;

    A_(5, 4) += dt * (-2 * Iw / coeff1 + 1) * c * (lb - lf) / coeff3;
    A_(5, 5) += dt * c * (Iw * (lb - lf) * (lb - lf) / coeff1 - B) / coeff3;

    B_(5, 0) += dt * ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (+1.0) + k * r / 2.0 * (lf + w)) / coeff3;
    B_(5, 1) += dt * ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (-1.0) + k * r / 2.0 * (lb + w)) / coeff3;
    B_(5, 2) += dt * ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (-1.0) + k * r / 2.0 * (lb + w)) / coeff3;
    B_(5, 3) += dt * ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (+1.0) + k * r / 2.0 * (lf + w)) / coeff3;
  }

  virtual ~Vehicle() {}
  void get_linear_matrix(double vx, double vy, double theta, Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                         Eigen::VectorXd& C) const {
    A = A_;
    B = B_;
    C = C_;

    A(0, 2) += params_->dt * (-vx * std::sin(theta) - vy * std::cos(theta));
    A(0, 3) += params_->dt * std::cos(theta);
    A(0, 4) += params_->dt * (-std::sin(theta));
    A(1, 2) += params_->dt * (vx * std::cos(theta) - vy * std::sin(theta));
    A(1, 3) += params_->dt * std::sin(theta);
    A(1, 4) += params_->dt * std::cos(theta);
    C(0) = params_->dt * (vx * std::sin(theta) + vy * std::cos(theta)) * theta;
    C(1) = params_->dt * (-vx * std::cos(theta) + vy * std::sin(theta)) * theta;
    return;
  }

  void plot() {
    double radius = vparams_.width / 2.0;
    const int n = 200;
    std::vector<double> x(n + 1), y(n + 1);
    for (int i = 0; i < n + 1; i++) {
      x[i] = state_(0) + radius * std::cos(i * 1.0 / (2 * M_PI));
      y[i] = state_(1) + radius * std::sin(i * 1.0 / (2 * M_PI));
    }
    plt::plot(x, y, "r-");
    plt::plot({state_(0), state_(0) + std::cos(state_(2)) * radius},
              {state_(1), state_(1) + std::sin(state_(2)) * radius}, "r-");
    plt::plot({state_(0)}, {state_(1)}, "*g");
  }

  Eigen::VectorXd next_state(const Eigen::VectorXd& state, Eigen::VectorXd u) const {
    Eigen::VectorXd u_copy = u;
    u_copy(0) = std::clamp(u(0), -vparams_.max_motor_torque, vparams_.max_motor_torque);
    u_copy(1) = std::clamp(u(1), -vparams_.max_motor_torque, vparams_.max_motor_torque);
    u_copy(2) = std::clamp(u(2), -vparams_.max_motor_torque, vparams_.max_motor_torque);
    u_copy(3) = std::clamp(u(3), -vparams_.max_motor_torque, vparams_.max_motor_torque);

    Eigen::MatrixXd A, B;
    Eigen::VectorXd C(num_state_);
    get_linear_matrix(state(3), state(4), state(2), A, B, C);

    Eigen::VectorXd new_state = A * state + B * u_copy + C;
    double v;
    v = std::clamp(new_state(3), vparams_.min_speed, vparams_.max_speed);
    new_state(3) = v;
    v = std::clamp(new_state(4), vparams_.min_speed, vparams_.max_speed);
    new_state(4) = v;
    return new_state;
  }

  int find_nearest_index(const Course& c, int pind) const {
    double mindist = 1e10;
    int minindex = pind;

    for (int i = pind; i < std::min(pind + params_->n_indices_search, static_cast<int>(c.size())); i++) {
      double dist = std::hypot(c.waypoints[i].x - state_(0), c.waypoints[i].y - state_(1));
      if (mindist > dist) {
        mindist = dist;
        minindex = i;
      }
    }
    return minindex;
  }

  Eigen::MatrixXd predict_motion(Eigen::VectorXd state, Eigen::MatrixXd inputs, const Eigen::MatrixXd& xref) const {
    Eigen::MatrixXd xbar = Eigen::MatrixXd::Zero(xref.rows(), xref.cols());

    xbar.col(0) = state;

    for (int i = 0; i < inputs.cols(); i++) {
      state = next_state(state, inputs.col(i));
      xbar.col(i + 1) = state;
    }

    return xbar;
  }

  void set_state(double x, double y, double theta, double vx, double vy, double thetadot) {
    state_(0) = x;
    state_(1) = y;
    state_(2) = theta;
    state_(3) = vx;
    state_(4) = vy;
    state_(5) = thetadot;
  }

  void set_state(const Eigen::MatrixXd& state) {
    state_ = state;
  }

  const Eigen::VectorXd& get_state() const {
    return state_;
  }

  void get_state(double& x, double& y, double& theta, double& vx, double& vy, double& thetadot) const {
    x = state_(0);
    y = state_(1);
    theta = state_(2);
    vx = state_(3);
    vy = state_(4);
    thetadot = state_(5);
  }

  bool is_arrived(const Course& c, const WayPoint& w, int iw) {
    double d = std::hypot(state_(0) - w.x, state_(1) - w.y);
    bool closeenough = (d < params_->goal_distance_tolerance) && (c.size() == iw + 1);
    bool stopped = std::abs(state_(3)) < params_->stop_speed && std::abs(state_(4)) < params_->stop_speed;
    return closeenough && stopped;
  }

  Eigen::VectorXd state_;
  VehicleParameters vparams_;

  std::shared_ptr<Parameters> params_;
  const int num_state_ = 6;
  const int num_input_ = 4;

  Eigen::MatrixXd A_, B_, C_;
};

std::pair<Eigen::MatrixXd, int> calc_ref_trajectory(const Vehicle& v, const Course& c,
                                                    const std::shared_ptr<Parameters>& params, int pind) {
  Eigen::MatrixXd xref = Eigen::MatrixXd(v.num_state_, params->horizon + 1);
  int i_closest = v.find_nearest_index(c, pind);

  if (i_closest <= pind) {
    i_closest = pind;  // move forward!
  }
  xref(0, 0) = c.waypoints[i_closest].x;
  xref(1, 0) = c.waypoints[i_closest].y;
  xref(2, 0) = c.waypoints[i_closest].theta;
  xref(3, 0) = c.waypoints[i_closest].speed;
  xref(4, 0) = 0;
  xref(5, 0) = 0;

  double x, y, theta, vx, vy, thetadot;
  double t = 0;
  v.get_state(x, y, theta, vx, vy, thetadot);
  double vel = std::hypot(vx, vy);

  double travel = 0.0;
  for (int i = 1; i < params->horizon + 1; i++) {
    travel += std::abs(c.waypoints[i_closest].speed) * params->dt;
    int dind = static_cast<int>(travel / params->dl);
    int j = i_closest + dind;
    if (j >= c.size()) {
      j = c.size() - 1;
    }
    std::cout << "-- " << j << std::endl;
    const WayPoint& w = c.waypoints[j];
    xref(0, i) = w.x;
    xref(1, i) = w.y;
    xref(2, i) = w.theta;
    xref(3, i) = w.speed;
    xref(4, i) = 0;
    xref(5, i) = 0;
  }
  return std::move(std::make_pair(xref, i_closest));
}

bool linear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar,
                        const Eigen::VectorXd& x0, const std::shared_ptr<Parameters>& params, Eigen::MatrixXd& ou);
bool iterative_linear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, Eigen::MatrixXd& ou,
                                  const std::shared_ptr<Parameters>& params) {
  int i = 0;
  bool sol_found = false;
  for (; i < params->max_iterations; i++) {
    Eigen::MatrixXd xbar = v.predict_motion(v.get_state(), ou, xref);
    Eigen::MatrixXd prev_ou = ou;
    sol_found = linear_mpc_control(v, xref, xbar, v.get_state(), params, ou);
    if (!sol_found) {
      LOG(WARNING) << "Solution not found (iter:" << i << ")";
      continue;
    }
    LOG(INFO) << "-- inputs at iteration " << i << ": ----";
    LOG(INFO) << ou;
    LOG(INFO) << "-------------------------------";
    double udiff = (ou - prev_ou).cwiseAbs().sum();
    LOG(INFO) << ">> udiff: " << udiff;
    if (udiff < params->du_th) {
      break;
    }
  }
  LOG(INFO) << "-- finished iteration : " << i;
  return sol_found;
}

// static int kStride = 10;
class MPCCostFunctor : public ceres::EvaluationCallback {
public:
  // typedef ceres::DynamicAutoDiffCostFunction<MPCCostFunctor, kStride> MPCCostFunction;
  // typedef ceres::DynamicCostFunction MPCCostFunction;

  MPCCostFunctor(const Vehicle& vehicle, const std::shared_ptr<Parameters>& params, const Eigen::VectorXd& x0,
                 const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar, std::vector<Obstacle*>& obstacles) {
    vehicle_ = &vehicle;
    params_ = params;
    x0_ = x0;
    // x_ = Eigen::MatrixXd::Zero(vehicle_->num_state_, params->horizon + 1);
    xref_ = xref;
    xbar_ = xbar;
    // xdiff_ = Eigen::MatrixXd::Zero(vehicle_->num_state_, params->horizon + 1);
    // udiff_ = Eigen::MatrixXd::Zero(vehicle_->num_input_, params->horizon + 1);
    // deriv_ = Eigen::MatrixXd::Zero(vehicle_->num_input_, params->horizon);
    obstacles_ = obstacles;
  }
  virtual ~MPCCostFunctor() {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> u =
    const auto u = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        parameters[0], vehicle_->num_input_, params_->horizon);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x_(vehicle_->num_state_, params_->horizon + 1);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xdiff_(vehicle_->num_state_, params_->horizon + 1),
        udiff_(vehicle_->num_input_, params_->horizon - 1);

    x_.col(0) = x0_.cast<T>();
    for (int t = 0; t < params_->horizon; t++) {
      vehicle_->get_linear_matrix(xbar_.cast<double>()(3, t), xbar_.cast<double>()(4, t), xbar_.cast<double>()(2, t),
                                  A_, B_, C_);
      x_.col(t + 1) = A_.cast<T>() * x_.col(t) + B_.cast<T>() * u.col(t) + C_;
      // if (x_.col(t+1)(4) > params_->target_speed) {
      // return false;
      //}
      if (t < params_->horizon - 1) {
        udiff_.col(t) = u.col(t + 1) - u.col(t);
      }
    }
    xdiff_.col(0).setZero();
    for (int t = 0; t < params_->horizon + 1; t++) {
      if (t != 0) {
        xdiff_.col(t) = x_.col(t) - xref_.col(t);
      }
    }

    int cnt = 0;
    auto residuals_xdiff = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        &residuals[cnt], vehicle_->num_state_, params_->horizon);
    cnt += vehicle_->num_state_ * params_->horizon;
    residuals_xdiff = (params_->Q.cast<T>() * xdiff_.block(0, 0, x0_.rows(), params_->horizon));

    auto residuals_udiff = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        &residuals[cnt], vehicle_->num_input_, params_->horizon - 1);
    cnt += vehicle_->num_input_ * (params_->horizon - 1);
    residuals_udiff = (params_->Rd.cast<T>() * udiff_);

    auto residuals_u = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        &residuals[cnt], vehicle_->num_input_, params_->horizon);
    cnt += vehicle_->num_input_ * params_->horizon;
    residuals_u = (params_->R.cast<T>() * u);

    // double denom = (2 * params_->inflation_radius * params_->inflation_radius);
    // double numer = params_->circumscribed_area_cost / (std::sqrt(2*M_PI) * params_->inflation_radius);

    for (auto&& o : obstacles_) {
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pos(vehicle_->num_state_, 1);
      pos << o->x(), o->y(), 0.0, 0.0, 0.0, 0.0;
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x_from_obstacle(vehicle_->num_state_, params_->horizon + 1);
      x_from_obstacle.block(0, 0, 2, x_.cols()) = x_.block(0, 0, 2, x_.cols());
      for (int t = 0; t < params_->horizon + 1; t++) {
        x_from_obstacle.col(t) -= pos.cast<T>();
        T d = x_from_obstacle.col(t).norm();
        T b = (T)(params_->inflation_radius + o->r());
        if (d < b) {
          residuals[cnt] = (T)(o->a()) * (b - d);
        } else {
          residuals[cnt] = T(0);
        }
        cnt += 1;
      }
      // residuals[4] += x_from_obstacle.colwise().squaredNorm().exp().sum();
      // residuals[4] += (-((x_.block(0, 0, 2, x_.cols()).colwise() - pos).colwise().squaredNorm())).exp().sum();
    }

    return true;
  }
  // virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
  //   Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::ColMajor> u =
  //       Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::ColMajor>>(parameters, params_->horizon);
  //   x_.col(0) = x0_;
  //   for (int t = 0; t < params->horizon; t++) {
  //     vehicle_->get_linear_matrix(x0_(4), x0_(2), A_, B_, C_);
  //     x_.col(t + 1) = A_ * x_.col(t) + B_ * u + C_;
  //     if (t != 0) {
  //       xdiff_.col(t) = x_.col(t) - xref_.col(t);
  //     }
  //     udiff_.col(t) = u.col(t+1) - u.col(t);
  //     deriv_.col(t) = (2 * params_->R * u.col(t)) - 2 * params_->R * u_ss_ + (4 * params->Rd * u) + (-2 * params->Rd
  //     * u_past);
  //   }
  // }

  static ceres::CostFunction* Create(const Vehicle& vehicle, const std::shared_ptr<Parameters>& params,
                                     Eigen::MatrixXd& u, std::vector<double*>& parameter_blocks,
                                     const Eigen::VectorXd& x0, const Eigen::MatrixXd& xref,
                                     const Eigen::MatrixXd& xbar) {
    int num_residuals = vehicle.num_state_ * params->horizon + vehicle.num_input_ * (params->horizon - 1) +
                        vehicle.num_input_ * params->horizon;


    double x, y, theta, vx, vy, thetadot;
    double t = 0;
    vehicle.get_state(x, y, theta, vx, vy, thetadot);

    std::vector<Obstacle*> surrounding_obstacles;
    if (params->local_area_x < 0 || params->local_area_y < 0) {
      for (auto&& obj : params->obstacles) {
        surrounding_obstacles.push_back(&obj);
        num_residuals += (params->horizon+1);
      }
    } else {
      for (auto&& obj : params->obstacles) {
        if (x - params->local_area_x/2.0 < obj.x() && obj.x() < x + params->local_area_x/2.0
            &&
            y - params->local_area_y/2.0 < obj.y() && obj.y() < y + params->local_area_y/2.0) {
          surrounding_obstacles.push_back(&obj);
          num_residuals += (params->horizon+1);
        }
      }
    }

    MPCCostFunctor* costfunctor = new MPCCostFunctor(vehicle, params, x0, xref, xbar, surrounding_obstacles);
    ceres::DynamicCostFunction* costfn = new ceres::DynamicAutoDiffCostFunction<MPCCostFunctor, 5>(costfunctor);
    // for (int i = 0; i < u.cols(); i++) {
    // parameter_blocks.push_back(&u.data()[u.rows()*i]);
    //}
    parameter_blocks.push_back(u.data());
    costfn->AddParameterBlock(u.size());
    costfn->SetNumResiduals(num_residuals);
    return costfn;
  }

  void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override {}

private:
  std::shared_ptr<Parameters> params_;
  const Vehicle* vehicle_;
  // memory allocations
  mutable Eigen::VectorXd x0_;
  mutable Eigen::MatrixXd A_, B_;
  mutable Eigen::VectorXd C_;
  Eigen::MatrixXd xref_, xbar_;

  std::vector<Obstacle*> obstacles_;

  // mutable Eigen::MatrixXd A_, B_,Bd_, Q_, Q_final_,  R_, R_delta_, disturbance_, insecure_, u_ss_, x_ss_, x0_,
  // u_prev_, x_states, u, deriv_wrt_u, u_past, lambdas_x, lambdas_u,lambdas_u_ref, u_horizon, u_current;
};

bool linear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar,
                        const Eigen::VectorXd& x0, const std::shared_ptr<Parameters>& params, Eigen::MatrixXd& ou) {
  ceres::Problem prob;

  // Eigen::MatrixXd u =
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(v.num_input_, params->horizon);
  std::vector<double*> parameter_blocks;
  ceres::CostFunction* costfn = MPCCostFunctor::Create(v, params, ou, parameter_blocks, x0, xref, xbar);
  prob.AddResidualBlock(costfn, NULL, parameter_blocks);
  for (int i = 0; i < ou.size(); i++) {
    prob.SetParameterLowerBound(parameter_blocks[0], i, -v.vparams_.max_motor_torque);
    prob.SetParameterUpperBound(parameter_blocks[0], i, v.vparams_.max_motor_torque);
  }

  ceres::Solver::Options options;
  // LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer, &options.minimizer_type))
  //<< "Invalid minimizer: " << FLAGS_minimizer << ", valid options are: trust_region and line_search.";
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;

  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 40;
  options.linear_solver_type = ceres::CGNR;
  options.num_threads = 4;

  options.update_state_every_iteration = true;
  // options.evaluation_callback =

  ceres::Solver::Summary summary;

  {
    ScopedTime time("optimization");
    ceres::Solve(options, &prob, &summary);
  }

  LOG(INFO) << summary.FullReport() << std::endl;

  // ou = u;
  return summary.IsSolutionUsable();
}

void plot_obstacles(const std::shared_ptr<Parameters>& params) {
  for (auto&& o : params->obstacles) {
    const int n = 100;
    std::vector<double> x(n + 1), y(n + 1);
    for (int i = 0; i < n + 1; i++) {
      x[i] = o.x() + o.r() * std::cos(i * 1.0 / (2 * M_PI));
      y[i] = o.y() + o.r() * std::sin(i * 1.0 / (2 * M_PI));
    }
    plt::plot(x, y, "g-");
  }
}

void main1(const std::shared_ptr<Parameters>& params) {
  Course c;
  c.load_csv(params->course_path);
  c.set_speed_profile(params->target_speed);
  c.smooth_theta();

  Vehicle vehicle(params);
  const WayPoint& w = c.waypoints[0];
  vehicle.set_state(w.x, w.y - 3, normalize_angle(w.theta - 2*M_PI / 3.0), 0, 0, 0);

  // Eigen::MatrixXd A, B;
  // Eigen::VectorXd C(5);
  // vehicle.get_linear_matrix(1.0, 0.5, A, B, C);
  // std::cout << "A" << std::endl;
  // std::cout << A << std::endl;
  // std::cout << "B" << std::endl;
  // std::cout << B << std::endl;
  // std::cout << "C" << std::endl;
  // std::cout << C << std::endl;
  // std::cout << "----" << std::endl;
  std::vector<double> logging_x, logging_y, logging_theta, logging_thetadot, logging_vx, logging_vy, logging_t;
  std::vector<Eigen::VectorXd> logging_u;

  double x, y, theta, vx, vy, thetadot;
  double t = 0;
  vehicle.get_state(x, y, theta, vx, vy, thetadot);
  logging_x.push_back(x);
  logging_y.push_back(y);
  logging_theta.push_back(theta);
  logging_vx.push_back(vx);
  logging_vy.push_back(vy);
  logging_thetadot.push_back(thetadot);
  logging_u.emplace_back(Eigen::VectorXd::Zero(vehicle.num_input_));
  logging_t.push_back(t);

  int target_index = vehicle.find_nearest_index(c, 0);

  // auto ou = Eigen::MatrixXd(vehicle.num_input_, params->horizon);
  Eigen::MatrixXd ou =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(vehicle.num_input_, params->horizon);
  auto ox = Eigen::MatrixXd(vehicle.num_state_, params->horizon + 1);

  Eigen::MatrixXd xref;
  while (t < params->max_time) {
    LOG(INFO) << "-----------------------------------------------------------------------------------------------------"
                 "-------------------------";
    std::tie(xref, target_index) = calc_ref_trajectory(vehicle, c, params, target_index);
    bool sol_found;
    double computation_time = 0;
    {
      ScopedTime st("iterative_linear_mpc_control");
      sol_found = iterative_linear_mpc_control(vehicle, xref, ou, params);
      computation_time = st.lap();
    }
    if (!sol_found) {
      LOG(WARNING) << "Failed to compute control. exit.";
      return;
    }
    // ou(0, 0) = 0.093;
    // ou(1, 0) = 0.09;
    Eigen::MatrixXd motion = vehicle.predict_motion(vehicle.get_state(), ou, xref);
    LOG(INFO) << "-- state:  ----";
    Eigen::IOFormat stateformat(4, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    LOG(INFO) << vehicle.get_state().format(stateformat);
    LOG(INFO) << "-- input:  ----";
    LOG(INFO) << ou.col(0).format(stateformat);
    LOG(INFO) << "-- inputs: ----";
    LOG(INFO) << ou;
    LOG(INFO) << "---------------";
    vehicle.set_state(vehicle.next_state(vehicle.get_state(), ou.col(0)));

    vehicle.get_state(x, y, theta, vx, vy, thetadot);
    logging_x.push_back(x);
    logging_y.push_back(y);
    logging_theta.push_back(theta);
    logging_vx.push_back(vx);
    logging_vy.push_back(vy);
    logging_thetadot.push_back(thetadot);
    logging_u.push_back(ou.col(0));
    logging_t.push_back(t);

    c.plot();
    plot_obstacles(params);

    std::vector<double> mpc_x, mpc_y;
    for (int i = 0; i < motion.cols(); i++) {
      mpc_x.push_back(motion(0, i));
      mpc_y.push_back(motion(1, i));
    }
    plt::plot(mpc_x, mpc_y, "xr");

    auto& w = c.waypoints[target_index];
    plt::plot({w.x}, {w.y}, "xg");

    std::vector<double> ref_x, ref_y;
    for (int i = 0; i < xref.cols(); i++) {
      ref_x.push_back(xref(0, i));
      ref_y.push_back(xref(1, i));
    }
    plt::plot(ref_x, ref_y, "xk");

    plt::plot(logging_x, logging_y, "ob");
    vehicle.plot();

    plt::axis("equal");
    plt::grid(true);
    // plt::xlim(-5 + x, 5 + x);
    // plt::ylim(-5 + y, 5 + y);
    std::stringstream ss;
    ss << "time[s] " << t << " | speed[m/s] " << std::hypot(vx, vy) << " | computation[ms] " << computation_time;
    plt::title(ss.str());
    plt::pause(0.0001);
    plt::clf();

    t += params->dt;
    if (t == params->dt) {
      std::cin.get();
    }
    if (vehicle.is_arrived(c, w, target_index)) {
      break;
    }
    // if (t > 10) {
    // break;
    //}
  }
  plt::clf();

  plt::figure();
  plt::plot(logging_t, logging_vx, "-r");
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("Speed(x) [m/s]");

  plt::figure();
  plt::plot(logging_t, logging_vy, "-r");
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("Speed(y) [m/s]");

  std::vector<double> logging_u0, logging_u1, logging_u2, logging_u3;
  for (auto&& v : logging_u) {
    logging_u0.push_back(v(0));
    logging_u1.push_back(v(1));
    logging_u2.push_back(v(2));
    logging_u3.push_back(v(3));
  }
  plt::figure();
  plt::plot(logging_t, logging_u0);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input_0");

  plt::figure();
  plt::plot(logging_t, logging_u1);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input_1");

  plt::figure();
  plt::plot(logging_t, logging_u2);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input_2");

  plt::figure();
  plt::plot(logging_t, logging_u3);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input 3");

  plt::show();
}

void main2(const std::shared_ptr<Parameters>& params) {
  Vehicle vehicle(params);
  vehicle.set_state(0, 0, M_PI / 2.0, 0, 0, 0);

  double x, y, theta, vx, vy, thetadot;
  double t = 0;
  vehicle.get_state(x, y, theta, vx, vy, thetadot);
  std::vector<double> logging_x, logging_y, logging_theta, logging_thetadot, logging_vx, logging_vy, logging_t;
  std::vector<Eigen::VectorXd> logging_u;

  Eigen::MatrixXd ou =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(vehicle.num_input_, params->horizon);
  auto ox = Eigen::MatrixXd(vehicle.num_state_, params->horizon + 1);

  Eigen::MatrixXd A, B;
  Eigen::VectorXd C(vehicle.num_state_);
  Eigen::VectorXd state;

  Eigen::MatrixXd xref(vehicle.num_state_, params->horizon + 1);
  while (t < params->max_time) {
    LOG(INFO) << "-----------------------------------------------------------------------------------------------------"
                 "-------------------------";
    for (int i = 0; i < ou.cols(); i++) {
      ou(0, i) = +0.07;
      ou(1, i) = -0.07;
      ou(2, i) = -0.07;
      ou(3, i) = +0.07;
    }
    state = vehicle.get_state();
    vehicle.get_linear_matrix(state(3), state(4), state(2), A, B, C);
    std::cout << "A" << std::endl;
    std::cout << A << std::endl;
    std::cout << "B" << std::endl;
    std::cout << B << std::endl;
    std::cout << "C" << std::endl;
    std::cout << C << std::endl;
    LOG(INFO) << "-- from state:  ----";
    Eigen::IOFormat stateformat(4, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    LOG(INFO) << state.format(stateformat);

    Eigen::MatrixXd motion = vehicle.predict_motion(vehicle.get_state(), ou, xref);
    std::vector<double> mpc_x, mpc_y;
    for (int i = 0; i < motion.cols(); i++) {
      mpc_x.push_back(motion(0, i));
      mpc_y.push_back(motion(1, i));
    }
      
    plt::plot(mpc_x, mpc_y, "xr");


    vehicle.set_state(vehicle.next_state(vehicle.get_state(), ou.col(0)));

    LOG(INFO) << "-- to state:  ----";
    state = vehicle.get_state();
    LOG(INFO) << state.format(stateformat);
    std::cout << "----" << std::endl;

    vehicle.get_state(x, y, theta, vx, vy, thetadot);
    logging_x.push_back(x);
    logging_y.push_back(y);
    logging_theta.push_back(theta);
    logging_vx.push_back(vx);
    logging_vy.push_back(vy);
    logging_thetadot.push_back(thetadot);
    logging_u.push_back(ou.col(0));
    logging_t.push_back(t);

    plt::plot(logging_x, logging_y, "ob");
    vehicle.plot();

    plt::axis("equal");
    plt::grid(true);
    plt::xlim(-5 + x, 5 + x);
    plt::ylim(-5 + y, 5 + y);
    std::stringstream ss;
    ss << "time[s] " << t << " | speed[m/s] " << std::hypot(vx, vy);
    plt::title(ss.str());
    plt::pause(0.0001);
    plt::clf();

    t += params->dt;
    // if (t == params->dt) {
    // std::cin.get();
    //}
    std::cin.get();
  }

  plt::figure();
  plt::plot(logging_t, logging_vx, "-r");
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("Speed(x) [m/s]");

  plt::figure();
  plt::plot(logging_t, logging_vy, "-r");
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("Speed(y) [m/s]");

  std::vector<double> logging_u0, logging_u1, logging_u2, logging_u3;
  for (auto&& v : logging_u) {
    logging_u0.push_back(v(0));
    logging_u1.push_back(v(1));
    logging_u2.push_back(v(2));
    logging_u3.push_back(v(3));
  }
  plt::figure();
  plt::plot(logging_t, logging_u0);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input_0");

  plt::figure();
  plt::plot(logging_t, logging_u1);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input_1");

  plt::figure();
  plt::plot(logging_t, logging_u2);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input_2");

  plt::figure();
  plt::plot(logging_t, logging_u3);
  plt::grid(true);
  plt::xlabel("Time [s]");
  plt::ylabel("input 3");

  plt::show();
}

DEFINE_string(parameters_path, "parameters_path", "the file path to params.json");
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

  fs::path p(FLAGS_parameters_path.c_str());
  if (fs::exists(p)) {
    std::ifstream t(p.string());
    std::stringstream buffer;
    buffer << t.rdbuf();
    picojson::value v;
    std::string err = picojson::parse(v, buffer.str());
    if (!err.empty()) {
      LOG(ERROR) << err;
      exit(1);
    }
    params->load_json(v);
  } else {
    exit(1);
  }

  params->print(LOG(INFO));
  std::cout << "press <Enter> to start..." << std::endl;
  std::cin.get();
  main1(params);
  return 0;
}
