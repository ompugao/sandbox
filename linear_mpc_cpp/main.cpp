#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#include "csv.h"

class ScopedTime {
public:
  ScopedTime(const std::string& msg = "") : msg_(msg) {
    start_ = std::chrono::system_clock::now();
  }
  void lap(const std::string& msg) {
    const auto duration_time = std::chrono::system_clock::now() - start_;
    const auto duration_ns = std::chrono::duration_cast<std::chrono::milliseconds>(duration_time);
    std::cerr << "[" << duration_ns.count() << " ns] " << msg << std::endl;
  }
  virtual ~ScopedTime() {
    this->lap(msg_);
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

class Parameters {
public:
  Parameters() {
    R.diagonal() << 0.1, 0.1;
    Rd.diagonal() << 0.001, 0.001;
    Q.diagonal() << 1, 1, 0.1, 0.0, 0.5;
    Qf = Q;
  }
  virtual ~Parameters() {}
  Eigen::Matrix2d R;
  Eigen::Matrix2d Rd;
  Eigen::Matrix<double, 5, 5> Q;
  Eigen::Matrix<double, 5, 5> Qf;

  const double goal_distance_tolerance = 1.5;
  const double stop_speed = 0.5;
  const double max_time = 100;

  double target_speed = 3.0;
  int n_indices_search = 10;
  double dt = 0.2;

  const double dl = 0.2;  // course tick

  int horizon = 5;  // horizon length
  std::string course_path;

  int max_iterations = 10;
  double du_th = 0.1;
};

class VehicleParameters {
public:
  VehicleParameters() {}
  virtual ~VehicleParameters() {}

  const double length = 2.0;
  const double width = 2.0;
  // const double wheel_len = 0.3;
  // const double wheel_width = 0.2;
  const double wheel_radius = 0.1;
  const double wheel_mass = 0.3;
  const double Iw = 1.0 / 2 * wheel_mass * wheel_radius * wheel_radius;
  const double mass = 5.0;
  const double Iv = 1.0 / 2 * mass * 0.5 * 0.5;

  const double input_gain = 1.0;
  const double damping_gain = 0.00001;

  const double max_speed = 5.0;
  const double min_speed = -2.0;

  const double max_motor_torque = 3.1415;
  const double max_motor_torque_vel = 1.57;
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
    io::CSVReader<4> in(path);
    in.read_header(io::ignore_extra_column, "cx", "cy", "ctheta", "ck");
    double cx, cy, ctheta, ck;
    while (in.read_row(cx, cy, ctheta, ck)) {
      waypoints.emplace_back(WayPoint(cx, cy, ctheta));
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

  std::vector<WayPoint> waypoints;
};

class Vehicle {
public:
  Vehicle(const std::shared_ptr<Parameters>& env_params) {
    params_ = env_params;
    state_ = Eigen::VectorXd(5);
  }
  virtual ~Vehicle() {}
  void get_linear_matrix(double v, double theta, Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::VectorXd& C) const {
    const double l = vparams_.width / 2.0;
    A = Eigen::MatrixXd::Identity(num_state_, num_state_);
    B = Eigen::MatrixXd::Zero(num_state_, num_input_);
    C = Eigen::VectorXd::Zero(num_state_);

    A(0, 2) += params_->dt * (-std::sin(theta)) * v;
    A(0, 4) += params_->dt * std::cos(theta);
    A(1, 2) += params_->dt * std::cos(theta) * v;
    A(1, 4) += params_->dt * std::sin(theta);
    A(2, 3) += params_->dt * 1.0;
    A(3, 3) += params_->dt * (-2.0 * vparams_.damping_gain * l * l) /
               (vparams_.wheel_radius * vparams_.Iv + l * l * 2.0 * vparams_.Iw);
    A(4, 4) += params_->dt * (-2.0 * vparams_.damping_gain) /
               (vparams_.wheel_radius * vparams_.wheel_radius * vparams_.mass + 2.0 * vparams_.Iw);

    const double b1 = vparams_.wheel_radius * vparams_.input_gain * l /
                      (vparams_.wheel_radius * vparams_.wheel_radius * vparams_.Iv + l * l * 2 * vparams_.Iw);
    const double b2 = vparams_.wheel_radius * vparams_.input_gain /
                      (vparams_.wheel_radius * vparams_.wheel_radius * vparams_.mass + 2 * vparams_.Iw);
    B(3, 0) += params_->dt * b1;
    B(3, 1) += params_->dt * (-b1);
    B(4, 0) += params_->dt * b2;
    B(4, 1) += params_->dt * b2;

    C(0) = params_->dt * v * std::sin(theta) * theta;
    C(1) = -params_->dt * v * std::cos(theta) * theta;

    return;
  }

  void plot() {
    double radius = vparams_.width / 2.0;
    const int n = 200;
    std::vector<double> x(n + 1), y(n + 1);
    for (int i = 0; i < n; i++) {
      x[i] = radius * std::cos(i / 2 * M_PI);
      y[i] = radius * std::sin(i / 2 * M_PI);
    }
    x[n] = 1;
    y[n] = 0;
    plt::plot(x, y, "r-");
    plt::plot({state_(0), state_(0) + std::cos(state_(2)) * radius},
              {state_(1), state_(1) + std::sin(state_(2)) * radius}, "r-");
    plt::plot({state_(0)}, {state_(1)}, "*g");
  }

  Eigen::VectorXd next_state(const Eigen::VectorXd& state, Eigen::Vector2d u) const {
    Eigen::Vector2d u_copy = u;
    u_copy(0) = std::clamp(u(0), -vparams_.max_motor_torque, vparams_.max_motor_torque);
    u_copy(1) = std::clamp(u(1), -vparams_.max_motor_torque, vparams_.max_motor_torque);

    Eigen::MatrixXd A, B;
    Eigen::VectorXd C(5);
    get_linear_matrix(state(4), state(2), A, B, C);

    Eigen::VectorXd new_state = A * state + B * u_copy + C;
    double v = std::clamp(new_state(4), vparams_.min_speed, vparams_.max_speed);
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

  void set_state(double x, double y, double theta, double thetadot, double v) {
    state_(0) = x;
    state_(1) = y;
    state_(2) = theta;
    state_(3) = thetadot;
    state_(4) = v;
  }

  void get_state(double& x, double& y, double& theta, double& thetadot, double& v) {
    x = state_(0);
    y = state_(1);
    theta = state_(2);
    thetadot = state_(3);
    v = state_(4);
  }

  bool is_arrived(const Course& c, const WayPoint& w, int iw) {
    double d = std::hypot(state_(0) - w.x, state_(1) - w.y);
    bool closeenough = (d < params_->goal_distance_tolerance) && (c.size() == iw + 1);
    bool stopped = std::abs(state_(4)) < params_->stop_speed;
    return closeenough && stopped;
  }

  Eigen::VectorXd state_;
  VehicleParameters vparams_;

  std::shared_ptr<Parameters> params_;
  const int num_state_ = 5;
  const int num_input_ = 2;
};

std::pair<Eigen::MatrixXd, int> calc_ref_trajectory(const Vehicle& v, const Course& c,
                                                    const std::shared_ptr<Parameters>& params, int pind) {
  auto xref = Eigen::MatrixXd(v.num_state_, params->horizon+1);
  int i_closest = v.find_nearest_index(c, pind);

  if (i_closest <= pind) {
    i_closest = pind;  // move forward!
  }
  xref(0, 0) = c.waypoints[i_closest].x;
  xref(1, 0) = c.waypoints[i_closest].y;
  xref(2, 0) = c.waypoints[i_closest].theta;
  xref(3, 0) = 0;
  xref(4, 0) = c.waypoints[i_closest].speed;

  double travel = 0.0;
  for (int i = 0; i < params->horizon + 1; i++) {
    travel += std::abs(v.state_(4)) * params->dt;
    int dind = static_cast<int>(travel / params->dl);
    int j = i_closest + dind;
    if (j >= c.size()) {
      j = c.size() - 1;
    }
    const WayPoint& w = c.waypoints[j];
    xref(0, i) = w.x;
    xref(1, i) = w.y;
    xref(2, i) = w.theta;
    xref(3, i) = 0;
    xref(4, i) = w.speed;
  }
  return std::move(std::make_pair(xref, i_closest));
}

bool linear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar,
                        const Eigen::VectorXd& x0, const std::shared_ptr<Parameters>& params, Eigen::MatrixXd& ou);
void iterative_linear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, Eigen::MatrixXd& ou,
                                  const std::shared_ptr<Parameters>& params) {
  int i = 0;
  for (; i < params->max_iterations; i++) {
    auto xbar = v.predict_motion(v.state_, ou, xref);
    Eigen::MatrixXd prev_ou = ou;
    bool sol_found = linear_mpc_control(v, xref, xbar, v.state_, params, ou);
    if (!sol_found) {
      LOG(WARNING) << "Solution not found (iter:" << i << ")";
      continue;
    }
    double udiff = (ou - prev_ou).cwiseAbs().sum();
    //double udiff = 0;
    if (udiff < params->du_th) {
      LOG(INFO) << "udiff: " << udiff;
      break;
    }
  }
  LOG(INFO) << "-- iteration : " << i;
}

// static int kStride = 10;
class MPCCostFunctor {
public:
  // typedef ceres::DynamicAutoDiffCostFunction<MPCCostFunctor, kStride> MPCCostFunction;
  // typedef ceres::DynamicCostFunction MPCCostFunction;

  MPCCostFunctor(const Vehicle& vehicle, const std::shared_ptr<Parameters>& params, const Eigen::VectorXd& x0,
                 const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar) {
    vehicle_ = &vehicle;
    params_ = params;
    x0_ = x0;
    // x_ = Eigen::MatrixXd::Zero(vehicle_->num_state_, params->horizon + 1);
    xref_ = xref;
    xbar_ = xbar;
    // xdiff_ = Eigen::MatrixXd::Zero(vehicle_->num_state_, params->horizon + 1);
    // udiff_ = Eigen::MatrixXd::Zero(vehicle_->num_input_, params->horizon + 1);
    // deriv_ = Eigen::MatrixXd::Zero(vehicle_->num_input_, params->horizon);
  }
  virtual ~MPCCostFunctor() {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> u =
    const auto u = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
        parameters[0], vehicle_->num_input_, params_->horizon);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x_(vehicle_->num_state_, params_->horizon + 1);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> xdiff_(vehicle_->num_state_, params_->horizon + 1),
        udiff_(vehicle_->num_input_, params_->horizon + 1), deriv_(vehicle_->num_input_, params_->horizon);

    x_.col(0) = x0_.cast<T>();
    for (int t = 0; t < params_->horizon; t++) {
      vehicle_->get_linear_matrix(xbar_.cast<double>()(4, t), xbar_.cast<double>()(2, t), A_, B_, C_);
      x_.col(t + 1) = A_.cast<T>() * x_.col(t) + B_.cast<T>() * u + C_;
      if (t != 0) {
        xdiff_.col(t) = x_.col(t) - xref_.col(t);
      }
      if (t < params_->horizon - 1) {
        udiff_.col(t) = u.col(t + 1) - u.col(t);
      }
    }
    residuals[0] =
        ((xdiff_.block(0, 0, x0_.rows(), params_->horizon).cwiseProduct(params_->Q.cast<T>() * xdiff_.block(0, 0, x0_.rows(), params_->horizon))).sum() + 
         (udiff_.cwiseProduct(params_->Rd.cast<T>() * udiff_)).sum() +
         (xdiff_.col(params_->horizon).cwiseProduct(params_->Qf.cast<T>() * xdiff_.col(params_->horizon))).sum());


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
                                     std::vector<double*>& parameter_blocks, const Eigen::VectorXd& x0,
                                     const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar) {
    MPCCostFunctor* costfunctor = new MPCCostFunctor(vehicle, params, x0, xref, xbar);
    ceres::DynamicCostFunction* costfn = new ceres::DynamicAutoDiffCostFunction<MPCCostFunctor, 5>(costfunctor);
    costfn->AddParameterBlock(parameter_blocks.size());
    costfn->SetNumResiduals(1);
    return costfn;
  }

private:
  std::shared_ptr<Parameters> params_;
  const Vehicle* vehicle_;
  // memory allocations
  mutable Eigen::VectorXd x0_;
  mutable Eigen::MatrixXd A_, B_;
  mutable Eigen::VectorXd C_;
  mutable Eigen::MatrixXd xref_, xbar_;

  // mutable Eigen::MatrixXd A_, B_,Bd_, Q_, Q_final_,  R_, R_delta_, disturbance_, insecure_, u_ss_, x_ss_, x0_,
  // u_prev_, x_states, u, deriv_wrt_u, u_past, lambdas_x, lambdas_u,lambdas_u_ref, u_horizon, u_current;
};

bool linear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar,
                        const Eigen::VectorXd& x0, const std::shared_ptr<Parameters>& params, Eigen::MatrixXd& ou) {
  ceres::Problem prob;

  // ceres::CostFunction* costfn = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  // prob.AddResidualBlock(costfn, NULL, &x);
  // prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x1, &x2);
  // prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x3, &x4);
  // prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x2, &x3);
  // prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x1, &x4);

  Eigen::MatrixXd u =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(v.num_input_, params->horizon);
  std::vector<double*> parameter_blocks;
  // for (int i = 0; i < u.cols(); i++) {
  //  parameter_blocks.push_back(&u.data()[u.rows()*i]);
  //}
  parameter_blocks.push_back(u.data());
  ceres::CostFunction* costfn = MPCCostFunctor::Create(v, params, parameter_blocks, x0, xref, xbar);
  prob.AddResidualBlock(costfn, NULL, parameter_blocks);
  for (int i = 0; i < u.size(); i++) {
    prob.SetParameterLowerBound(parameter_blocks[0], i, v.vparams_.min_speed);
    prob.SetParameterUpperBound(parameter_blocks[0], i, v.vparams_.max_speed);
  }

  ceres::Solver::Options options;
  // LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer, &options.minimizer_type))
  //<< "Invalid minimizer: " << FLAGS_minimizer << ", valid options are: trust_region and line_search.";
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;

  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::CGNR;

  ceres::Solver::Summary summary;

  {
    ScopedTime time("optimization");
    ceres::Solve(options, &prob, &summary);
  }

  LOG(INFO) << summary.FullReport() << std::endl;

  ou = u;
  return summary.IsSolutionUsable();
}

void main1(const std::shared_ptr<Parameters>& params) {
  Course c;
  c.load_csv(params->course_path);
  c.set_speed_profile(params->target_speed);
  c.smooth_theta();

  Vehicle vehicle(params);
  const WayPoint& w = c.waypoints[0];
  vehicle.set_state(w.x, w.y, normalize_angle(w.theta + M_PI / 6.0), 0, 0);

  std::vector<double> logging_x, logging_y, logging_theta, logging_thetadot, logging_v, logging_t;
  std::vector<Eigen::Vector2d> logging_u;

  double x, y, theta, thetadot, v;
  double t = 0;
  vehicle.get_state(x, y, theta, thetadot, v);
  logging_x.push_back(x);
  logging_y.push_back(y);
  logging_theta.push_back(theta);
  logging_thetadot.push_back(thetadot);
  logging_v.push_back(v);
  logging_u.emplace_back(Eigen::Vector2d::Zero());
  logging_t.push_back(t);

  int target_index = vehicle.find_nearest_index(c, 0);

  auto ou = Eigen::MatrixXd(vehicle.num_input_, params->horizon);
  auto ox = Eigen::MatrixXd(vehicle.num_state_, params->horizon + 1);

  Eigen::MatrixXd xref;
  while (t < params->max_time) {
    std::tie(xref, target_index) = calc_ref_trajectory(vehicle, c, params, target_index);
    iterative_linear_mpc_control(vehicle, xref, ou, params);
  }
}

DEFINE_string(course_path, "course_path", "the file path to path.csv");
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  params->course_path = FLAGS_course_path;
  main1(params);
  return 0;
}

// class QuadratiCostFunction : public ceres::SizedCostFunction<1, 1> {
// public:
//   QuadratiCostFunction() {}
//   virtual ~QuadratiCostFunction() {}
//   virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
//     const double x = parameters[0][0];
//     residuals[0] = 10 - x;
//
//     if (jacobians != nullptr && jacobians[0] != nullptr) {
//       jacobians[0][0] = -1.0;
//     }
//     return true;
//   }
// };
//
// struct F4 {
//   template <typename T>
//   bool operator()(const T* const x1, const T* const x4, T* residual) const {
//     residual[0] = T(std::sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
//     return true;
//   }
// };
//
// struct F3 {
//   template <typename T>
//   bool operator()(const T* const x2, const T* const x3, T* residual) const {
//     residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
//     return true;
//   }
// };
//
// struct F2 {
//   template <typename T>
//   bool operator()(const T* const x3, const T* const x4, T* residual) const {
//     residual[0] = T(std::sqrt(5.0)) * (x3[0] - x4[0]);
//     return true;
//   }
// };
//
// struct F1 {
//   template <typename T>
//   bool operator()(const T* const x1, const T* const x2, T* residual) const {
//     residual[0] = x1[0] + 10.0 * x2[0];
//     return true;
//   }
// };
// DEFINE_string(minimizer, "trust_region", "Minimizer type to use, choices are: line_search & trust_region");

//    gflags::ParseCommandLineFlags(&argc, &argv, true);
//    google::InitGoogleLogging(argv[0]);
//
//    double x1 = 3.0;
//    double x2 = -1.0;
//    double x3 = 0.0;
//    double x4 = 1.0;
//
//    ceres::Problem prob;
//
//    // ceres::CostFunction* costfn = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
//    // ceres::CostFunction* costfn = new QuadratiCostFunction();
//    // prob.AddResidualBlock(costfn, NULL, &x);
//    prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x1, &x2);
//    prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x3, &x4);
//    prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x2, &x3);
//    prob.AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x1, &x4);
//
//    ceres::Solver::Options options;
//    LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer, &options.minimizer_type))
//        << "Invalid minimizer: " << FLAGS_minimizer << ", valid options are: trust_region and line_search.";
//
//    options.minimizer_progress_to_stdout = true;
//    options.max_num_iterations = 100;
//    options.linear_solver_type = ceres::CGNR;
//
//    ceres::Solver::Summary summary;
//
//    {
//        ScopedTime time("optimization");
//        ceres::Solve(options, &prob, &summary);
//    }
//
//    std::cout << summary.FullReport() << std::endl;
//    std::cout << "Final x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 << ", x4 = " << x4 << "\n";

