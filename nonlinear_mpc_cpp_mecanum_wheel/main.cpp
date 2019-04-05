#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#include <fstream>

#include <fenv.h>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include "picojson.h"

#include "scopedtime.h"
#include "obstacle.h"
#include "parameters.h"
#include "course.h"
#include "vehicle.h"
#include "utils.h"

#include "nonlinearmpc.h"

class ScopedFloatingPointNumberChecker {
  public:
    ScopedFloatingPointNumberChecker(int except = FE_INEXACT) :except_(except) {
      feenableexcept(except_);
    }
    virtual ~ScopedFloatingPointNumberChecker() {
      fedisableexcept(except_);
    }
    int except_;
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
      vehicle_->get_linear_matrix(xbar_.cast<double>().col(t), A_, B_, C_);
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
        num_residuals += (params->horizon + 1);
      }
    } else {
      for (auto&& obj : params->obstacles) {
        if (x - params->local_area_x / 2.0 < obj.x() && obj.x() < x + params->local_area_x / 2.0 &&
            y - params->local_area_y / 2.0 < obj.y() && obj.y() < y + params->local_area_y / 2.0) {
          surrounding_obstacles.push_back(&obj);
          num_residuals += (params->horizon + 1);
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

bool nonlinear_mpc_control(const Vehicle& v, const Eigen::MatrixXd& xref, const Eigen::MatrixXd& xbar,
                           const Eigen::VectorXd& x0, const std::shared_ptr<Parameters>& params, Eigen::MatrixXd& ou) {return true;}

void plot_obstacles(const std::shared_ptr<Parameters>& params) {
  for (auto&& o : params->obstacles) {
    const int n = 100;
    std::vector<double> x(n + 1), y(n + 1);
    for (int i = 0; i < n + 1; i++) {
      x[i] = o.x() + o.r() * std::cos(i * 1.0 / (2 * M_PI));
      y[i] = o.y() + o.r() * std::sin(i * 1.0 / (2 * M_PI));
    }
    if (params->visualize) {
      plt::plot(x, y, "g-");
    }
  }
}

void main1(const std::shared_ptr<Parameters>& params) {
  Course c;
  c.load_csv(params->course_path);
  c.set_speed_profile(params->target_speed);
  c.smooth_theta();

  Vehicle vehicle(params);
  const WayPoint& w = c.waypoints[0];
  // vehicle.set_state(w.x, w.y - 3, normalize_angle(w.theta - 2 * M_PI / 3.0), 0, 0, 0);
  vehicle.set_state(w.x, w.y, normalize_angle(w.theta + 2 * M_PI / 3.0), 0, 0, 0);

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

  NonLinearMPC nmpc(vehicle, params);

  Eigen::MatrixXd xref;

  while (t < params->max_time) {
    t += params->dt;
    LOG(INFO) << "-----------------------------------------------------------------------------------------------------"
                 "-------------------------";
    std::tie(xref, target_index) = calc_ref_trajectory(vehicle, c, params, target_index);

    std::vector<Obstacle*> surrounding_obstacles;
    if (params->local_area_x < 0 || params->local_area_y < 0) {
      for (auto&& obj : params->obstacles) {
        surrounding_obstacles.push_back(&obj);
      }
    } else {
      for (auto&& obj : params->obstacles) {
        if (x - params->local_area_x / 2.0 < obj.x() && obj.x() < x + params->local_area_x / 2.0 &&
            y - params->local_area_y / 2.0 < obj.y() && obj.y() < y + params->local_area_y / 2.0) {
          surrounding_obstacles.push_back(&obj);
        }
      }
    }

    {
      ScopedTime st("nonlinear_mpc_control");
      ou = nmpc.compute_input(t, xref, surrounding_obstacles);
    }
    // ou(0, 0) = 0.093;
    // ou(1, 0) = 0.09;
    Eigen::MatrixXd motion = vehicle.predict_motion(vehicle.get_state(), ou, xref, params->dt);
    LOG(INFO) << "-- state:  ----";
    Eigen::IOFormat stateformat(4, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    LOG(INFO) << vehicle.get_state().format(stateformat);
    LOG(INFO) << "-- input:  ----";
    LOG(INFO) << ou.col(0).format(stateformat);
    LOG(INFO) << "-- inputs: ----";
    LOG(INFO) << ou;
    LOG(INFO) << "---------------";
    vehicle.set_state(vehicle.compute_next_state_linear(vehicle.get_state(), ou.col(0), params->dt));

    vehicle.get_state(x, y, theta, vx, vy, thetadot);
    logging_x.push_back(x);
    logging_y.push_back(y);
    logging_theta.push_back(theta);
    logging_vx.push_back(vx);
    logging_vy.push_back(vy);
    logging_thetadot.push_back(thetadot);
    logging_u.push_back(ou.col(0));
    logging_t.push_back(t);

    if (params->visualize) {

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

/*
 *    std::vector<double> ref_x, ref_y;
 *    for (int i = 0; i < xref.cols(); i++) {
 *      ref_x.push_back(xref(0, i));
 *      ref_y.push_back(xref(1, i));
 *    }
 *    plt::plot(ref_x, ref_y, "xk");
 *
 */
    plt::plot(logging_x, logging_y, "ob");
    vehicle.plot();

    plt::axis("equal");
    plt::grid(true);
    // plt::xlim(-5 + x, 5 + x);
    // plt::ylim(-5 + y, 5 + y);
    std::stringstream ss;
    ss << "time[s] " << t << " | speed[m/s] " << std::hypot(vx, vy);
    plt::title(ss.str());
    plt::pause(0.0001);
    plt::clf();
    }

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
  if (params->visualize) {

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

  Eigen::MatrixXd xref;
  while (t < params->max_time) {
    LOG(INFO) << "-----------------------------------------------------------------------------------------------------"
                 "-------------------------";
    ou(0, 0) = +0.07;
    ou(1, 0) = -0.07;
    ou(2, 0) = -0.07;
    ou(3, 0) = +0.07;

    state = vehicle.get_state();
    vehicle.get_linear_matrix(state, A, B, C);
    std::cout << "A" << std::endl;
    std::cout << A << std::endl;
    std::cout << "B" << std::endl;
    std::cout << B << std::endl;
    std::cout << "C" << std::endl;
    std::cout << C << std::endl;
    LOG(INFO) << "-- from state:  ----";
    Eigen::IOFormat stateformat(4, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    LOG(INFO) << state.format(stateformat);

    vehicle.set_state(vehicle.compute_next_state_linear(vehicle.get_state(), ou.col(0)));

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

    if (params->visualize) {
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

    }
    t += params->dt;
    // if (t == params->dt) {
    // std::cin.get();
    //}
    std::cin.get();
  }

  if (params->visualize) {
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
