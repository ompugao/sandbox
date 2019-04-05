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

// https://gist.githubusercontent.com/javidcf/25066cf85e71105d57b6/raw/c8d037ace7f83b4bcda09f2d6e22066f60466fdf/pseudoinverse.cpp
#include <Eigen/Dense>

template <class MatT>
MatT pseudoinverse(
    const MatT& mat, typename MatT::Scalar tolerance = typename MatT::Scalar{1e-4})  // choose appropriately
{
  typedef typename MatT::Scalar Scalar;
  auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& singularValues = svd.singularValues();
  MatT singularValuesInv(mat.cols(), mat.rows());
  singularValuesInv.setZero();
  for (unsigned int i = 0; i < singularValues.size(); ++i) {
    if (singularValues(i) > tolerance) {
      singularValuesInv(i, i) = Scalar{1} / singularValues(i);
    } else {
      singularValuesInv(i, i) = Scalar{0};
    }
  }
  return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}


class NonLinearMPC {
public:
  NonLinearMPC(const Vehicle& v, const std::shared_ptr<Parameters> params) {
    vehicle_ = &v;
    params_ = params;
    U_ = Eigen::MatrixXd::Zero((vehicle_->num_input_ + num_dummy_inputs_ + num_constraints_), params_->horizon);
    U_.middleRows(vehicle_->num_input_, num_dummy_inputs_).fill(0.49);
    U_.middleRows(vehicle_->num_input_+num_dummy_inputs_, num_constraints_).fill(0.011);
    //U_.block(vehicle_->num_input_, 0, num_dummy_inputs_, params_->horizon) = Eigen::MatrixXd::Ones(num_dummy_inputs_, params_->horizon)*0.49;
    //U_.block(vehicle_->num_input_+num_dummy_inputs_, 0, num_constraints_, params_->horizon) = Eigen::MatrixXd::Ones(num_constraints_, params_->horizon)*0.011;
    //lambdas_ = Eigen::MatrixXd::Zero((vehicle_->num_input_ + num_constraints_), params_->horizon);
    // mu_ = Eigen::VectorXd::Ones(params_->horizon);
  }

  virtual ~NonLinearMPC() {}

  Eigen::MatrixXd compute_input(const double t /*[s] */, const Eigen::MatrixXd& xref, std::vector<Obstacle*>& obstacles) {
    // this dt: sampling time
    double dt = params_->tf * (1.0 - std::exp(-params_->alpha * t)) / params_->horizon;

    // clang-format: off
    Eigen::MatrixXd xbar = vehicle_->predict_motion(vehicle_->get_state(), U_.topRows(vehicle_->num_input_), xref, dt);
    Eigen::MatrixXd lambdas = predict_adjoint_variables(xbar, xref, obstacles, U_, dt);
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    Eigen::MatrixXd F = compute_F(xbar, lambdas, U_);
    fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    Eigen::MatrixXd next_state = vehicle_->compute_next_state_linear(vehicle_->get_state(), U_.topLeftCorner(vehicle_->num_input_, 1), params_->ht);
    Eigen::MatrixXd xbar_next = vehicle_->predict_motion(next_state, U_.topRows(vehicle_->num_input_), xref, dt);
    Eigen::MatrixXd lambdas_next = predict_adjoint_variables(xbar_next, xref, obstacles, U_, dt);
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    Eigen::MatrixXd F_next = compute_F(xbar_next, lambdas_next, U_);
    fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
        

    Eigen::MatrixXd du = U_ * params_->ht; // XXX: 本当？
    Eigen::MatrixXd next_U = (U_ + du);
    Eigen::MatrixXd xbar_next2 = vehicle_->predict_motion(next_state, next_U.topRows(vehicle_->num_input_), xref, dt);
    Eigen::MatrixXd lambdas_next2 = predict_adjoint_variables(xbar_next2, xref, obstacles, next_U, dt);
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    Eigen::MatrixXd F_next2 = compute_F(xbar_next2, lambdas_next2, next_U);
    fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    Eigen::MatrixXd b = - params_->zeta * F - ((F_next - F) / params_->ht) - ((F_next2 - F_next) / params_->ht);
    auto b_vec = Eigen::Map<Eigen::VectorXd>(b.data(), b.size());
    // clang-format: on

    // now, starts c/gmres
    double b_norm = b_vec.norm();
    
    int n = (vehicle_->num_input_ + num_dummy_inputs_ + num_constraints_) * params_->horizon;
    Eigen::MatrixXd bases(n, n+1);
    bases.col(0) = b_vec.normalized();
    Eigen::MatrixXd hs = Eigen::MatrixXd::Zero(n+1, n+1);
    Eigen::VectorXd e = Eigen::VectorXd::Zero(n+1);
    e(0) = 1.0;

    Eigen::VectorXd ys_pre = Eigen::VectorXd::Zero(n+1);

    auto U_vec = Eigen::Map<Eigen::VectorXd>(U_.data(), U_.size());
    Eigen::VectorXd du_new(n);
    du_new = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; i++) {
      Eigen::VectorXd du = bases.col(i) * params_->ht;
      Eigen::VectorXd next_U = (U_vec + du);
      Eigen::MatrixXd next_U_block = Eigen::Map<Eigen::MatrixXd>(next_U.data(), vehicle_->num_input_+num_dummy_inputs_+num_constraints_, params_->horizon);
      Eigen::MatrixXd next_u_input_block = next_U_block.block(0, 0, vehicle_->num_input_, params_->horizon);

      Eigen::MatrixXd xbar_next3 = vehicle_->predict_motion(next_state, next_u_input_block, xref, dt);
      Eigen::MatrixXd lambdas_next3 = predict_adjoint_variables(xbar_next2, xref, obstacles, next_U_block, dt);
      feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
      Eigen::MatrixXd F_next3 = compute_F(xbar_next3, lambdas_next3, next_U_block);
      fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

      Eigen::MatrixXd diff_F = (F_next2 - F_next) / params_->ht;
      auto Av = Eigen::Map<Eigen::VectorXd>(diff_F.data(), diff_F.size());
      Eigen::VectorXd sum_Av = Eigen::VectorXd::Zero(n);
      for (int j = 0; j < i; j++) {
        hs(j, i) =  Av.dot(bases.col(j));
        sum_Av += hs(j, i) * bases.col(j);
      }

      Eigen::VectorXd v_estimated = Av - sum_Av;
      hs(i+1, i) = v_estimated.norm();
      bases.col(i+1) = v_estimated / hs(i+1, i);

      if (i == 0) {
        continue;
      }
      Eigen::MatrixXd hsblock = hs.block(0, 0, i+1, i);
      // auto ys = pseudoinverse<Eigen::MatrixXd>(hsblock) * b_norm * e.segment(0, i+1);
      Eigen::VectorXd ys = pseudoinverse(hsblock) * b_norm * e.segment(0, i+1);
      // auto ys = hs.block(0, 0, i+1, i).completeOrthogonalDecomposition().pseudoInverse() * b_norm * e.segment(0, i+1);

      Eigen::VectorXd val = b_norm * e.segment(0, i+1) - hs.block(0, 0, i+1, i) * ys.segment(0, i);

      LOG(INFO) << "---------------------";
      LOG(INFO) << "i: " << i;
      Eigen::IOFormat format(4, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
      LOG(INFO) << "val: "<< val.format(format);
      LOG(INFO) << "val norm: "<< val.squaredNorm();

      if (val.squaredNorm() < params_->iteration_threshold || i == n - 1) {
        du_new = du + bases.block(0, 0, bases.rows(), i-1) * ys_pre.segment(0, i-1);
        break;
      }
      ys_pre = ys;
    }

    U_ += Eigen::Map<Eigen::MatrixXd>(du_new.data(), (vehicle_->num_input_ + num_dummy_inputs_ + num_constraints_), params_->horizon);

    Eigen::MatrixXd xbar_new = vehicle_->predict_motion(vehicle_->get_state(), U_.block(0, 0, vehicle_->num_input_, params_->horizon), xref, dt);
    Eigen::MatrixXd lambdas_new = predict_adjoint_variables(xbar, xref, obstacles, U_, dt);
    Eigen::MatrixXd F_new = compute_F(xbar, lambdas, U_);

    LOG(INFO) << "new F: " << F_new.norm();
    
    Eigen::MatrixXd new_u = U_.block(0, 0, vehicle_->num_input_, U_.cols());
    return std::move(new_u);
  }

  Eigen::MatrixXd predict_adjoint_variables(const Eigen::MatrixXd& states, const Eigen::MatrixXd& xref, const std::vector<Obstacle*>& obstacles, const Eigen::MatrixXd& U, double dt) {
    const int N = params_->horizon;
    Eigen::MatrixXd lambdas(vehicle_->num_state_, params_->horizon);
    lambdas.col(N-1) = (params_->Qf * states.col(N-1)).transpose();
    for (int i = N-1; i >= 1; i--) {
      lambdas.col(i-1) = compute_previous_lambda(states.col(i), xref.col(i), lambdas.col(i), obstacles, U.col(i), dt);
    }
    return std::move(lambdas);
  }

  Eigen::VectorXd compute_previous_lambda(const Eigen::VectorXd& state, const Eigen::VectorXd& state_ref,
                                          const Eigen::VectorXd& lambda, const std::vector<Obstacle*>& obstacles,
                                          const Eigen::VectorXd& _u, double dt) {
    const auto& u = _u.segment(0, vehicle_->num_input_);
    const auto& dummy = _u.segment(vehicle_->num_input_, num_dummy_inputs_);
    const auto& mu = _u.segment(vehicle_->num_input_ + num_dummy_inputs_, num_constraints_);

    auto tmp = lambda;
    tmp -= params_->Q * (state - state_ref) * dt;
    for (auto&& o : obstacles) {
      Eigen::VectorXd pos;
      pos << o->x(), o->y(), 0.0, 0.0, 0.0, 0.0;
      auto diff = state - pos;
      double d = params_->inflation_radius + o->r();
      double norm = diff.norm();
      if (norm < d) {
        tmp -= o->a() * ((d - norm) / norm) * diff * dt;
      }
    }
    Eigen::MatrixXd A, B;
    Eigen::VectorXd C;
    vehicle_->get_linear_matrix_diff(state, A, B, C);
    tmp -= lambda.transpose() * A * dt;
    // clang-format off
    Eigen::MatrixXd dCdx(num_constraints_, vehicle_->num_state_);
    dCdx << 0, 0, 0, 0,          0,          0,
            0, 0, 0, 0,          0,          0,
            0, 0, 0, 0,          0,          0,
            0, 0, 0, 0,          0,          0,
            0, 0, 0, 2*state(4), 2*state(5), 0;
    // clang-format on
    tmp -= mu.transpose() * dCdx * dt;
    return tmp;
  }

  Eigen::MatrixXd compute_F(const Eigen::MatrixXd& states, const Eigen::MatrixXd& lambdas, const Eigen::MatrixXd& U) {
    int num_all_inputs = vehicle_->num_input_ + num_dummy_inputs_;
    int step = num_all_inputs + num_constraints_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> F(step, params_->horizon);
    F.fill(0);
    int head = 0;
    for (int i = 0; i < params_->horizon; i++) {
      // const auto& u = Eigen::Map<const Eigen::VectorXd>(U.block(head, 1, vehicle_->num_input_, 1));
      // const auto& dummy = Eigen::Map<const Eigen::VectorXd(U.block(head + vehicle_->num_input_, 1, num_dummy_inputs_,
      // 1));
      const Eigen::MatrixXd& u = U.block(0, i, vehicle_->num_input_, 1);
      const Eigen::MatrixXd& dummy = U.block(vehicle_->num_input_, i, num_dummy_inputs_, 1);
      const Eigen::MatrixXd& mu = U.block(num_all_inputs, i, num_constraints_, 1);
      // compute dH/dU
      // dL/dU
      F.block(0, i, vehicle_->num_input_, 1) += (params_->R * u);
      F.block(vehicle_->num_input_, i, num_dummy_inputs_, 1) -= Eigen::MatrixXd::Ones(num_dummy_inputs_, 1) * 0.001;
      // df/dU
      F.block(0, i, vehicle_->num_input_, 1) += (lambdas.col(i).transpose() * vehicle_->B_).transpose();


      // clang-format off
      Eigen::MatrixXd dCdU(num_constraints_, num_all_inputs);
      dCdU << 2*u(0, 0),         0,         0,         0, 2*dummy(0, 0),             0,             0,             0,              0,
                      0, 2*u(1, 0) ,        0,         0,             0, 2*dummy(1, 0),             0,             0,              0,
                      0,         0, 2*u(2, 0),         0,             0,             0, 2*dummy(2, 0),             0,              0,
                      0,         0,         0, 2*u(3, 0),             0,             0,             0, 2*dummy(3, 0),              0,
                      0,         0,         0,         0,             0,             0,             0,             0, 2*dummy(4,   0);
      // clang-format on
      // dC/dU
      F.block(0, i, num_all_inputs, 1) += (mu.transpose() * dCdU).transpose();

      // constraints
      // clang-format off
      const double& max_motor_torque = vehicle_->vparams_.max_motor_torque;
      const double max_motor_torque_sqrn = max_motor_torque * max_motor_torque;
      F(num_all_inputs + 0, i) += u(0, 0)*u(0, 0) + dummy(0, 0)*dummy(0, 0) - max_motor_torque_sqrn;
      F(num_all_inputs + 1, i) += u(1, 0)*u(1, 0) + dummy(1, 0)*dummy(1, 0) - max_motor_torque_sqrn;
      F(num_all_inputs + 2, i) += u(2, 0)*u(2, 0) + dummy(2, 0)*dummy(2, 0) - max_motor_torque_sqrn;
      F(num_all_inputs + 3, i) += u(3, 0)*u(3, 0) + dummy(3, 0)*dummy(3, 0) - max_motor_torque_sqrn;
      // clang-format on
      const auto& state = states.col(i);
      const double& vx = state(3);
      const double& vy = state(4);

      const double max_speed_sqrn = vehicle_->vparams_.max_speed * vehicle_->vparams_.max_speed;
      F(num_all_inputs + 4, i) += vx * vx + vy * vy + dummy(4, 0) * dummy(4, 0) - max_speed_sqrn;

      //head += step;
    }
    return std::move(F);
  }

  const Vehicle* vehicle_;
  const int num_constraints_ = 5;
  const int num_dummy_inputs_ = 5;  // == num_constraints_; ?
  std::shared_ptr<Parameters> params_;

  Eigen::MatrixXd U_;
  //Eigen::MatrixXd lambdas_;  // (i, ht*j)
  // Eigen::VectorXd mu_;
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
