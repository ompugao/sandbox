#include "nonlinearmpc.h"
#include <fenv.h>
#include <glog/logging.h>


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



NonLinearMPC::NonLinearMPC(const Vehicle& v, const std::shared_ptr<Parameters> params) {
  vehicle_ = &v;
  params_ = params;
  U_ = Eigen::MatrixXd::Zero((vehicle_->num_input_ + num_dummy_inputs_ + num_constraints_), params_->horizon);
}

void NonLinearMPC::initialize() {
  // U_.middleRows(vehicle_->num_input_, num_dummy_inputs_).fill(0.49);
  // U_.middleRows(vehicle_->num_input_+num_dummy_inputs_, num_constraints_).fill(0.011);

  //U_.block(vehicle_->num_input_, 0, num_dummy_inputs_, params_->horizon) = Eigen::MatrixXd::Ones(num_dummy_inputs_, params_->horizon)*0.49;
  //U_.block(vehicle_->num_input_+num_dummy_inputs_, 0, num_constraints_, params_->horizon) = Eigen::MatrixXd::Ones(num_constraints_, params_->horizon)*0.011;
  //lambdas_ = Eigen::MatrixXd::Zero((vehicle_->num_input_ + num_constraints_), params_->horizon);
  // mu_ = Eigen::VectorXd::Ones(params_->horizon);
}

Eigen::MatrixXd NonLinearMPC::compute_input(const double t /*[s] */, const Eigen::MatrixXd& xref, std::vector<Obstacle*>& obstacles) {
  // this dt: sampling time
  double dt = params_->tf * (1.0 - std::exp(-params_->alpha * (t))) / params_->horizon;

  // clang-format: off
  Eigen::MatrixXd xbar = vehicle_->predict_motion(vehicle_->get_state(), U_.topRows(vehicle_->num_input_), xref, dt);
  Eigen::MatrixXd lambdas = predict_adjoint_variables(xbar, xref, obstacles, U_, dt);
  Eigen::MatrixXd F = compute_F(xbar, lambdas, U_);

  Eigen::MatrixXd next_state = vehicle_->compute_next_state_linear(vehicle_->get_state(), U_.topLeftCorner(vehicle_->num_input_, 1), params_->ht);
  Eigen::MatrixXd xbar_next = vehicle_->predict_motion(next_state, U_.topRows(vehicle_->num_input_), xref, dt);
  Eigen::MatrixXd lambdas_next = predict_adjoint_variables(xbar_next, xref, obstacles, U_, dt);
  Eigen::MatrixXd F_next = compute_F(xbar_next, lambdas_next, U_);
      

  Eigen::MatrixXd du = U_ * params_->ht; // XXX: 本当？
  Eigen::MatrixXd next_U = (U_ + du);
  Eigen::MatrixXd xbar_next2 = vehicle_->predict_motion(next_state, next_U.topRows(vehicle_->num_input_), xref, dt);
  Eigen::MatrixXd lambdas_next2 = predict_adjoint_variables(xbar_next2, xref, obstacles, next_U, dt);
  Eigen::MatrixXd F_next2 = compute_F(xbar_next2, lambdas_next2, next_U);

  Eigen::MatrixXd b = - params_->zeta * F - ((F_next - F) / params_->ht) - ((F_next2 - F_next) / params_->ht);
  Eigen::Map<Eigen::VectorXd> b_vec(b.data(), b.size());
  // clang-format: on

  // now, starts c/gmres
  double b_norm = b_vec.norm();
  
  int n = (vehicle_->num_input_ + num_dummy_inputs_ + num_constraints_) * params_->horizon;
  Eigen::MatrixXd bases = Eigen::MatrixXd::Zero(n, n+1);
  bases.col(0) = b_vec.normalized();
  Eigen::MatrixXd hs = Eigen::MatrixXd::Zero(n+1, n+1);
  Eigen::VectorXd e = Eigen::VectorXd::Zero(n+1);
  e(0) = 1.0;

  Eigen::VectorXd ys_pre = Eigen::VectorXd::Zero(n+1);

  Eigen::Map<Eigen::VectorXd> U_vec(U_.data(), U_.size());
  Eigen::VectorXd du_new = Eigen::VectorXd::Zero(n);

  for (int i = 0; i < n; i++) {
    Eigen::VectorXd du = bases.col(i) * params_->ht;
    Eigen::VectorXd next_U = (U_vec + du);
    Eigen::MatrixXd next_U_block = Eigen::Map<Eigen::MatrixXd>(next_U.data(), vehicle_->num_input_+num_dummy_inputs_+num_constraints_, params_->horizon);
    Eigen::MatrixXd next_u_input_block = next_U_block.block(0, 0, vehicle_->num_input_, params_->horizon);

    Eigen::MatrixXd xbar_next3 = vehicle_->predict_motion(next_state, next_u_input_block, xref, dt);
    Eigen::MatrixXd lambdas_next3 = predict_adjoint_variables(xbar_next2, xref, obstacles, next_U_block, dt);
    Eigen::MatrixXd F_next3 = compute_F(xbar_next3, lambdas_next3, next_U_block);

    Eigen::MatrixXd diff_F = (F_next3 - F_next) / params_->ht;
    Eigen::Map<Eigen::VectorXd> Av(diff_F.data(), diff_F.size());
    Eigen::VectorXd sum_Av = Eigen::VectorXd::Zero(n);
    for (int j = 0; j < i+1; j++) {
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
    Eigen::VectorXd ys = pseudoinverse(hsblock) * (b_norm * e.segment(0, i+1));
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

  Eigen::MatrixXd du_new_block = Eigen::Map<Eigen::MatrixXd>(du_new.data(), (vehicle_->num_input_ + num_dummy_inputs_ + num_constraints_), params_->horizon) * params_->ht; // XXX: params_->dt???
  U_ += du_new_block;

  Eigen::MatrixXd xbar_new = vehicle_->predict_motion(vehicle_->get_state(), U_.block(0, 0, vehicle_->num_input_, params_->horizon), xref, dt);
  Eigen::MatrixXd lambdas_new = predict_adjoint_variables(xbar_new, xref, obstacles, U_, dt);
  Eigen::MatrixXd F_new = compute_F(xbar_new, lambdas_new, U_);

  LOG(INFO) << "new F: " << F_new.norm();
  
  Eigen::MatrixXd new_u = U_.block(0, 0, vehicle_->num_input_, U_.cols());
  return std::move(new_u);
}

Eigen::MatrixXd NonLinearMPC::predict_adjoint_variables(const Eigen::MatrixXd& states, const Eigen::MatrixXd& xref, const std::vector<Obstacle*>& obstacles, const Eigen::MatrixXd& U, double dt) {
  const int N = params_->horizon;
  Eigen::MatrixXd lambdas = Eigen::MatrixXd::Zero(vehicle_->num_state_, N + 1);
  lambdas.col(N) = (params_->Qf * states.col(states.cols()-1)).transpose();
  for (int i = N - 1; i >= 0; i--) {
    lambdas.col(i) = compute_previous_lambda(states.col(i), xref.col(i), lambdas.col(i+1), obstacles, U.col(i), dt);
  }
  return std::move(lambdas);
}

Eigen::VectorXd NonLinearMPC::compute_previous_lambda(const Eigen::VectorXd& state, const Eigen::VectorXd& state_ref,
                                        const Eigen::VectorXd& lambda, const std::vector<Obstacle*>& obstacles,
                                        const Eigen::VectorXd& _u, double dt) {
  const auto& u = _u.segment(0, vehicle_->num_input_);
  const auto& dummy = _u.segment(vehicle_->num_input_, num_dummy_inputs_);
  const auto& mu = _u.segment(vehicle_->num_input_ + num_dummy_inputs_, num_constraints_);

  Eigen::VectorXd tmp = lambda;
  tmp += params_->Q * (state - state_ref) * dt;
  // for (auto&& o : obstacles) {
  //   Eigen::VectorXd pos(vehicle_->num_state_);
  //   pos << o->x(), o->y(), 0.0, 0.0, 0.0, 0.0;
  //   Eigen::VectorXd diff = state - pos;
  //   double d = params_->inflation_radius + o->r();
  //   double norm = diff.norm();
  //   if (norm < d) {
  //     // XXX?
  //     tmp += o->a() * ((d - norm) / norm) * diff * dt;
  //   }
  // }
  Eigen::MatrixXd dfdx = vehicle_->get_dfdz(state, u);
  tmp += lambda.transpose() * dfdx * dt;
  // clang-format off
  // Eigen::MatrixXd dCdx(num_constraints_, vehicle_->num_state_);
  // dCdx << 0, 0, 0, 0,          0,          0,
  //         0, 0, 0, 0,          0,          0,
  //         0, 0, 0, 0,          0,          0,
  //         0, 0, 0, 0,          0,          0,
  //         0, 0, 0, 2*state(3), 2*state(4), 0;
  // clang-format on
  // XXX?
  // tmp += mu.transpose() * dCdx * dt;
  return std::move(tmp);
}

Eigen::MatrixXd NonLinearMPC::compute_F(const Eigen::MatrixXd& states, const Eigen::MatrixXd& lambdas, const Eigen::MatrixXd& U) {
  const int num_all_inputs = vehicle_->num_input_ + num_dummy_inputs_;
  const int step = num_all_inputs + num_constraints_;
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
    // F.block(vehicle_->num_input_, i, num_dummy_inputs_, 1) -= Eigen::MatrixXd::Ones(num_dummy_inputs_, 1) * 0.001;
    // df/dU
    // Eigen::MatrixXd dfdu = vehicle_->get_dfdu(states.col(i), u);
    F(0, i) += (lambdas(3, i+1) + 2.0 * mu(0, 0) * u(0, 0));
    F(1, i) += (lambdas(2, i+1) * states(3, i) / vehicle_->vparams_.wb * std::cos(u(1, 0)) * std::cos(u(1, 0)) + 2.0*mu(1, 0)*u(1, 0));
    F(2, i) += (- vehicle_->vparams_.phi_v + 2.0*mu(0, 0) * dummy(0, 0));
    F(3, i) += (- vehicle_->vparams_.phi_omega + 2.0*mu(1, 0) * dummy(1, 0));

    // clang-format off
    ////////// Eigen::MatrixXd dCdU(num_constraints_, num_all_inputs);
    ////////// dCdU << 2*u(0, 0),         0,         0,         0, 2*dummy(0, 0),             0,             0,             0,              0,
    //////////                 0, 2*u(1, 0) ,        0,         0,             0, 2*dummy(1, 0),             0,             0,              0,
    //////////                 0,         0, 2*u(2, 0),         0,             0,             0, 2*dummy(2, 0),             0,              0,
    //////////                 0,         0,         0, 2*u(3, 0),             0,             0,             0, 2*dummy(3, 0),              0,
    //////////                 //0,         0,         0,         0,             0,             0,             0,             0,              0;
    //////////                 0,         0,         0,         0,             0,             0,             0,             0, 2*dummy(4,   0);
    ////////// // clang-format on
    ////////// // dC/dU
    ////////// F.block(0, i, num_all_inputs, 1) += (mu.transpose() * dCdU).transpose();

    ////////// // constraints
    ////////// // clang-format off
    ////////// const double& max_motor_torque = vehicle_->vparams_.max_motor_torque;
    ////////// const double max_motor_torque_sqrn = max_motor_torque * max_motor_torque;
    ////////// F(num_all_inputs + 0, i) += u(0, 0)*u(0, 0) + dummy(0, 0)*dummy(0, 0) - max_motor_torque_sqrn;
    ////////// F(num_all_inputs + 1, i) += u(1, 0)*u(1, 0) + dummy(1, 0)*dummy(1, 0) - max_motor_torque_sqrn;
    ////////// F(num_all_inputs + 2, i) += u(2, 0)*u(2, 0) + dummy(2, 0)*dummy(2, 0) - max_motor_torque_sqrn;
    ////////// F(num_all_inputs + 3, i) += u(3, 0)*u(3, 0) + dummy(3, 0)*dummy(3, 0) - max_motor_torque_sqrn;
    ////////// // clang-format on
    ////////// const auto& state = states.col(i);
    ////////// const double& vx = state(3);
    ////////// const double& vy = state(4);

    ////////// const double max_speed_sqrn = vehicle_->vparams_.max_speed * vehicle_->vparams_.max_speed;
    ////////// F(num_all_inputs + 4, i) += vx * vx + vy * vy + dummy(4, 0) * dummy(4, 0) - max_speed_sqrn;
    const double& u_a_max = vehicle_->vparams_.u_a_max;
    const double uu = u_a_max * u_a_max;
    const double& u_omega_max = vehicle_->vparams_.u_omega_max;
    const double uu_omega = u_omega_max * u_omega_max;
    F(num_all_inputs + 0, i) += u(0, 0)*u(0, 0) + dummy(0, 0)*dummy(0, 0) - uu;
    F(num_all_inputs + 1, i) += u(1, 0)*u(1, 0) + dummy(1, 0)*dummy(1, 0) - uu_omega;

    //head += step;
  }
  return std::move(F);
}

