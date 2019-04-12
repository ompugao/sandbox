#include "vehicle.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

Vehicle::Vehicle(const std::shared_ptr<Parameters>& env_params) {
  params_ = env_params;
  state_ = Eigen::VectorXd(num_state_);

  A_ = Eigen::MatrixXd::Zero(num_state_, num_state_);
  B_ = Eigen::MatrixXd::Zero(num_state_, num_input_);
  C_ = Eigen::VectorXd::Zero(num_state_);

  // const double& dt = params_->dt;
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
  A_(2, 5) += 1;

  const double coeff1 = r * r * M + 2 * Iw;
  A_(3, 3) += (-2 * c / coeff1);
  B_(3, 0) += (k * r / 2.0 * (-1)) / coeff1;
  B_(3, 1) += (k * r / 2.0 * (-1)) / coeff1;
  B_(3, 2) += (k * r / 2.0 * (+1)) / coeff1;
  B_(3, 3) += (k * r / 2.0 * (+1)) / coeff1;

  const double coeff2 = coeff1 - Iw * Iw * (lb - lf) * (lb - lf) / A;

  A_(4, 4) += (Iw * (lb - lf) * (lb - lf) / A - 2) * c / coeff2;
  A_(4, 5) += (Iw * B / A + 1) * c * (lb - lf) / coeff2;
  B_(4, 0) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lf + w)) + k * r / 2) / coeff2;
  B_(4, 1) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lb + w)) - k * r / 2) / coeff2;
  B_(4, 2) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lb + w)) - k * r / 2) / coeff2;
  B_(4, 3) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lf + w)) + k * r / 2) / coeff2;

  const double coeff3 = A - Iw * Iw * (lb - lf) * (lb - lf) / coeff1;

  A_(5, 4) += (-2 * Iw / coeff1 + 1) * c * (lb - lf) / coeff3;
  A_(5, 5) += c * (Iw * (lb - lf) * (lb - lf) / coeff1 - B) / coeff3;

  B_(5, 0) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (+1.0) + k * r / 2.0 * (lf + w)) / coeff3;
  B_(5, 1) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (-1.0) + k * r / 2.0 * (lb + w)) / coeff3;
  B_(5, 2) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (-1.0) + k * r / 2.0 * (lb + w)) / coeff3;
  B_(5, 3) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (+1.0) + k * r / 2.0 * (lf + w)) / coeff3;

  Abar_ = A_;
  Bbar_ = B_;
  Cbar_ = C_;
}

void Vehicle::get_linear_matrix_diff(const Eigen::VectorXd& state, Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                                     Eigen::VectorXd& C) const {
  const double& vx = state(3);
  const double& vy = state(4);
  const double& theta = state(2);
  A = Abar_;
  B = Bbar_;
  C = Cbar_;

  A(0, 2) += (-vx * std::sin(theta) - vy * std::cos(theta));
  A(0, 3) += std::cos(theta);
  A(0, 4) += (-std::sin(theta));
  A(1, 2) += (vx * std::cos(theta) - vy * std::sin(theta));
  A(1, 3) += std::sin(theta);
  A(1, 4) += std::cos(theta);
  C(0) = (vx * std::sin(theta) + vy * std::cos(theta)) * theta;
  C(1) = (-vx * std::cos(theta) + vy * std::sin(theta)) * theta;
  return;
}

void Vehicle::get_linear_matrix(const Eigen::VectorXd& state, Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                                Eigen::VectorXd& C, double dt) const {
  if (dt < 0) {
    dt = params_->dt;
  }
  const double& vx = state(3);
  const double& vy = state(4);
  const double& theta = state(2);

  A = Abar_;
  B = Bbar_;
  C = Cbar_;

  A(0, 2) += (-vx * std::sin(theta) - vy * std::cos(theta));
  A(0, 3) += std::cos(theta);
  A(0, 4) += (-std::sin(theta));
  A(1, 2) += (vx * std::cos(theta) - vy * std::sin(theta));
  A(1, 3) += std::sin(theta);
  A(1, 4) += std::cos(theta);
  A *= dt;
  A += Eigen::MatrixXd::Identity(num_state_, num_state_);
  B *= dt;
  C(0) = dt * (vx * std::sin(theta) + vy * std::cos(theta)) * theta;
  C(1) = dt * (-vx * std::cos(theta) + vy * std::sin(theta)) * theta;
  return;
}

void Vehicle::plot() {
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

Eigen::VectorXd Vehicle::compute_next_state_linear(const Eigen::VectorXd& state, Eigen::VectorXd u,
                                                   double dt) const {
  if (dt < 0) {
    dt = params_->dt;
  }
  Eigen::VectorXd u_copy = u;
  u_copy(0) = std::clamp(u(0), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  u_copy(1) = std::clamp(u(1), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  u_copy(2) = std::clamp(u(2), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  u_copy(3) = std::clamp(u(3), -vparams_.max_motor_torque, vparams_.max_motor_torque);

  Eigen::MatrixXd A, B;
  Eigen::VectorXd C(num_state_);
  get_linear_matrix(state, A, B, C, dt);

  Eigen::VectorXd new_state = A * state + B * u_copy + C;
  double v;
  v = std::clamp(new_state(3), vparams_.min_speed, vparams_.max_speed);
  new_state(3) = v;
  v = std::clamp(new_state(4), vparams_.min_speed, vparams_.max_speed);
  new_state(4) = v;
  return std::move(new_state);
}

Eigen::VectorXd Vehicle::compute_next_state_rk4(const Eigen::VectorXd& state, Eigen::VectorXd u, double dt) const {
  if (dt < 0) {
    dt = params_->dt;
  }
  Eigen::VectorXd u_copy = u;
  u_copy(0) = std::clamp(u(0), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  u_copy(1) = std::clamp(u(1), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  u_copy(2) = std::clamp(u(2), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  u_copy(3) = std::clamp(u(3), -vparams_.max_motor_torque, vparams_.max_motor_torque);

  Eigen::MatrixXd A, B;
  Eigen::VectorXd C(num_state_);

  auto f = [&](const Eigen::VectorXd& state, const Eigen::VectorXd& u) -> Eigen::VectorXd {
    //Eigen::VectorXd xydiff(num_state_);
    //const double& theta = state(2);
    //const double& vx = state(3);
    //const double& vy = state(4);

    //xydiff(0) = vx * std::cos(theta) - vy * std::sin(theta);
    //xydiff(1) = vx * std::sin(theta) + vy * std::cos(theta);
    get_linear_matrix_diff(state, A, B, C);
    return A * state + B * u + C;
  };
  auto k1 = f(state, u);
  auto k2 = f(state + 0.5 * k1 * dt, u);
  auto k3 = f(state + 0.5 * k2 * dt, u);
  auto k4 = f(state + k3 * dt, u);
  auto k = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0;
  return state + k;
}

int Vehicle::find_nearest_index(const Course& c, int pind) const {
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

Eigen::MatrixXd Vehicle::predict_motion(Eigen::VectorXd state, Eigen::MatrixXd inputs, const Eigen::MatrixXd& xref,
                                        double dt) const {
  if (dt < 0) {
    dt = params_->dt;
  }
  Eigen::MatrixXd xbar = Eigen::MatrixXd::Zero(xref.rows(), xref.cols());

  xbar.col(0) = state;

  for (int i = 0; i < inputs.cols(); i++) {
    state = compute_next_state_linear(state, inputs.col(i), dt);
    xbar.col(i + 1) = state;
  }

  return xbar;
}

Eigen::MatrixXd Vehicle::predict_motion_rk4(Eigen::VectorXd state, Eigen::MatrixXd inputs, const Eigen::MatrixXd& xref,
                                            double dt) const {
  Eigen::MatrixXd xbar = Eigen::MatrixXd::Zero(xref.rows(), xref.cols());
  if (dt < 0) {
    dt = params_->dt;
  }

  xbar.col(0) = state;

  for (int i = 0; i < inputs.cols(); i++) {
    state = compute_next_state_rk4(state, inputs.col(i), dt);
    xbar.col(i + 1) = state;
  }

  return xbar;
}

void Vehicle::set_state(double x, double y, double theta, double vx, double vy, double thetadot) {
  state_(0) = x;
  state_(1) = y;
  state_(2) = theta;
  state_(3) = vx;
  state_(4) = vy;
  state_(5) = thetadot;
}

void Vehicle::set_state(const Eigen::MatrixXd& state) {
  state_ = state;
}

const Eigen::VectorXd& Vehicle::get_state() const {
  return state_;
}

void Vehicle::get_state(double& x, double& y, double& theta, double& vx, double& vy, double& thetadot) const {
  x = state_(0);
  y = state_(1);
  theta = state_(2);
  vx = state_(3);
  vy = state_(4);
  thetadot = state_(5);
}

bool Vehicle::is_arrived(const Course& c, const WayPoint& w, int iw) {
  double d = std::hypot(state_(0) - w.x, state_(1) - w.y);
  bool closeenough = (d < params_->goal_distance_tolerance) && (c.size() == iw + 1);
  bool stopped = std::abs(state_(3)) < params_->stop_speed && std::abs(state_(4)) < params_->stop_speed;
  return closeenough && stopped;
}

