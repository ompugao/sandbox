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
  // const double B = (lb + w) * (lb + w) + (lf + w) * (lf + w);
  // const double A = r * r * Iv + B * Iw;

  // // theta
  // A_(2, 5) += 1;

  // const double coeff1 = r * r * M + 2 * Iw;
  // A_(3, 3) += (-2 * c / coeff1);
  // B_(3, 0) += (k * r / 2.0 * (-1)) / coeff1;
  // B_(3, 1) += (k * r / 2.0 * (-1)) / coeff1;
  // B_(3, 2) += (k * r / 2.0 * (+1)) / coeff1;
  // B_(3, 3) += (k * r / 2.0 * (+1)) / coeff1;

  // const double coeff2 = coeff1 - Iw * Iw * (lb - lf) * (lb - lf) / A;

  // A_(4, 4) += (Iw * (lb - lf) * (lb - lf) / A - 2) * c / coeff2;
  // A_(4, 5) += (Iw * B / A + 1) * c * (lb - lf) / coeff2;
  // B_(4, 0) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lf + w)) + k * r / 2) / coeff2;
  // B_(4, 1) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lb + w)) - k * r / 2) / coeff2;
  // B_(4, 2) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lb + w)) - k * r / 2) / coeff2;
  // B_(4, 3) += ((k * r / 2.0 * Iw / A * (lb - lf) * (lf + w)) + k * r / 2) / coeff2;

  // const double coeff3 = A - Iw * Iw * (lb - lf) * (lb - lf) / coeff1;

  // A_(5, 4) += (-2 * Iw / coeff1 + 1) * c * (lb - lf) / coeff3;
  // A_(5, 5) += c * (Iw * (lb - lf) * (lb - lf) / coeff1 - B) / coeff3;

  // B_(5, 0) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (+1.0) + k * r / 2.0 * (lf + w)) / coeff3;
  // B_(5, 1) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (-1.0) + k * r / 2.0 * (lb + w)) / coeff3;
  // B_(5, 2) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (-1.0) + k * r / 2.0 * (lb + w)) / coeff3;
  // B_(5, 3) += ((k * r / 2.0 * Iw * (lb - lf) / coeff1) * (+1.0) + k * r / 2.0 * (lf + w)) / coeff3;

  // Abar_ = A_;
  // Bbar_ = B_;
  // Cbar_ = C_;
}

Eigen::VectorXd Vehicle::get_diff(const Eigen::VectorXd& state, const Eigen::VectorXd& u) const {
  const double x     = state[0];
  const double y     = state[1];
  const double theta = state[2];
  const double v     = state[3];
  Eigen::VectorXd diff_state(num_state_);
  diff_state << v*std::cos(theta), v*std::sin(theta), v / vparams_.wb * std::sin(u[1]), u[0];
  return std::move(diff_state);
}

Eigen::MatrixXd Vehicle::get_dfdz(const Eigen::VectorXd& state, const Eigen::VectorXd& u) const {
  const double x     = state[0];
  const double y     = state[1];
  const double theta = state[2];
  const double v     = state[3];
  Eigen::MatrixXd dfdz(num_state_, num_state_);
  dfdz << 0, 0, -v*std::sin(theta), std::cos(theta),
          0, 0, v*std::cos(theta), std::sin(theta),
          0, 0, 0, 1.0/vparams_.wb*std::sin(u[1]),
          0, 0, 0, 0;
  return std::move(dfdz);
}

Eigen::MatrixXd Vehicle::get_dfdu(const Eigen::VectorXd& state, const Eigen::VectorXd& u) const {
  const double x     = state[0];
  const double y     = state[1];
  const double theta = state[2];
  const double v     = state[3];
  Eigen::MatrixXd dfdu(num_state_, num_input_);
  dfdu << 0, 0,
          0, 0,
          0,1.0/vparams_.wb*std::cos(u[1]),
          1, 0;
  return std::move(dfdu);
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
  // u_copy(0) = std::clamp(u(0), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  // u_copy(1) = std::clamp(u(1), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  // u_copy(2) = std::clamp(u(2), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  // u_copy(3) = std::clamp(u(3), -vparams_.max_motor_torque, vparams_.max_motor_torque);

  Eigen::VectorXd new_state = state + get_diff(state, u_copy) * dt;
  //double v;
  //v = std::clamp(new_state(3), vparams_.min_speed, vparams_.max_speed);
  //new_state(3) = v;
  return std::move(new_state);
}

Eigen::VectorXd Vehicle::compute_next_state_rk4(const Eigen::VectorXd& state, Eigen::VectorXd u, double dt) const {
  if (dt < 0) {
    dt = params_->dt;
  }
  Eigen::VectorXd u_copy = u;
  // u_copy(0) = std::clamp(u(0), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  // u_copy(1) = std::clamp(u(1), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  // u_copy(2) = std::clamp(u(2), -vparams_.max_motor_torque, vparams_.max_motor_torque);
  // u_copy(3) = std::clamp(u(3), -vparams_.max_motor_torque, vparams_.max_motor_torque);

  Eigen::MatrixXd A, B;
  Eigen::VectorXd C(num_state_);

  auto k1 = get_diff(state, u);
  auto k2 = get_diff(state + 0.5 * k1 * dt, u);
  auto k3 = get_diff(state + 0.5 * k2 * dt, u);
  auto k4 = get_diff(state + k3 * dt, u);
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
  vy = 0;
  thetadot = 0;
}

bool Vehicle::is_arrived(const Course& c, const WayPoint& w, int iw) {
  double d = std::hypot(state_(0) - w.x, state_(1) - w.y);
  bool closeenough = (d < params_->goal_distance_tolerance) && (c.size() == iw + 1);
  bool stopped = std::abs(state_(3)) < params_->stop_speed;
  return closeenough && stopped;
}

