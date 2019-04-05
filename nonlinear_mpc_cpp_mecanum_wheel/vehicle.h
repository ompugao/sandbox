#pragma once
#include <memory>
#include <Eigen/Core>
#include "parameters.h"
#include "course.h"

class VehicleParameters {
public:
  VehicleParameters() {}
  virtual ~VehicleParameters() {}

  const double lf = 0.5;
  const double lb = 0.5;

  const double width = 2.0;
  // const double wheel_len = 0.3;
  // const double wheel_width = 0.2;
  const double wheel_radius = 0.1;
  const double wheel_mass = 0.3;
  const double Iw = 1.0 / 2 * wheel_mass * wheel_radius * wheel_radius;
  const double mass = 5.0;
  const double Iv = 1.0 / 2 * mass * 0.5 * 0.5;

  const double input_gain = 1.0;
  const double damping_gain = 0.01;

  const double max_speed = 5.0;
  const double min_speed = -5.0;

  const double max_motor_torque = 1.57 * 1.0;
  const double max_motor_torque_vel = 1.57;
};

class Vehicle {
public:
  Vehicle(const std::shared_ptr<Parameters>& env_params);

  virtual ~Vehicle() {}
  void get_linear_matrix_diff(const Eigen::VectorXd& state, Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                         Eigen::VectorXd& C) const;

  void get_linear_matrix(const Eigen::VectorXd& state, Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                         Eigen::VectorXd& C, double dt = -1.0) const;

  void plot();

  Eigen::VectorXd compute_next_state_linear(const Eigen::VectorXd& state, Eigen::VectorXd u, double dt = -1) const;

  Eigen::VectorXd compute_next_state_rk4(const Eigen::VectorXd& state, Eigen::VectorXd u, double dt = -1) const;

  int find_nearest_index(const Course& c, int pind) const;

  Eigen::MatrixXd predict_motion(Eigen::VectorXd state, Eigen::MatrixXd inputs, const Eigen::MatrixXd& xref, double dt = -1) const;

  Eigen::MatrixXd predict_motion_rk4(Eigen::VectorXd state, Eigen::MatrixXd inputs, const Eigen::MatrixXd& xref, double dt = -1) const;

  void set_state(double x, double y, double theta, double vx, double vy, double thetadot);

  void set_state(const Eigen::MatrixXd& state);

  const Eigen::VectorXd& get_state() const;

  void get_state(double& x, double& y, double& theta, double& vx, double& vy, double& thetadot) const;

  bool is_arrived(const Course& c, const WayPoint& w, int iw);

  Eigen::VectorXd state_;
  VehicleParameters vparams_;

  std::shared_ptr<Parameters> params_;
  const int num_state_ = 6;
  const int num_input_ = 4;

  Eigen::MatrixXd A_, B_, C_;
  Eigen::MatrixXd Abar_, Bbar_, Cbar_;
};
