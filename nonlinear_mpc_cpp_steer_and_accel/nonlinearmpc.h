#pragma once

#include "vehicle.h"
#include "obstacle.h"
#include <memory>
#include <Eigen/Dense>

class NonLinearMPC {
public:
  NonLinearMPC(const Vehicle& v, const std::shared_ptr<Parameters> params);

  virtual ~NonLinearMPC() {}

  void initialize();
  Eigen::MatrixXd compute_input(const double t /*[s] */, const Eigen::MatrixXd& xref, std::vector<Obstacle*>& obstacles);

  Eigen::MatrixXd predict_adjoint_variables(const Eigen::MatrixXd& states, const Eigen::MatrixXd& xref, const std::vector<Obstacle*>& obstacles, const Eigen::MatrixXd& U, double dt);
  Eigen::VectorXd compute_previous_lambda(const Eigen::VectorXd& state, const Eigen::VectorXd& state_ref,
                                          const Eigen::VectorXd& lambda, const std::vector<Obstacle*>& obstacles,
                                          const Eigen::VectorXd& _u, double dt);

  Eigen::MatrixXd compute_F(const Eigen::MatrixXd& states, const Eigen::MatrixXd& lambdas, const Eigen::MatrixXd& U);

  const Vehicle* vehicle_;
  const int num_constraints_ = 2;
  const int num_dummy_inputs_ = 2;  // == num_constraints_; ?
  std::shared_ptr<Parameters> params_;

  Eigen::MatrixXd U_;
  //Eigen::MatrixXd lambdas_;  // (i, ht*j)
  // Eigen::VectorXd mu_;
};

