#pragma once

class Obstacle {
public:
  Obstacle(double x, double y, double r, double circumscribed_area_cost, double inflation_radius);
  virtual ~Obstacle();

  const double& x();
  const double& y();
  const double& r();
  const double& a();

  void step(double dt);
  double x_, y_, r_;
  double a_;
};
