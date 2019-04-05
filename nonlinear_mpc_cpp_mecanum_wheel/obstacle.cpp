#include "obstacle.h"
Obstacle::Obstacle(double x, double y, double r, double circumscribed_area_cost, double inflation_radius) {
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
Obstacle::~Obstacle() {}

const double& Obstacle::x() {
  return x_;
}
const double& Obstacle::y() {
  return y_;
}
const double& Obstacle::r() {
  return r_;
}
const double& Obstacle::a() {
  return a_;
}

void Obstacle::step(double dt) {}

