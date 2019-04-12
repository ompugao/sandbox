#pragma once
#include <vector>
#include <string>

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
  Course();
  virtual ~Course();
  size_t size() const;
  void load_csv(const std::string& path);

  void set_speed_profile(const double target_v);

  void smooth_theta();

  void plot();
  std::vector<WayPoint> waypoints;

  // for visualization
  std::vector<double> xs, ys;
};

