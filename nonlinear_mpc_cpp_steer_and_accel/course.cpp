#include "course.h"
#include "csv.h"
#include "utils.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

Course::Course() {}
Course::~Course() {}
size_t Course::size() const {
  return waypoints.size();
}
void Course::load_csv(const std::string& path) {
  waypoints.clear();
  xs.clear();
  ys.clear();

  io::CSVReader<4> in(path);
  in.read_header(io::ignore_extra_column, "cx", "cy", "ctheta", "ck");
  double cx, cy, ctheta, ck;
  while (in.read_row(cx, cy, ctheta, ck)) {
    waypoints.emplace_back(WayPoint(cx, cy, ctheta));
    xs.emplace_back(cx);
    ys.emplace_back(cy);
  }
}

void Course::set_speed_profile(const double target_v) {
  for (int i = 0; i < waypoints.size() - 1; i++) {
    double dx = waypoints[i + 1].x - waypoints[i].x;
    double dy = waypoints[i + 1].y - waypoints[i].y;
    double movedir = std::atan2(dy, dx);
    double dir = 0.0;
    if (dx != 0.0 && dy != 0.0) {
      double d_angle = std::abs(normalize_angle(movedir - waypoints[i].theta));
      if (d_angle > M_PI / 4.0) {
        dir = -1.0;
      } else {
        dir = 1.0;
      }
      waypoints[i].speed = dir * target_v;
    }
  }
  waypoints[waypoints.size() - 1].speed = 0;
}

void Course::smooth_theta() {
  for (int i = 0; i < waypoints.size() - 1; i++) {
    WayPoint& w_n = waypoints[i + 1];
    WayPoint& w = waypoints[i];
    double dtheta = w_n.theta - w.theta;
    while (dtheta >= M_PI / 2.0) {
      w_n.theta -= M_PI * 2.0;
      dtheta = w_n.theta - w.theta;
    }

    while (dtheta <= -M_PI / 2.0) {
      w_n.theta += M_PI * 2.0;
      dtheta = w_n.theta - w.theta;
    }
  }
}

void Course::plot() {
  plt::plot(xs, ys, "-r");
}

