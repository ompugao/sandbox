#pragma once
#include <cmath>

inline double normalize_angle_positive(double angle) {
  return std::fmod(std::fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
}

inline double normalize_angle(double angle) {
  double a = normalize_angle_positive(angle);
  if (a > M_PI) {
    a -= 2.0 * M_PI;
  }
  return a;
}

