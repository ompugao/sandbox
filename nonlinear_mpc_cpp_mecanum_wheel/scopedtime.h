#pragma once
#include <chrono>
#include <string>

class ScopedTime {
public:
  ScopedTime(const std::string& msg = "");
  void lap(const std::string& msg);
  virtual ~ScopedTime();
private:
  std::chrono::system_clock::time_point start_;
  std::string msg_;
};
