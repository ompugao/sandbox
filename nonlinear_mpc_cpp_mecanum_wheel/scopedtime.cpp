#include "scopedtime.h"
#include <iostream>

ScopedTime::ScopedTime(const std::string& msg) : msg_(msg) {
  start_ = std::chrono::system_clock::now();
}
void ScopedTime::lap(const std::string& msg) {
  const auto duration_time = std::chrono::system_clock::now() - start_;
  const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_time);
  std::cerr << "[" << duration_ns.count() << " ns] " << msg << std::endl;
}
ScopedTime::~ScopedTime() {
  this->lap(msg_);
}
