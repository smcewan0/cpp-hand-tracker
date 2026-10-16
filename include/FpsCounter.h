#pragma once

#include <chrono>

class FpsCounter {
public:
    double tick();
    double fps() const { return fps_; }

private:
    std::chrono::steady_clock::time_point lastTick_{};
    double fps_{0.0};
};
