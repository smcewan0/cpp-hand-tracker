#include "FpsCounter.h"

double FpsCounter::tick() {
    const auto now = std::chrono::steady_clock::now();
    if (lastTick_.time_since_epoch().count() != 0) {
        const std::chrono::duration<double> elapsed = now - lastTick_;
        if (elapsed.count() > 0.0) {
            const double instantFps = 1.0 / elapsed.count();
            fps_ = fps_ <= 0.0 ? instantFps : fps_ * 0.9 + instantFps * 0.1;
        }
    }
    lastTick_ = now;
    return fps_;
}
