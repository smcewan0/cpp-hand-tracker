#include "LandmarkSmoother.h"

#include <algorithm>

LandmarkSmoother::LandmarkSmoother(float alpha)
    : alpha_(std::clamp(alpha, 0.0f, 1.0f)) {}

void LandmarkSmoother::smooth(std::vector<HandResult>& hands) {
    if (hands.empty()) {
        reset();
        return;
    }

    if (previous_.size() != hands.size()) {
        previous_.resize(hands.size());
        initialized_.assign(hands.size(), false);
    }

    for (size_t handIndex = 0; handIndex < hands.size(); ++handIndex) {
        auto& current = hands[handIndex].landmarks;
        auto& previous = previous_[handIndex];

        if (!initialized_[handIndex]) {
            previous = current;
            initialized_[handIndex] = true;
            continue;
        }

        for (size_t i = 0; i < current.size(); ++i) {
            current[i].x = previous[i].x * (1.0f - alpha_) + current[i].x * alpha_;
            current[i].y = previous[i].y * (1.0f - alpha_) + current[i].y * alpha_;
            current[i].z = previous[i].z * (1.0f - alpha_) + current[i].z * alpha_;
            previous[i] = current[i];
        }
    }
}

void LandmarkSmoother::reset() {
    previous_.clear();
    initialized_.clear();
}
