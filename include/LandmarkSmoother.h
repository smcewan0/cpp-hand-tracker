#pragma once

#include "Types.h"

#include <array>
#include <vector>

class LandmarkSmoother {
public:
    explicit LandmarkSmoother(float alpha = 0.35f);

    void smooth(std::vector<HandResult>& hands);
    void reset();

private:
    float alpha_{0.35f};
    std::vector<std::array<cv::Point3f, kHandLandmarkCount>> previous_;
    std::vector<bool> initialized_;
};
