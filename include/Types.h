#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

constexpr int kHandLandmarkCount = 21;

struct FrameOverlay {
    bool active{false};
    std::array<cv::Point2f, 4> corners{};
};

struct PalmDetection {
    cv::Rect2f box;
    std::array<cv::Point2f, 7> landmarks{};
    float score{0.0f};
};

struct HandResult {
    cv::Rect2f box;
    std::array<cv::Point3f, kHandLandmarkCount> landmarks{};
    std::array<cv::Point3f, kHandLandmarkCount> worldLandmarks{};
    float handedness{0.0f};
    float confidence{0.0f};
    bool pinching{false};
    float pinchDistance{0.0f};
    std::string gesture{"OPEN"};
};
