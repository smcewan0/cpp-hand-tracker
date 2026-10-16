#pragma once

#include "Types.h"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <optional>
#include <string>

class HandLandmarkDetector {
public:
    explicit HandLandmarkDetector(const std::string& modelPath, float confidenceThreshold = 0.8f);

    std::optional<HandResult> infer(const cv::Mat& frame, const PalmDetection& palm);

private:
    struct CropResult {
        cv::Mat image;
        cv::Point2f topLeft;
        cv::Point2f bottomRight;
        cv::Point2f bias;
    };

    struct PreprocessResult {
        cv::Mat blob;
        cv::Point2f rotatedTopLeft;
        cv::Point2f rotatedBottomRight;
        double angleDegrees{0.0};
        cv::Matx23f rotationMatrix{};
        cv::Point2f padBias;
    };

    CropResult cropAndPad(const cv::Mat& image,
                          cv::Point2f topLeft,
                          cv::Point2f bottomRight,
                          bool forRotation) const;
    PreprocessResult preprocess(const cv::Mat& frame, const PalmDetection& palm) const;
    std::optional<HandResult> postprocess(const std::vector<cv::Mat>& outputs,
                                          const PreprocessResult& prep) const;

    cv::dnn::Net net_;
    float confidenceThreshold_{0.8f};
    cv::Size inputSize_{224, 224};
};
