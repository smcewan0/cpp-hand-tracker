#pragma once

#include "Types.h"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class PalmDetector {
public:
    explicit PalmDetector(const std::string& modelPath,
                          float scoreThreshold = 0.75f,
                          float nmsThreshold = 0.3f,
                          int topK = 32);

    std::vector<PalmDetection> infer(const cv::Mat& frame);

private:
    struct PreprocessResult {
        cv::Mat blob;
        cv::Point2f padBias;
    };

    PreprocessResult preprocess(const cv::Mat& frame) const;
    std::vector<PalmDetection> postprocess(const std::vector<cv::Mat>& outputs,
                                           const cv::Size& originalSize,
                                           const cv::Point2f& padBias) const;
    std::vector<cv::Point2f> generateAnchors() const;

    cv::dnn::Net net_;
    cv::Size inputSize_{192, 192};
    float scoreThreshold_{0.75f};
    float nmsThreshold_{0.3f};
    int topK_{32};
    std::vector<cv::Point2f> anchors_;
};
