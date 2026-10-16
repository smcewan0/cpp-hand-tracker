#include "PalmDetector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

cv::Mat makeNhwcBlob(const cv::Mat& bgrImage) {
    cv::Mat rgb;
    cv::cvtColor(bgrImage, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

    const int sizes[] = {1, rgbFloat.rows, rgbFloat.cols, 3};
    cv::Mat blob(4, sizes, CV_32F);
    std::memcpy(blob.ptr<float>(), rgbFloat.ptr<float>(),
                static_cast<size_t>(rgbFloat.total()) * 3U * sizeof(float));
    return blob;
}

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

PalmDetector::PalmDetector(const std::string& modelPath,
                           float scoreThreshold,
                           float nmsThreshold,
                           int topK)
    : net_(cv::dnn::readNet(modelPath)),
      scoreThreshold_(scoreThreshold),
      nmsThreshold_(nmsThreshold),
      topK_(topK),
      anchors_(generateAnchors()) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<PalmDetection> PalmDetector::infer(const cv::Mat& frame) {
    const PreprocessResult prep = preprocess(frame);
    net_.setInput(prep.blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    return postprocess(outputs, frame.size(), prep.padBias);
}

PalmDetector::PreprocessResult PalmDetector::preprocess(const cv::Mat& frame) const {
    if (frame.empty()) {
        throw std::invalid_argument("Cannot preprocess an empty frame.");
    }

    const float ratio = std::min(static_cast<float>(inputSize_.width) / frame.cols,
                                 static_cast<float>(inputSize_.height) / frame.rows);
    const cv::Size resizedSize(std::max(1, static_cast<int>(frame.cols * ratio)),
                               std::max(1, static_cast<int>(frame.rows * ratio)));

    cv::Mat resized;
    cv::resize(frame, resized, resizedSize);

    const int padW = inputSize_.width - resizedSize.width;
    const int padH = inputSize_.height - resizedSize.height;
    const int left = padW / 2;
    const int top = padH / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, padH - top, left, padW - left,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return {makeNhwcBlob(padded), cv::Point2f(left / ratio, top / ratio)};
}

std::vector<PalmDetection> PalmDetector::postprocess(const std::vector<cv::Mat>& outputs,
                                                     const cv::Size& originalSize,
                                                     const cv::Point2f& padBias) const {
    if (outputs.size() < 2) {
        throw std::runtime_error("Palm detector returned fewer outputs than expected.");
    }

    const cv::Mat boxes = outputs[0].reshape(1, static_cast<int>(outputs[0].total() / 18));
    const cv::Mat scores = outputs[1].reshape(1, static_cast<int>(outputs[1].total()));

    std::vector<float> confidences;
    std::vector<cv::Rect> nmsBoxes;
    std::vector<PalmDetection> candidates;
    const float scale = static_cast<float>(std::max(originalSize.width, originalSize.height));

    const int rows = std::min({boxes.rows, scores.rows, static_cast<int>(anchors_.size())});
    for (int i = 0; i < rows; ++i) {
        const float score = sigmoid(scores.at<float>(i, 0));
        if (score < scoreThreshold_) {
            continue;
        }

        const cv::Point2f anchor = anchors_[i];
        const float cx = boxes.at<float>(i, 0) / inputSize_.width + anchor.x;
        const float cy = boxes.at<float>(i, 1) / inputSize_.height + anchor.y;
        const float width = boxes.at<float>(i, 2) / inputSize_.width;
        const float height = boxes.at<float>(i, 3) / inputSize_.height;

        const cv::Point2f topLeft((cx - width * 0.5f) * scale - padBias.x,
                                  (cy - height * 0.5f) * scale - padBias.y);
        const cv::Point2f bottomRight((cx + width * 0.5f) * scale - padBias.x,
                                      (cy + height * 0.5f) * scale - padBias.y);

        PalmDetection palm;
        palm.box = cv::Rect2f(topLeft, bottomRight);
        palm.score = score;

        for (int j = 0; j < 7; ++j) {
            const float x = (boxes.at<float>(i, 4 + j * 2) / inputSize_.width + anchor.x) * scale - padBias.x;
            const float y = (boxes.at<float>(i, 5 + j * 2) / inputSize_.height + anchor.y) * scale - padBias.y;
            palm.landmarks[static_cast<size_t>(j)] = cv::Point2f(x, y);
        }

        confidences.push_back(score);
        candidates.push_back(palm);
        nmsBoxes.emplace_back(cvRound(palm.box.x), cvRound(palm.box.y),
                              cvRound(palm.box.width), cvRound(palm.box.height));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(nmsBoxes, confidences, scoreThreshold_, nmsThreshold_, indices, 1.0f, topK_);

    std::vector<PalmDetection> palms;
    palms.reserve(indices.size());
    for (const int index : indices) {
        palms.push_back(candidates[static_cast<size_t>(index)]);
    }

    std::sort(palms.begin(), palms.end(), [](const PalmDetection& lhs, const PalmDetection& rhs) {
        return lhs.score > rhs.score;
    });

    return palms;
}

std::vector<cv::Point2f> PalmDetector::generateAnchors() const {
    std::vector<cv::Point2f> anchors;
    anchors.reserve(2016);

    const auto appendGrid = [&anchors](int gridSize, int anchorsPerCell) {
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                const cv::Point2f center((x + 0.5f) / gridSize, (y + 0.5f) / gridSize);
                for (int i = 0; i < anchorsPerCell; ++i) {
                    anchors.push_back(center);
                }
            }
        }
    };

    appendGrid(24, 2);
    appendGrid(12, 6);
    return anchors;
}
