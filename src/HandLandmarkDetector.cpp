#include "HandLandmarkDetector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

constexpr int kPalmBaseIndex = 0;
constexpr int kMiddleFingerBaseIndex = 2;
constexpr double kPi = 3.14159265358979323846;

cv::Point2f multiply(const cv::Point2f& lhs, const cv::Point2f& rhs) {
    return {lhs.x * rhs.x, lhs.y * rhs.y};
}

cv::Mat makeNhwcBlobFromRgb(const cv::Mat& rgbImage) {
    cv::Mat rgbFloat;
    rgbImage.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

    const int sizes[] = {1, rgbFloat.rows, rgbFloat.cols, 3};
    cv::Mat blob(4, sizes, CV_32F);
    std::memcpy(blob.ptr<float>(), rgbFloat.ptr<float>(),
                static_cast<size_t>(rgbFloat.total()) * 3U * sizeof(float));
    return blob;
}

cv::Point2f transformPoint(const cv::Point2f& point, const cv::Matx23f& matrix) {
    return {
        matrix(0, 0) * point.x + matrix(0, 1) * point.y + matrix(0, 2),
        matrix(1, 0) * point.x + matrix(1, 1) * point.y + matrix(1, 2),
    };
}

}  // namespace

HandLandmarkDetector::HandLandmarkDetector(const std::string& modelPath, float confidenceThreshold)
    : net_(cv::dnn::readNet(modelPath)), confidenceThreshold_(confidenceThreshold) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::optional<HandResult> HandLandmarkDetector::infer(const cv::Mat& frame, const PalmDetection& palm) {
    const PreprocessResult prep = preprocess(frame, palm);
    if (prep.blob.empty()) {
        return std::nullopt;
    }

    net_.setInput(prep.blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    return postprocess(outputs, prep);
}

HandLandmarkDetector::CropResult HandLandmarkDetector::cropAndPad(const cv::Mat& image,
                                                                  cv::Point2f topLeft,
                                                                  cv::Point2f bottomRight,
                                                                  bool forRotation) const {
    const cv::Point2f shiftVector = forRotation ? cv::Point2f(0.0f, 0.0f) : cv::Point2f(0.0f, -0.4f);
    const float enlargeFactor = forRotation ? 4.0f : 3.0f;

    const cv::Point2f size = bottomRight - topLeft;
    const cv::Point2f shift = multiply(shiftVector, size);
    topLeft += shift;
    bottomRight += shift;

    const cv::Point2f center = (topLeft + bottomRight) * 0.5f;
    const cv::Point2f halfSize = (bottomRight - topLeft) * (enlargeFactor * 0.5f);
    topLeft = center - halfSize;
    bottomRight = center + halfSize;

    const int x1 = std::clamp(cvRound(topLeft.x), 0, image.cols - 1);
    const int y1 = std::clamp(cvRound(topLeft.y), 0, image.rows - 1);
    const int x2 = std::clamp(cvRound(bottomRight.x), x1 + 1, image.cols);
    const int y2 = std::clamp(cvRound(bottomRight.y), y1 + 1, image.rows);

    const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat crop = image(roi).clone();

    const int sideLength = forRotation
                               ? static_cast<int>(std::ceil(std::sqrt(static_cast<double>(crop.rows * crop.rows + crop.cols * crop.cols))))
                               : std::max(crop.rows, crop.cols);
    const int padW = sideLength - crop.cols;
    const int padH = sideLength - crop.rows;
    const int left = padW / 2;
    const int top = padH / 2;

    cv::Mat padded;
    cv::copyMakeBorder(crop, padded, top, padH - top, left, padW - left,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return {
        padded,
        cv::Point2f(static_cast<float>(x1), static_cast<float>(y1)),
        cv::Point2f(static_cast<float>(x2), static_cast<float>(y2)),
        cv::Point2f(static_cast<float>(x1 - left), static_cast<float>(y1 - top)),
    };
}

HandLandmarkDetector::PreprocessResult HandLandmarkDetector::preprocess(const cv::Mat& frame,
                                                                        const PalmDetection& palm) const {
    if (frame.empty()) {
        throw std::invalid_argument("Cannot preprocess an empty frame.");
    }

    const cv::Point2f palmTopLeft(palm.box.x, palm.box.y);
    const cv::Point2f palmBottomRight(palm.box.x + palm.box.width, palm.box.y + palm.box.height);
    const CropResult initialCrop = cropAndPad(frame, palmTopLeft, palmBottomRight, true);

    cv::Mat rgb;
    cv::cvtColor(initialCrop.image, rgb, cv::COLOR_BGR2RGB);

    const cv::Point2f padBias = initialCrop.bias;
    const cv::Point2f localTopLeft = initialCrop.topLeft - padBias;
    const cv::Point2f localBottomRight = initialCrop.bottomRight - padBias;

    std::array<cv::Point2f, 7> localLandmarks{};
    for (size_t i = 0; i < palm.landmarks.size(); ++i) {
        localLandmarks[i] = palm.landmarks[i] - padBias;
    }

    const cv::Point2f p1 = localLandmarks[kPalmBaseIndex];
    const cv::Point2f p2 = localLandmarks[kMiddleFingerBaseIndex];
    double radians = kPi / 2.0 - std::atan2(-(p2.y - p1.y), p2.x - p1.x);
    radians = radians - 2.0 * kPi * std::floor((radians + kPi) / (2.0 * kPi));
    const double angleDegrees = radians * 180.0 / kPi;

    const cv::Point2f palmCenter = (localTopLeft + localBottomRight) * 0.5f;
    const cv::Mat rotation = cv::getRotationMatrix2D(palmCenter, angleDegrees, 1.0);
    cv::Matx23f rotationMatrix;
    rotation.convertTo(rotationMatrix, CV_32F);

    cv::Mat rotated;
    cv::warpAffine(rgb, rotated, rotationMatrix, rgb.size());

    cv::Point2f minPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f maxPoint(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (const cv::Point2f& point : localLandmarks) {
        const cv::Point2f rotatedPoint = transformPoint(point, rotationMatrix);
        minPoint.x = std::min(minPoint.x, rotatedPoint.x);
        minPoint.y = std::min(minPoint.y, rotatedPoint.y);
        maxPoint.x = std::max(maxPoint.x, rotatedPoint.x);
        maxPoint.y = std::max(maxPoint.y, rotatedPoint.y);
    }

    const CropResult finalCrop = cropAndPad(rotated, minPoint, maxPoint, false);
    cv::Mat resized;
    cv::resize(finalCrop.image, resized, inputSize_, 0.0, 0.0, cv::INTER_AREA);

    return {
        makeNhwcBlobFromRgb(resized),
        finalCrop.topLeft,
        finalCrop.bottomRight,
        static_cast<float>(angleDegrees),
        rotationMatrix,
        padBias,
    };
}

std::optional<HandResult> HandLandmarkDetector::postprocess(const std::vector<cv::Mat>& outputs,
                                                            const PreprocessResult& prep) const {
    if (outputs.size() < 4) {
        throw std::runtime_error("Hand landmark model returned fewer outputs than expected.");
    }

    const float confidence = outputs[1].ptr<float>()[0];
    if (confidence < confidenceThreshold_) {
        return std::nullopt;
    }

    const float handedness = outputs[2].ptr<float>()[0];
    const float* landmarkData = outputs[0].ptr<float>();
    const float* worldData = outputs[3].ptr<float>();

    const cv::Point2f rotatedSize = prep.rotatedBottomRight - prep.rotatedTopLeft;
    const float scaleFactor = std::max(rotatedSize.x / inputSize_.width,
                                       rotatedSize.y / inputSize_.height);

    const cv::Mat coordRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), prep.angleDegrees, 1.0);
    cv::Matx23f coordRotationMatrix;
    coordRotation.convertTo(coordRotationMatrix, CV_32F);

    cv::Mat inverseRotation;
    cv::invertAffineTransform(cv::Mat(prep.rotationMatrix), inverseRotation);
    cv::Matx23f inverseRotationMatrix;
    inverseRotation.convertTo(inverseRotationMatrix, CV_32F);

    const cv::Point2f rotatedCenter = (prep.rotatedTopLeft + prep.rotatedBottomRight) * 0.5f;
    const cv::Point2f originalCenter = transformPoint(rotatedCenter, inverseRotationMatrix);

    HandResult result;
    result.handedness = handedness;
    result.confidence = confidence;

    cv::Point2f minPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f maxPoint(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    for (int i = 0; i < kHandLandmarkCount; ++i) {
        cv::Point3f point(
            (landmarkData[i * 3] - inputSize_.width * 0.5f) * scaleFactor,
            (landmarkData[i * 3 + 1] - inputSize_.height * 0.5f) * scaleFactor,
            landmarkData[i * 3 + 2] * scaleFactor);

        const float rotatedX = point.x * coordRotationMatrix(0, 0) + point.y * coordRotationMatrix(1, 0);
        const float rotatedY = point.x * coordRotationMatrix(0, 1) + point.y * coordRotationMatrix(1, 1);
        point.x = rotatedX + originalCenter.x + prep.padBias.x;
        point.y = rotatedY + originalCenter.y + prep.padBias.y;

        result.landmarks[static_cast<size_t>(i)] = point;
        minPoint.x = std::min(minPoint.x, point.x);
        minPoint.y = std::min(minPoint.y, point.y);
        maxPoint.x = std::max(maxPoint.x, point.x);
        maxPoint.y = std::max(maxPoint.y, point.y);

        cv::Point3f worldPoint(worldData[i * 3], worldData[i * 3 + 1], worldData[i * 3 + 2]);
        const float worldRotatedX = worldPoint.x * coordRotationMatrix(0, 0) + worldPoint.y * coordRotationMatrix(1, 0);
        const float worldRotatedY = worldPoint.x * coordRotationMatrix(0, 1) + worldPoint.y * coordRotationMatrix(1, 1);
        result.worldLandmarks[static_cast<size_t>(i)] = cv::Point3f(worldRotatedX, worldRotatedY, worldPoint.z);
    }

    const cv::Point2f boxSize = maxPoint - minPoint;
    const cv::Point2f shiftedMin = minPoint + multiply(cv::Point2f(0.0f, -0.1f), boxSize);
    const cv::Point2f shiftedMax = maxPoint + multiply(cv::Point2f(0.0f, -0.1f), boxSize);
    const cv::Point2f center = (shiftedMin + shiftedMax) * 0.5f;
    const cv::Point2f halfSize = (shiftedMax - shiftedMin) * (1.65f * 0.5f);
    result.box = cv::Rect2f(center - halfSize, center + halfSize);

    return result;
}
