#include "Visualization.h"

#include <array>
#include <cmath>
#include <string>

namespace {

const std::array<std::pair<int, int>, 21> kConnections{{
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {5, 9}, {9, 10}, {10, 11}, {11, 12},
    {9, 13}, {13, 14}, {14, 15}, {15, 16},
    {13, 17}, {17, 18}, {18, 19}, {19, 20},
    {0, 17},
}};

cv::Point toPoint(const cv::Point3f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

}  // namespace

void drawHands(cv::Mat& frame, const std::vector<HandResult>& hands) {
    for (const HandResult& hand : hands) {
        const cv::Scalar shadowColor(0, 0, 0);
        const cv::Scalar lineColor(255, 255, 255);
        const cv::Scalar pointColor(255, 255, 255);

        for (const auto& connection : kConnections) {
            cv::line(frame,
                     toPoint(hand.landmarks[static_cast<size_t>(connection.first)]),
                     toPoint(hand.landmarks[static_cast<size_t>(connection.second)]),
                     shadowColor, 3, cv::LINE_AA);
            cv::line(frame,
                     toPoint(hand.landmarks[static_cast<size_t>(connection.first)]),
                     toPoint(hand.landmarks[static_cast<size_t>(connection.second)]),
                     lineColor, 1, cv::LINE_AA);
        }

        for (size_t i = 0; i < hand.landmarks.size(); ++i) {
            const int radius = i == 8 ? 5 : 2;
            cv::circle(frame, toPoint(hand.landmarks[i]), radius + 1, shadowColor, cv::FILLED, cv::LINE_AA);
            cv::circle(frame, toPoint(hand.landmarks[i]), radius, pointColor, cv::FILLED, cv::LINE_AA);
        }

        cv::circle(frame, toPoint(hand.landmarks[8]), 9, shadowColor, 2, cv::LINE_AA);
        cv::circle(frame, toPoint(hand.landmarks[8]), 8, lineColor, 1, cv::LINE_AA);

        if (hand.pinching) {
            cv::line(frame, toPoint(hand.landmarks[4]), toPoint(hand.landmarks[8]),
                     shadowColor, 4, cv::LINE_AA);
            cv::line(frame, toPoint(hand.landmarks[4]), toPoint(hand.landmarks[8]),
                     lineColor, 2, cv::LINE_AA);
        }

        if (hand.gesture != "OPEN") {
            const cv::Point labelOrigin = toPoint(hand.landmarks[0]) + cv::Point(12, -18);
            cv::putText(frame, hand.gesture, labelOrigin,
                        cv::FONT_HERSHEY_SIMPLEX, 0.58, shadowColor, 3, cv::LINE_AA);
            cv::putText(frame, hand.gesture, labelOrigin,
                        cv::FONT_HERSHEY_SIMPLEX, 0.58, lineColor, 1, cv::LINE_AA);
        }
    }
}

void drawFrameOverlay(cv::Mat& frame, const FrameOverlay& overlay) {
    if (!overlay.active) {
        return;
    }

    std::vector<cv::Point> polygon;
    polygon.reserve(overlay.corners.size());
    for (const cv::Point2f& corner : overlay.corners) {
        polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }

    cv::Rect bounds = cv::boundingRect(polygon) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (bounds.empty()) {
        return;
    }

    std::vector<cv::Point> localPolygon;
    localPolygon.reserve(polygon.size());
    for (const cv::Point& point : polygon) {
        localPolygon.emplace_back(point.x - bounds.x, point.y - bounds.y);
    }

    cv::Mat frameRoi = frame(bounds);
    cv::Mat layer = frameRoi.clone();
    const cv::Scalar purple(70, 18, 95);
    cv::fillConvexPoly(layer, localPolygon, purple, cv::LINE_AA);
    cv::addWeighted(layer, 0.58, frameRoi, 0.42, 0.0, frameRoi);

    cv::polylines(frame, polygon, true, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::polylines(frame, polygon, true, cv::Scalar(210, 190, 225), 1, cv::LINE_AA);
}

void drawHud(cv::Mat& frame, double fps, int handCount) {
    const std::string fpsText = cv::format("FPS: %.1f", fps);
    const std::string statusText = handCount > 0 ? cv::format("Tracking: %d hand(s)", handCount) : "Tracking: searching";

    cv::rectangle(frame, cv::Rect(8, 8, 250, 58), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, fpsText, cv::Point(18, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, statusText, cv::Point(18, 56),
                cv::FONT_HERSHEY_SIMPLEX, 0.58,
                handCount > 0 ? cv::Scalar(255, 255, 255) : cv::Scalar(180, 180, 180),
                1, cv::LINE_AA);
}
