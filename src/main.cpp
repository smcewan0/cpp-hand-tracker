#include "FpsCounter.h"
#include "HandLandmarkDetector.h"
#include "LandmarkSmoother.h"
#include "PalmDetector.h"
#include "Visualization.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

constexpr std::array<int, 7> kPalmLandmarkIds = {0, 5, 9, 13, 17, 1, 2};

struct FrameOverlayTracker {
    FrameOverlay overlay;
    int validFrames{0};
    int missingFrames{0};
};

std::filesystem::path executableDirectory(char** argv) {
#ifdef _WIN32
    std::vector<char> buffer(MAX_PATH);
    DWORD size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    while (size == buffer.size()) {
        buffer.resize(buffer.size() * 2);
        size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    }
    if (size > 0) {
        return std::filesystem::path(std::string(buffer.data(), size)).parent_path();
    }
#endif
    return std::filesystem::absolute(argv[0]).parent_path();
}

std::string resolveAssetPath(const std::string& requestedPath, const std::filesystem::path& exeDir) {
    const std::filesystem::path path(requestedPath);
    if (std::filesystem::exists(path)) {
        return path.string();
    }

    const std::array<std::filesystem::path, 3> candidates = {
        exeDir / path,
        exeDir.parent_path() / path,
        std::filesystem::current_path() / path,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    return requestedPath;
}

void waitOnLaunchError() {
#ifdef _WIN32
    std::cerr << "Press Enter to exit...";
    std::string ignored;
    std::getline(std::cin, ignored);
#endif
}

float distance2d(const cv::Point3f& lhs, const cv::Point3f& rhs) {
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f handCenter(const HandResult& hand) {
    cv::Point2f sum(0.0f, 0.0f);
    for (const cv::Point3f& point : hand.landmarks) {
        sum += cv::Point2f(point.x, point.y);
    }
    return sum * (1.0f / static_cast<float>(hand.landmarks.size()));
}

float handScale(const HandResult& hand) {
    const float wristToMiddleMcp = distance2d(hand.landmarks[0], hand.landmarks[9]);
    const float boxDiagonal = std::sqrt(hand.box.width * hand.box.width + hand.box.height * hand.box.height);
    return std::max({wristToMiddleMcp, boxDiagonal * 0.28f, 30.0f});
}

void updatePinchState(HandResult& hand) {
    hand.pinchDistance = distance2d(hand.landmarks[4], hand.landmarks[8]);
    const float diagonal = std::sqrt(hand.box.width * hand.box.width + hand.box.height * hand.box.height);
    const float threshold = std::max(20.0f, diagonal * 0.08f);
    hand.pinching = hand.pinchDistance < threshold;
}

bool fingerFolded(const HandResult& hand, int mcp, int pip, int tip) {
    const cv::Point3f& wrist = hand.landmarks[0];
    return distance2d(hand.landmarks[tip], wrist) < distance2d(hand.landmarks[pip], wrist) * 1.08f ||
           distance2d(hand.landmarks[tip], hand.landmarks[mcp]) < distance2d(hand.landmarks[pip], hand.landmarks[mcp]) * 1.25f;
}

bool fingerExtended(const HandResult& hand, int mcp, int pip, int tip) {
    const cv::Point3f& wrist = hand.landmarks[0];
    return distance2d(hand.landmarks[tip], wrist) > distance2d(hand.landmarks[pip], wrist) * 1.10f &&
           distance2d(hand.landmarks[tip], hand.landmarks[mcp]) > distance2d(hand.landmarks[pip], hand.landmarks[mcp]) * 1.04f;
}

void updateGesture(HandResult& hand) {
    updatePinchState(hand);

    const float scale = handScale(hand);
    const bool indexFolded = fingerFolded(hand, 5, 6, 8);
    const bool middleFolded = fingerFolded(hand, 9, 10, 12);
    const bool ringFolded = fingerFolded(hand, 13, 14, 16);
    const bool pinkyFolded = fingerFolded(hand, 17, 18, 20);
    const int foldedFingers = static_cast<int>(indexFolded) +
                              static_cast<int>(middleFolded) +
                              static_cast<int>(ringFolded) +
                              static_cast<int>(pinkyFolded);

    const bool thumbExtended = distance2d(hand.landmarks[4], hand.landmarks[0]) >
                               distance2d(hand.landmarks[3], hand.landmarks[0]) * 1.10f;
    const bool thumbPointsUp = hand.landmarks[4].y < hand.landmarks[2].y - scale * 0.20f &&
                               hand.landmarks[4].y < hand.landmarks[0].y - scale * 0.25f;

    if (hand.pinching) {
        hand.gesture = "PINCH";
    } else if (foldedFingers >= 3 && thumbExtended && thumbPointsUp) {
        hand.gesture = "THUMBS UP";
    } else {
        hand.gesture = "OPEN";
    }
}

bool overlapsTrackedPalm(const PalmDetection& palm, const std::vector<HandResult>& trackedHands) {
    const cv::Point2f palmCenter(palm.box.x + palm.box.width * 0.5f,
                                 palm.box.y + palm.box.height * 0.5f);

    for (const HandResult& tracked : trackedHands) {
        const float centerDistance = static_cast<float>(cv::norm(palmCenter - handCenter(tracked)));
        if (centerDistance < handScale(tracked) * 1.35f) {
            return true;
        }
    }

    return false;
}

bool overlapsTrackedHand(const HandResult& candidate, const std::vector<HandResult>& trackedHands) {
    const cv::Point2f candidateCenter = handCenter(candidate);
    const float candidateScale = handScale(candidate);

    for (const HandResult& tracked : trackedHands) {
        const float centerDistance = static_cast<float>(cv::norm(candidateCenter - handCenter(tracked)));
        const float threshold = std::max(candidateScale, handScale(tracked)) * 1.15f;
        if (centerDistance < threshold) {
            return true;
        }
    }

    return false;
}

void discoverAdditionalHands(cv::Mat& frame,
                             PalmDetector& palmDetector,
                             HandLandmarkDetector& handDetector,
                             std::vector<HandResult>& hands,
                             int maxHands) {
    std::vector<PalmDetection> palms = palmDetector.infer(frame);
    for (const PalmDetection& palm : palms) {
        if (static_cast<int>(hands.size()) >= maxHands) {
            break;
        }
        if (overlapsTrackedPalm(palm, hands)) {
            continue;
        }

        std::optional<HandResult> hand = handDetector.infer(frame, palm);
        if (hand.has_value() && !overlapsTrackedHand(*hand, hands)) {
            hands.push_back(*hand);
        }
    }
}

PalmDetection palmFromTrackedHand(const HandResult& hand) {
    PalmDetection palm;
    palm.score = hand.confidence;

    cv::Point2f minPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f maxPoint(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < kPalmLandmarkIds.size(); ++i) {
        const cv::Point3f& source = hand.landmarks[static_cast<size_t>(kPalmLandmarkIds[i])];
        const cv::Point2f point(source.x, source.y);
        palm.landmarks[i] = point;
        minPoint.x = std::min(minPoint.x, point.x);
        minPoint.y = std::min(minPoint.y, point.y);
        maxPoint.x = std::max(maxPoint.x, point.x);
        maxPoint.y = std::max(maxPoint.y, point.y);
    }

    const cv::Point2f center = (minPoint + maxPoint) * 0.5f;
    const cv::Point2f size = maxPoint - minPoint;
    const float side = std::max({size.x, size.y, hand.box.width * 0.25f, hand.box.height * 0.25f, 24.0f});
    palm.box = cv::Rect2f(center.x - side * 0.5f,
                          center.y - side * 0.5f,
                          side,
                          side);
    return palm;
}

std::vector<HandResult> updateFromTrackedHands(const cv::Mat& frame,
                                               HandLandmarkDetector& handDetector,
                                               const std::vector<HandResult>& trackedHands,
                                               int maxHands) {
    std::vector<HandResult> hands;
    hands.reserve(std::min(static_cast<int>(trackedHands.size()), maxHands));

    for (const HandResult& previous : trackedHands) {
        if (static_cast<int>(hands.size()) >= maxHands) {
            break;
        }

        std::optional<HandResult> hand = handDetector.infer(frame, palmFromTrackedHand(previous));
        if (hand.has_value()) {
            hands.push_back(*hand);
        }
    }

    return hands;
}

std::vector<HandResult> relocalizeHands(const cv::Mat& frame,
                                        PalmDetector& palmDetector,
                                        HandLandmarkDetector& handDetector,
                                        int maxHands) {
    std::vector<PalmDetection> palms = palmDetector.infer(frame);
    if (static_cast<int>(palms.size()) > maxHands) {
        palms.resize(static_cast<size_t>(maxHands));
    }

    std::vector<HandResult> hands;
    hands.reserve(palms.size());
    for (const PalmDetection& palm : palms) {
        std::optional<HandResult> hand = handDetector.infer(frame, palm);
        if (hand.has_value()) {
            hands.push_back(*hand);
        }
    }

    return hands;
}

cv::Point toPoint(const cv::Point3f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

cv::Point2f toPoint2f(const cv::Point3f& point) {
    return {point.x, point.y};
}

bool isFrameCornerPose(const HandResult& hand) {
    const float scale = handScale(hand);
    const cv::Point2f thumbTip = toPoint2f(hand.landmarks[4]);
    const cv::Point2f indexTip = toPoint2f(hand.landmarks[8]);

    const bool indexIsOut = fingerExtended(hand, 5, 6, 8);
    const float thumbSpread = static_cast<float>(cv::norm(thumbTip - indexTip));
    const float verticalSpread = thumbTip.y - indexTip.y;
    const float horizontalSpread = std::abs(thumbTip.x - indexTip.x);

    return indexIsOut &&
           thumbSpread > scale * 0.24f &&
           thumbSpread < scale * 2.05f &&
           verticalSpread > scale * 0.05f &&
           horizontalSpread > scale * 0.05f;
}

bool buildRawFrameOverlay(const std::vector<HandResult>& hands, FrameOverlay& overlay) {
    if (hands.size() < 2) {
        return false;
    }

    const HandResult* first = &hands[0];
    const HandResult* second = &hands[1];
    if (handCenter(*first).x > handCenter(*second).x) {
        std::swap(first, second);
    }

    if (!isFrameCornerPose(*first) || !isFrameCornerPose(*second)) {
        return false;
    }

    const float leftScale = handScale(*first);
    const float rightScale = handScale(*second);
    const float averageScale = (leftScale + rightScale) * 0.5f;
    const float handSeparation = std::abs(handCenter(*second).x - handCenter(*first).x);
    if (handSeparation < std::max(95.0f, averageScale * 1.10f)) {
        return false;
    }

    const std::array<cv::Point2f, 2> leftPair = {
        toPoint2f(first->landmarks[8]),
        toPoint2f(first->landmarks[4]),
    };
    const std::array<cv::Point2f, 2> rightPair = {
        toPoint2f(second->landmarks[8]),
        toPoint2f(second->landmarks[4]),
    };

    if (cv::norm(leftPair[0] - leftPair[1]) < leftScale * 0.20f ||
        cv::norm(rightPair[0] - rightPair[1]) < rightScale * 0.20f) {
        return false;
    }

    const cv::Point2f leftTop = leftPair[0].y <= leftPair[1].y ? leftPair[0] : leftPair[1];
    const cv::Point2f leftBottom = leftPair[0].y > leftPair[1].y ? leftPair[0] : leftPair[1];
    const cv::Point2f rightTop = rightPair[0].y <= rightPair[1].y ? rightPair[0] : rightPair[1];
    const cv::Point2f rightBottom = rightPair[0].y > rightPair[1].y ? rightPair[0] : rightPair[1];

    const float topWidth = static_cast<float>(cv::norm(rightTop - leftTop));
    const float bottomWidth = static_cast<float>(cv::norm(rightBottom - leftBottom));
    const float leftHeight = static_cast<float>(cv::norm(leftBottom - leftTop));
    const float rightHeight = static_cast<float>(cv::norm(rightBottom - rightTop));
    if (std::min(topWidth, bottomWidth) < std::max(80.0f, averageScale * 0.95f) ||
        std::min(leftHeight, rightHeight) < std::max(22.0f, averageScale * 0.18f)) {
        return false;
    }

    overlay.active = true;
    overlay.corners = {leftTop, rightTop, rightBottom, leftBottom};
    return true;
}

FrameOverlay updateFrameOverlay(const std::vector<HandResult>& hands, FrameOverlayTracker& tracker) {
    FrameOverlay raw;
    if (!buildRawFrameOverlay(hands, raw)) {
        tracker.validFrames = 0;
        ++tracker.missingFrames;
        if (tracker.missingFrames > 18) {
            tracker.overlay.active = false;
        }
        return tracker.overlay;
    }

    ++tracker.validFrames;
    tracker.missingFrames = 0;

    if (!tracker.overlay.active && tracker.validFrames < 2) {
        return tracker.overlay;
    }

    constexpr float kAlpha = 0.24f;
    if (!tracker.overlay.active) {
        tracker.overlay = raw;
    } else {
        for (size_t i = 0; i < tracker.overlay.corners.size(); ++i) {
            tracker.overlay.corners[i] = tracker.overlay.corners[i] * (1.0f - kAlpha) +
                                         raw.corners[i] * kAlpha;
        }
        tracker.overlay.active = true;
    }

    return tracker.overlay;
}

}  // namespace

int main(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ? |      | Show usage}"
        "{palm_model     | models/palm_detection_mediapipe_2023feb.onnx | Path to palm detector ONNX model}"
        "{hand_model     | models/handpose_estimation_mediapipe_2023feb.onnx | Path to hand landmark ONNX model}"
        "{camera         | 0    | Camera index}"
        "{max_hands      | 2    | Maximum hands to track}"
        "{frame_width    | 640  | Requested capture width}"
        "{frame_height   | 360  | Requested capture height}"
        "{target_fps     | 35   | Requested camera FPS}"
        "{detect_every   | 0    | Run full palm detection every N frames while tracking; 0 means only after loss}"
        "{discover_every | 24   | Search for additional hands every N frames while under max_hands}"
        "{lost_grace     | 5    | Number of missed frames to keep the last good hand visible}"
        "{palm_score     | 0.65 | Palm detection confidence threshold}"
        "{landmark_score | 0.55 | Hand landmark confidence threshold}"
        "{smooth_alpha   | 0.42 | Exponential smoothing alpha, 0..1}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    try {
        const std::filesystem::path exeDir = executableDirectory(argv);
        const std::string palmModel = resolveAssetPath(parser.get<std::string>("palm_model"), exeDir);
        const std::string handModel = resolveAssetPath(parser.get<std::string>("hand_model"), exeDir);

        cv::setUseOptimized(true);
        cv::setNumThreads(std::max(1U, std::thread::hardware_concurrency()));

        PalmDetector palmDetector(palmModel, parser.get<float>("palm_score"));
        HandLandmarkDetector handDetector(handModel,
                                          parser.get<float>("landmark_score"));
        LandmarkSmoother smoother(parser.get<float>("smooth_alpha"));
        FpsCounter fpsCounter;

#ifdef _WIN32
        cv::VideoCapture capture(parser.get<int>("camera"), cv::CAP_DSHOW);
#else
        cv::VideoCapture capture(parser.get<int>("camera"), cv::CAP_ANY);
#endif
        if (!capture.isOpened()) {
            std::cerr << "Could not open camera " << parser.get<int>("camera") << ".\n";
            waitOnLaunchError();
            return 1;
        }

        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        capture.set(cv::CAP_PROP_FRAME_WIDTH, parser.get<int>("frame_width"));
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, parser.get<int>("frame_height"));
        capture.set(cv::CAP_PROP_FPS, parser.get<int>("target_fps"));
        capture.set(cv::CAP_PROP_BUFFERSIZE, 1);

        cv::namedWindow("Finger Tracker", cv::WINDOW_NORMAL);
        const int maxHands = std::max(1, parser.get<int>("max_hands"));
        const int detectEvery = std::max(0, parser.get<int>("detect_every"));
        const int discoverEvery = std::max(1, parser.get<int>("discover_every"));
        const int lostGrace = std::max(0, parser.get<int>("lost_grace"));
        int frameIndex = 0;
        int missedFrames = 0;
        std::vector<HandResult> trackedHands;
        FrameOverlayTracker overlayTracker;

        while (true) {
            cv::Mat frame;
            capture >> frame;
            if (frame.empty()) {
                std::cerr << "Camera returned an empty frame.\n";
                break;
            }

            std::vector<HandResult> hands;
            const bool hasTrack = !trackedHands.empty();
            const bool periodicRelocalize = hasTrack && detectEvery > 0 && frameIndex % detectEvery == 0;
            const bool discoverMoreHands = hasTrack && frameIndex % discoverEvery == 0;
            bool heldPreviousHands = false;

            if (hasTrack) {
                hands = updateFromTrackedHands(frame, handDetector, trackedHands, maxHands);
            }

            if (hands.empty()) {
                ++missedFrames;
                hands = relocalizeHands(frame, palmDetector, handDetector, maxHands);
                if (hands.empty() && missedFrames <= lostGrace && hasTrack) {
                    hands = trackedHands;
                    heldPreviousHands = true;
                } else if (hands.empty()) {
                    missedFrames = lostGrace + 1;
                }
            } else if (periodicRelocalize) {
                std::vector<HandResult> relocalizedHands = relocalizeHands(frame, palmDetector, handDetector, maxHands);
                if (relocalizedHands.size() > hands.size()) {
                    hands = std::move(relocalizedHands);
                }
            } else if (discoverMoreHands && static_cast<int>(hands.size()) < maxHands) {
                discoverAdditionalHands(frame, palmDetector, handDetector, hands, maxHands);
            }

            smoother.smooth(hands);
            for (HandResult& hand : hands) {
                updateGesture(hand);
            }

            if (!hands.empty() && !heldPreviousHands) {
                missedFrames = 0;
            }

            const double fps = fpsCounter.tick();
            const FrameOverlay overlay = updateFrameOverlay(hands, overlayTracker);
            drawFrameOverlay(frame, overlay);
            drawHands(frame, hands);
            drawHud(frame, fps, static_cast<int>(hands.size()));
            trackedHands = hands;
            ++frameIndex;

            cv::imshow("Finger Tracker", frame);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        waitOnLaunchError();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        waitOnLaunchError();
        return 1;
    }

    return 0;
}
