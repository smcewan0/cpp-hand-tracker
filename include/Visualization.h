#pragma once

#include "Types.h"

#include <opencv2/opencv.hpp>

#include <vector>

void drawHands(cv::Mat& frame, const std::vector<HandResult>& hands);
void drawFrameOverlay(cv::Mat& frame, const FrameOverlay& overlay);
void drawHud(cv::Mat& frame, double fps, int handCount);
