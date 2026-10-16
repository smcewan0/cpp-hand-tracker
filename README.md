# 🖐️ C++ Hand Tracker

**Real-time hand and fingertip tracking powered by C++, OpenCV, and ONNX.**

A computer vision application that detects and tracks hand landmarks in real time using a webcam. Built with a focus on performance, accuracy, and smooth tracking.

## 🚀 Features

- 🖐️ **Hand Tracking** — Detects and tracks hand landmarks in real time.
- 📏 **Distance Estimation** — Calculates estimated distances between fingertips.
- 📊 **Live FPS Counter** — Displays real-time application performance.
- 🎯 **Landmark Smoothing** — Reduces jitter for more stable tracking.
- 👁️ **Palm Detection** — Identifies hands using a dedicated detection model.
- ⚡ **Optimized C++** — Built using C++17 with OpenCV's DNN module.

## 💻 Tech Stack

- **Language:** C++17
- **Computer Vision:** OpenCV
- **Inference:** ONNX models via OpenCV DNN
- **Build System:** CMake
- **Dependencies:** vcpkg
- **Models:** MediaPipe-derived palm detection and hand landmark models

## 📸 Demo

Real-time fingertip detection, hand landmark visualization, and performance monitoring.



https://github.com/user-attachments/assets/0fb435db-31e2-4da0-bc9c-9e45a724c8dc





## 📁 Project Structure

```text
cpp-hand-tracker/
├── include/          # Header files
├── src/              # C++ implementation
├── models/           # ONNX detection models
├── scripts/          # Build and setup scripts
├── CMakeLists.txt     # CMake configuration
└── vcpkg.json        # Dependency manifest
```

## ⚙️ Building

### Requirements

- C++17-compatible compiler
- CMake 3.24+
- OpenCV (core, imgproc, highgui, videoio, dnn)
- vcpkg (recommended)
- Webcam

### Build Instructions

Configure the project with CMake and your vcpkg toolchain:

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

Build the application:

```bash
cmake --build build --config Release
```

Run the resulting `finger_tracker` executable from the appropriate build output directory.

Ensure the required ONNX models are available at the paths expected by the application.

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

**Developed by Sam** ☕  
*Turning caffeine into code.*
