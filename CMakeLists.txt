cmake_minimum_required(VERSION 3.24)

project(FingerTracker VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_executable(finger_tracker
    src/main.cpp
    src/FpsCounter.cpp
    src/HandLandmarkDetector.cpp
    src/LandmarkSmoother.cpp
    src/PalmDetector.cpp
    src/Visualization.cpp
)

target_include_directories(finger_tracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(finger_tracker PRIVATE
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(finger_tracker PRIVATE /W4 /permissive-)
else()
    target_compile_options(finger_tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()
