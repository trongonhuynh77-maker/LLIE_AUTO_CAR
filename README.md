# 🚗 Autonomous Vehicle Navigation using LLIE & YOLO on Jetson Orin Nano

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: Jetson Orin Nano](https://img.shields.io/badge/Hardware-Jetson_Orin_Nano_67_TOPS-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin-nano)
[![Camera: IMX335](https://img.shields.io/badge/Camera-IMX335-blue)](https://www.sony-semicon.com/)

This is the official repository for the conference paper: **"[Your Paper Title Here]"**.

* **Author:** Phạm Hà Gia Bảo
* **Supervisor:** Dr. Dương Minh Thiện

This project provides the complete control source code for an autonomous scale vehicle utilizing an Ackerman steering mechanism. By combining deep learning algorithms such as **LLIE (Low-Light Image Enhancement)** for complex lighting environments and **YOLO** for object detection, the system achieves robust, flexible, and real-time operation at the edge (Edge AI).

## 🌟 Key Features

The system is designed to navigate complex tracks and traffic scenarios:
- 🛣️ **Lane Keeping:** Automatically detects lanes and calculates the steering angle to maintain the vehicle in the center.
- 🛑 **Traffic Sign Recognition:** Recognizes speed limits, directional signs, and stop signs using YOLO.
- 🚧 **Obstacle Avoidance:** Detects static/dynamic obstacles on the road to adjust the trajectory or apply emergency braking.
- 🌉 **Bridge Navigation:** Adjusts speed and handles camera pitch angle changes when traversing slopes.
- 🚦 **Intersection Navigation:** Identifies intersections, processes traffic signals, and makes safe turning decisions.

## 🛠️ Hardware Setup

- **Chassis:** Scale car with an Ackerman steering mechanism.
- **Compute:** NVIDIA Jetson Orin Nano (67 TOPS) providing multi-threading capabilities for Vision and Control pipelines.
- **Camera:** Sony IMX335 sensor providing High Dynamic Range (HDR) input, supporting the LLIE algorithm to restore details in dark areas.

## 📂 Project Structure

Based on the current project structure[cite: 1], the system is modularized for easy development and maintenance:

```text
├── best10.pt              # Trained YOLO weights for traffic signs and obstacles detection[cite: 1]
├── config.py              # System configuration parameters (PID, camera params, speed limits)[cite: 1]
├── main.py                # Main execution script[cite: 1]
├── navigator.py           # Navigation algorithms for intersections and bridges[cite: 1]
├── proc_hardware.py       # Hardware communication (motor and steering servo control)[cite: 1]
├── proc_vision.py         # Computer vision pipeline: LLIE enhancement and YOLO inference[cite: 1]
├── proc_web.py            # Backend data stream processing for web monitoring[cite: 1]
├── utils.py               # Utility functions (math operations, data format conversion)[cite: 1]
└── web_server.py          # Web server for remote monitoring and control[cite: 1]
