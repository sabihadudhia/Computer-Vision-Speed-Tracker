# Vehicle Speed Detection Comparison

## Overview
Compare state-of-the-art object detection models (YOLOv8, Faster R-CNN, SSD) for vehicle detection, speed estimation, and traffic monitoring. The project tracks vehicles across video frames and identifies speeding violations.

## Features
- Multi-model comparison on the same video
- Consistent vehicle tracking using DeepSORT
- Speed estimation based on pixel displacement and real-world scale
- Detect and alert for speeding vehicles
- Generate analytics: detection performance, tracking statistics, speed distributions
- Visualizations: individual output videos, side-by-side comparison, charts

## Technologies
- Python 3.8+
- PyTorch, Torchvision
- OpenCV
- Ultralytics YOLOv8
- deep-sort-realtime
- NumPy, pandas, matplotlib, seaborn

## Setup / Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-speed-detection-comparison.git
cd vehicle-speed-detection-comparison
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0/yolov8n.pt
```

## Usage
- Configure script parameters (video_path, speed_limit, frame_width_meters, etc.)
- Run the comparison:
```bash
python unified_comparison.py
```
- Output files:
  - YOLOv8 video: yolov8_output.mp4
  - Faster R-CNN video: frcnn_output.mp4
  - SSD video: ssd_output.mp4
  - Side-by-side comparison: comparison_side_by_side.avi
  - Stats: unified_comparison_stats.txt, unified_comparison_data.csv
  - Charts in plots/ directory
- Key Parameters
  - speed_limit: Speed threshold in km/h
  - frame_width_meters: Real-world scene width for speed calculation
  - confidence_threshold: Minimum detection confidence
  - max_frames_to_process: Optional frame limit
  - id_match_threshold: IoU threshold for cross-model ID consistency

## Project Structure
```bash
├── unified_comparison.py    # Main script
├── videos/                  # Input/output videos
├── plots/                   # Charts and visualizations
├── yolov8n.pt               # YOLOv8 weights
└── README.md                # Documentation
```
