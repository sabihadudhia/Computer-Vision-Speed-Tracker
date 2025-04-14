# Vehicle Speed Detection Comparison

This project provides a comparison between three state-of-the-art object detection models (**YOLOv8**, **Faster R-CNN**, and **SSD**) for vehicle speed detection and traffic monitoring. The comparison script processes video footage, detects vehicles, tracks them across frames, calculates their speeds, and identifies speeding violations.

---

## Features

- **Multi-Model Comparison**  
  Simultaneously runs YOLOv8, Faster R-CNN, and SSD on the same video for side-by-side comparison  
- **Consistent Vehicle Tracking**  
  DeepSORT algorithm with global ID system maintains consistent vehicle IDs across models  
- **Speed Estimation**  
  Calculates vehicle speeds based on pixel displacement and real-world scale  
- **Speeding Violation Detection**  
  Identifies vehicles exceeding a defined speed limit  
- **Alert System**  
  Displays visual alerts for speeding vehicles with ID and speed info  

### Analytics

- Detection performance (FPS, detection time)  
- Tracking statistics  
- Speed distributions  
- Speeding violation summaries  

### Visualizations

- Individual output videos per model  
- Side-by-side comparison video  
- Comparative charts and graphs  

---

## Requirements

- Python 3.8+  
- PyTorch  
- Torchvision  
- OpenCV  
- Ultralytics YOLOv8  
- `deep-sort-realtime`  
- NumPy  
- Matplotlib  
- Pandas  
- Seaborn  

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-speed-detection-comparison.git
cd vehicle-speed-detection-comparison
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 Weights

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0/yolov8n.pt
```

---

## Usage

1. **Configure** the script (e.g., `video_path`, `speed_limit`, `frame_width_meters`, etc.)
2. **Run the comparison:**

```bash
python unified_comparison.py
```

3. **View Results:**
   - YOLOv8: `yolov8_output.mp4`  
   - Faster R-CNN: `frcnn_output.mp4`  
   - SSD: `ssd_output.mp4`  
   - Side-by-side: `comparison_side_by_side.avi`  
   - Stats: `unified_comparison_stats.txt`, `unified_comparison_data.csv`  
   - Charts: `plots/` directory  

---

## Key Parameters

- `speed_limit`: Speed threshold in km/h  
- `frame_width_meters`: Real-world scene width for speed calculation  
- `confidence_threshold`: Minimum confidence for detection  
- `max_frames_to_process`: Optional limit on processed frames  
- `id_match_threshold`: IoU threshold for cross-model ID consistency  

---

## License

This project is licensed under the [MIT License](LICENSE).
```

---
