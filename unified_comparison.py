import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.transforms import functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

# File paths
video_path = r"C:\Users\sabih\OneDrive\Desktop\CV Project\IMG_0075.MP4"
output_dir = r"C:\Users\sabih\OneDrive\Desktop\CV Project\Unified_Comparison"
os.makedirs(output_dir, exist_ok=True)

# Output files
yolo_video_path = os.path.join(output_dir, "yolov8_output.mp4")
frcnn_video_path = os.path.join(output_dir, "frcnn_output.mp4")
ssd_video_path = os.path.join(output_dir, "ssd_output.mp4")
side_by_side_video_path = os.path.join(output_dir, "comparison_side_by_side.avi")
stats_output_path = os.path.join(output_dir, "unified_comparison_stats.txt")
csv_output_path = os.path.join(output_dir, "unified_comparison_data.csv")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Parameters
speed_limit = 55  # km/h
frame_width_meters = 20.5  # Width of the scene in meters
confidence_threshold = 0.4  # Minimum confidence for object detection
max_frames_to_process = None  # Set to a number to limit processing, None for full video

# Global ID mapping system
global_track_ids = {}  # Maps (model, local_id) to global_id
next_global_id = 1  # Counter for assigning global IDs
id_match_threshold = 0.7  # IoU threshold for matching detections across models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models
print("Loading models...")
# YOLOv8
yolo_model_path = r"C:\Users\sabih\OneDrive\Desktop\CV Project\YOLOv8\yolov8n.pt"
yolo_model = YOLO(yolo_model_path)

# Faster R-CNN
frcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
frcnn_model.to(device)
frcnn_model.eval()

# SSD
ssd_model = ssd300_vgg16(pretrained=True)
ssd_model.to(device)
ssd_model.eval()

# COCO class indices for vehicles 
vehicle_classes = [2, 5, 7]  # Car, bus, truck (in zero-based indexing)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pixel_to_meter_ratio = frame_width_meters / frame_width
print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")

# Reduce output resolution for MP4 compatibility
output_width = 1920
output_height = 1080

# Calculate aspect ratio-preserving dimensions
aspect_ratio = frame_width / frame_height
if (aspect_ratio > (output_width / output_height)):  # width is the limiting factor
    new_width = output_width
    new_height = int(output_width / aspect_ratio)
else:  # height is the limiting factor
    new_height = output_height
    new_width = int(output_height * aspect_ratio)

# Initialize video writers with reduced resolution
yolo_out = cv2.VideoWriter(yolo_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
frcnn_out = cv2.VideoWriter(frcnn_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
ssd_out = cv2.VideoWriter(ssd_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
side_by_side_out = cv2.VideoWriter(side_by_side_video_path, cv2.VideoWriter_fourcc(*'XVID'), 
                                  fps, (new_width*3, new_height))

# Helper function to prepare image for PyTorch models
def prepare_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to tensor
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0).to(device)

# Function to run detection with YOLOv8
def detect_yolo(frame):
    start_time = time.time()
    results = yolo_model(frame)
    elapsed = time.time() - start_time
    
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if cls in vehicle_classes and conf > confidence_threshold:
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'cls': cls
            })
    
    return detections, elapsed

# Function to run detection with Faster R-CNN
def detect_frcnn(frame):
    image_tensor = prepare_image(frame)
    
    start_time = time.time()
    with torch.no_grad():
        prediction = frcnn_model(image_tensor)
    elapsed = time.time() - start_time
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        # Adjust for 1-indexed labels in COCO dataset
        label_idx = int(label) - 1
        if label_idx in vehicle_classes and score > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': float(score),
                'cls': label_idx
            })
    
    return detections, elapsed

# Function to run detection with SSD
def detect_ssd(frame):
    image_tensor = prepare_image(frame)
    
    start_time = time.time()
    with torch.no_grad():
        prediction = ssd_model(image_tensor)
    elapsed = time.time() - start_time
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        # Adjust for 1-indexed labels in COCO dataset
        label_idx = int(label) - 1
        if label_idx in vehicle_classes and score > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': float(score),
                'cls': label_idx
            })
    
    return detections, elapsed

# Function to convert detection format for DeepSORT
def format_for_deepsort(detections):
    tracker_input = []
    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        tracker_input.append(([x1, y1, w, h], det['conf'], None))
    return tracker_input

# Function to track vehicles and calculate speed
def track_and_calculate_speed(detections, tracker, vehicle_positions, vehicle_speeds, vehicle_max_speeds, frame_count, frame):
    # Convert to DeepSORT format
    tracker_input = format_for_deepsort(detections)
    
    # Update tracker
    tracks = tracker.update_tracks(tracker_input, frame=frame)
    
    speeds = {}
    tracked_vehicles = set()
    
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
            
        # Get track ID and bounding box
        track_id = track.track_id
        tracked_vehicles.add(track_id)
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate center position
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Calculate speed
        if track_id in vehicle_positions:
            prev_center, prev_frame = vehicle_positions[track_id]
            
            # Calculate distance in pixels
            dist_pixels = np.linalg.norm(np.array(center) - np.array(prev_center))
            
            # Convert to meters
            dist_meters = dist_pixels * pixel_to_meter_ratio
            time_diff = (frame_count - prev_frame) / fps
            
            # Simplified speed calculation in km/h
            if time_diff > 0:
                speed = (dist_meters / time_diff) * 3.6
                speeds[track_id] = speed
                vehicle_speeds[track_id] = speed
                
                # Update max speed
                if track_id not in vehicle_max_speeds or speed > vehicle_max_speeds[track_id]:
                    vehicle_max_speeds[track_id] = speed
        
        # Store current position
        vehicle_positions[track_id] = (center, frame_count)
    
    return speeds, tracked_vehicles, tracks  # Return tracks as well

# Function to calculate IoU between bounding boxes
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If the intersection is empty, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

# Function to assign consistent global IDs across different models by matching detections based on spatial overlap (IoU)
def get_global_id(model_name, local_id, bbox, frame_detections):
    """
    Assign consistent global IDs across different models by matching detections
    based on spatial overlap (IoU)
    """
    global next_global_id, global_track_ids
    
    # If we've already assigned a global ID for this model+local_id
    model_key = (model_name, local_id)
    if model_key in global_track_ids:
        return global_track_ids[model_key]
    
    # Calculate IoU with detections from other models in the same frame
    best_match_id = None
    best_match_iou = 0
    
    # Check if this detection matches any from other models
    for other_model, other_dets in frame_detections.items():
        if other_model == model_name:
            continue
            
        for other_id, other_bbox in other_dets.items():
            # Calculate IoU between bboxes
            iou = calculate_iou(bbox, other_bbox)
            
            # If there's a good match with an already globally tracked object
            if iou > id_match_threshold and iou > best_match_iou:
                other_model_key = (other_model, other_id)
                if other_model_key in global_track_ids:
                    best_match_id = global_track_ids[other_model_key]
                    best_match_iou = iou
    
    # If we found a good match with existing global ID
    if best_match_id is not None:
        global_track_ids[model_key] = best_match_id
        return best_match_id
    
    # Otherwise, assign a new global ID
    global_track_ids[model_key] = next_global_id
    next_global_id += 1
    return global_track_ids[model_key]

# Function to draw detection and tracking results with alert system
def draw_results(frame, tracks, speeds, speeders, model_name, speed_limit, frame_detections):
    result_frame = frame.copy()
    
    # Alert system variables
    show_alert = False
    alert_speed = 0
    alert_id = None
    
    # Store current frame's detections for ID matching
    if model_name not in frame_detections:
        frame_detections[model_name] = {}
    
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
            
        # Get track ID and bounding box
        local_track_id = track.track_id
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Store bbox for global ID matching
        frame_detections[model_name][local_track_id] = [x1, y1, x2, y2]
        
        # Get global ID
        global_id = get_global_id(model_name, local_track_id, [x1, y1, x2, y2], frame_detections)
        
        # Use global ID for display and tracking
        track_id = global_id
        
        # Default color (green)
        color = (0, 255, 0)
        
        # Add speed label
        label = f"ID:{track_id}"
        if local_track_id in speeds:
            speed = speeds[local_track_id]
            label += f" {int(speed)}km/h"
            
            # Mark speeding vehicles in red
            if speed > speed_limit:
                color = (0, 0, 255)
                # Update alert system
                if not show_alert or speed > alert_speed:
                    show_alert = True
                    alert_speed = speed
                    alert_id = track_id
        
        # Draw bounding box and label
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        # Add background for text
        cv2.rectangle(result_frame, (x1, y1-20), (x1 + len(label)*8, y1), color, -1)
        cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add model info and stats
    cv2.putText(result_frame, f"Model: {model_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, f"Speed Limit: {speed_limit} km/h", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, f"Vehicles: {len(speeds)} | Speeders: {len(speeders)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Alert system overlay for speeding vehicles
    if show_alert:
        # Add red alert banner at the top
        cv2.rectangle(result_frame, (0, 120), (frame_width, 180), (0, 0, 255), -1)
        alert_text = f"SPEEDING ALERT! ID: {alert_id}, Speed: {int(alert_speed)} km/h"
        cv2.putText(result_frame, alert_text, (frame_width//2 - 180, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    return result_frame

# Initialize trackers
yolo_tracker = DeepSort(max_age=30, n_init=3)
frcnn_tracker = DeepSort(max_age=30, n_init=3)
ssd_tracker = DeepSort(max_age=30, n_init=3)

# Storage for positions and speeds
yolo_positions = {}
frcnn_positions = {}
ssd_positions = {}
yolo_speeds = {}
frcnn_speeds = {}
ssd_speeds = {}
yolo_max_speeds = {}
frcnn_max_speeds = {}
ssd_max_speeds = {}
yolo_speeders = set()
frcnn_speeders = set()
ssd_speeders = set()

# Speed violations log
yolo_speed_violations = []
frcnn_speed_violations = []
ssd_speed_violations = []

# Track frame-by-frame data
frame_data = {
    'frame': [],
    'yolo_detections': [], 
    'frcnn_detections': [], 
    'ssd_detections': [],
    'yolo_tracked': [], 
    'frcnn_tracked': [], 
    'ssd_tracked': [],
    'yolo_time': [], 
    'frcnn_time': [], 
    'ssd_time': []
}

# Performance metrics
metrics = {
    'yolo': {
        'detection_times': [], 
        'tracking_times': [],
        'detections_per_frame': [], 
        'speeds': [], 
        'tracked_vehicles_count': []
    },
    'frcnn': {
        'detection_times': [], 
        'tracking_times': [],
        'detections_per_frame': [], 
        'speeds': [], 
        'tracked_vehicles_count': []
    },
    'ssd': {
        'detection_times': [], 
        'tracking_times': [],
        'detections_per_frame': [], 
        'speeds': [], 
        'tracked_vehicles_count': []
    }
}

print("Starting unified model comparison...")
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or (max_frames_to_process is not None and frame_count >= max_frames_to_process):
        break
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}/{total_frames}")
    
    # Reset detection storage for each frame
    frame_detections = {}
    
    # Run detections with all models
    yolo_dets, yolo_time = detect_yolo(frame)
    frcnn_dets, frcnn_time = detect_frcnn(frame)
    ssd_dets, ssd_time = detect_ssd(frame)
    
    # Track vehicles and calculate speeds
    yolo_tracking_start = time.time()
    yolo_frame_speeds, yolo_tracked, yolo_tracks = track_and_calculate_speed(
        yolo_dets, yolo_tracker, yolo_positions, yolo_speeds, yolo_max_speeds, frame_count, frame
    )
    yolo_tracking_time = time.time() - yolo_tracking_start
    
    frcnn_tracking_start = time.time()
    frcnn_frame_speeds, frcnn_tracked, frcnn_tracks = track_and_calculate_speed(
        frcnn_dets, frcnn_tracker, frcnn_positions, frcnn_speeds, frcnn_max_speeds, frame_count, frame
    )
    frcnn_tracking_time = time.time() - frcnn_tracking_start
    
    ssd_tracking_start = time.time()
    ssd_frame_speeds, ssd_tracked, ssd_tracks = track_and_calculate_speed(
        ssd_dets, ssd_tracker, ssd_positions, ssd_speeds, ssd_max_speeds, frame_count, frame
    )
    ssd_tracking_time = time.time() - ssd_tracking_start
    
    # Check for speeders and log violations
    for vehicle_id, speed in yolo_speeds.items():
        if speed > speed_limit and vehicle_id not in yolo_speeders:
            yolo_speeders.add(vehicle_id)
            timestamp = frame_count / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            violation = {
                "id": vehicle_id,
                "speed": speed,
                "time": f"{minutes:02d}:{seconds:02d}",
                "frame": frame_count
            }
            yolo_speed_violations.append(violation)
            
    for vehicle_id, speed in frcnn_speeds.items():
        if speed > speed_limit and vehicle_id not in frcnn_speeders:
            frcnn_speeders.add(vehicle_id)
            timestamp = frame_count / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            violation = {
                "id": vehicle_id,
                "speed": speed,
                "time": f"{minutes:02d}:{seconds:02d}",
                "frame": frame_count
            }
            frcnn_speed_violations.append(violation)
            
    for vehicle_id, speed in ssd_speeds.items():
        if speed > speed_limit and vehicle_id not in ssd_speeders:
            ssd_speeders.add(vehicle_id)
            timestamp = frame_count / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            violation = {
                "id": vehicle_id,
                "speed": speed,
                "time": f"{minutes:02d}:{seconds:02d}",
                "frame": frame_count
            }
            ssd_speed_violations.append(violation)
    
    # Store frame metrics
    frame_data['frame'].append(frame_count)
    frame_data['yolo_detections'].append(len(yolo_dets))
    frame_data['frcnn_detections'].append(len(frcnn_dets))
    frame_data['ssd_detections'].append(len(ssd_dets))
    frame_data['yolo_tracked'].append(len(yolo_tracked))
    frame_data['frcnn_tracked'].append(len(frcnn_tracked))
    frame_data['ssd_tracked'].append(len(ssd_tracked))
    frame_data['yolo_time'].append(yolo_time + yolo_tracking_time)
    frame_data['frcnn_time'].append(frcnn_time + frcnn_tracking_time)
    frame_data['ssd_time'].append(ssd_time + ssd_tracking_time)
    
    # Store metrics
    metrics['yolo']['detection_times'].append(yolo_time)
    metrics['yolo']['tracking_times'].append(yolo_tracking_time)
    metrics['yolo']['detections_per_frame'].append(len(yolo_dets))
    metrics['yolo']['tracked_vehicles_count'].append(len(yolo_tracked))
    for speed in yolo_frame_speeds.values():
        metrics['yolo']['speeds'].append(speed)
    
    metrics['frcnn']['detection_times'].append(frcnn_time)
    metrics['frcnn']['tracking_times'].append(frcnn_tracking_time)
    metrics['frcnn']['detections_per_frame'].append(len(frcnn_dets))
    metrics['frcnn']['tracked_vehicles_count'].append(len(frcnn_tracked))
    for speed in frcnn_frame_speeds.values():
        metrics['frcnn']['speeds'].append(speed)
    
    metrics['ssd']['detection_times'].append(ssd_time)
    metrics['ssd']['tracking_times'].append(ssd_tracking_time)
    metrics['ssd']['detections_per_frame'].append(len(ssd_dets))
    metrics['ssd']['tracked_vehicles_count'].append(len(ssd_tracked))
    for speed in ssd_frame_speeds.values():
        metrics['ssd']['speeds'].append(speed)
    
    # Draw results on frames with alert system
    yolo_frame = draw_results(frame, yolo_tracks, yolo_speeds, yolo_speeders, "YOLOv8", speed_limit, frame_detections)
    frcnn_frame = draw_results(frame, frcnn_tracks, frcnn_speeds, frcnn_speeders, "Faster R-CNN", speed_limit, frame_detections)
    ssd_frame = draw_results(frame, ssd_tracks, ssd_speeds, ssd_speeders, "SSD", speed_limit, frame_detections)

    # Resize frames for video output
    yolo_frame_resized = cv2.resize(yolo_frame, (new_width, new_height))
    frcnn_frame_resized = cv2.resize(frcnn_frame, (new_width, new_height))
    ssd_frame_resized = cv2.resize(ssd_frame, (new_width, new_height))

    # Create side-by-side comparison at reduced size
    side_by_side = np.hstack((yolo_frame_resized, frcnn_frame_resized, ssd_frame_resized))

    # Write individual and side-by-side frames
    yolo_out.write(yolo_frame_resized)
    frcnn_out.write(frcnn_frame_resized)
    ssd_out.write(ssd_frame_resized)
    side_by_side_out.write(side_by_side)
    
    # Display (optional, can be disabled for faster processing)
    if frame_count % 30 == 0:  # Show every 30th frame
        cv2.imshow("Unified Comparison", cv2.resize(side_by_side, (0, 0), fx=0.6, fy=0.6))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
yolo_out.release()
frcnn_out.release()
ssd_out.release()
side_by_side_out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time

# Calculate summary statistics
summary = {
    'yolo': {
        'avg_detection_time': np.mean(metrics['yolo']['detection_times']),
        'avg_tracking_time': np.mean(metrics['yolo']['tracking_times']),
        'total_avg_time': np.mean(metrics['yolo']['detection_times']) + np.mean(metrics['yolo']['tracking_times']),
        'fps': 1.0 / (np.mean(metrics['yolo']['detection_times']) + np.mean(metrics['yolo']['tracking_times'])),
        'avg_detections': np.mean(metrics['yolo']['detections_per_frame']),
        'avg_tracked': np.mean(metrics['yolo']['tracked_vehicles_count']),
        'avg_speed': np.mean(metrics['yolo']['speeds']) if metrics['yolo']['speeds'] else 0,
        'max_speed': np.max(list(yolo_max_speeds.values())) if yolo_max_speeds else 0,
        'total_unique_vehicles': len(yolo_max_speeds),
        'total_speeders': len(yolo_speeders),
        'speeder_percentage': (len(yolo_speeders) / len(yolo_max_speeds) * 100) if yolo_max_speeds else 0
    },
    'frcnn': {
        'avg_detection_time': np.mean(metrics['frcnn']['detection_times']),
        'avg_tracking_time': np.mean(metrics['frcnn']['tracking_times']),
        'total_avg_time': np.mean(metrics['frcnn']['detection_times']) + np.mean(metrics['frcnn']['tracking_times']),
        'fps': 1.0 / (np.mean(metrics['frcnn']['detection_times']) + np.mean(metrics['frcnn']['tracking_times'])),
        'avg_detections': np.mean(metrics['frcnn']['detections_per_frame']),
        'avg_tracked': np.mean(metrics['frcnn']['tracked_vehicles_count']),
        'avg_speed': np.mean(metrics['frcnn']['speeds']) if metrics['frcnn']['speeds'] else 0,
        'max_speed': np.max(list(frcnn_max_speeds.values())) if frcnn_max_speeds else 0,
        'total_unique_vehicles': len(frcnn_max_speeds),
        'total_speeders': len(frcnn_speeders),
        'speeder_percentage': (len(frcnn_speeders) / len(frcnn_max_speeds) * 100) if frcnn_max_speeds else 0
    },
    'ssd': {
        'avg_detection_time': np.mean(metrics['ssd']['detection_times']),
        'avg_tracking_time': np.mean(metrics['ssd']['tracking_times']),
        'total_avg_time': np.mean(metrics['ssd']['detection_times']) + np.mean(metrics['ssd']['tracking_times']),
        'fps': 1.0 / (np.mean(metrics['ssd']['detection_times']) + np.mean(metrics['ssd']['tracking_times'])),
        'avg_detections': np.mean(metrics['ssd']['detections_per_frame']),
        'avg_tracked': np.mean(metrics['ssd']['tracked_vehicles_count']),
        'avg_speed': np.mean(metrics['ssd']['speeds']) if metrics['ssd']['speeds'] else 0,
        'max_speed': np.max(list(ssd_max_speeds.values())) if ssd_max_speeds else 0,
        'total_unique_vehicles': len(ssd_max_speeds),
        'total_speeders': len(ssd_speeders),
        'speeder_percentage': (len(ssd_speeders) / len(ssd_max_speeds) * 100) if ssd_max_speeds else 0
    }
}

# Create and save plots
print("Generating comparison plots...")

# Create dataframe from frame_data
df = pd.DataFrame(frame_data)

# 1. Detection count per frame
plt.figure(figsize=(12, 6))
plt.plot(df['frame'], df['yolo_detections'], label='YOLOv8')
plt.plot(df['frame'], df['frcnn_detections'], label='Faster R-CNN')
plt.plot(df['frame'], df['ssd_detections'], label='SSD')
plt.xlabel('Frame')
plt.ylabel('Detections')
plt.title('Object Detections per Frame')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, 'detections_per_frame.png'))

# 2. Tracking count per frame
plt.figure(figsize=(12, 6))
plt.plot(df['frame'], df['yolo_tracked'], label='YOLOv8')
plt.plot(df['frame'], df['frcnn_tracked'], label='Faster R-CNN')
plt.plot(df['frame'], df['ssd_tracked'], label='SSD')
plt.xlabel('Frame')
plt.ylabel('Tracked Vehicles')
plt.title('Tracked Vehicles per Frame')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, 'tracking_per_frame.png'))

# 3. Processing time per frame
plt.figure(figsize=(12, 6))
plt.plot(df['frame'], df['yolo_time'], label='YOLOv8')
plt.plot(df['frame'], df['frcnn_time'], label='Faster R-CNN')
plt.plot(df['frame'], df['ssd_time'], label='SSD')
plt.xlabel('Frame')
plt.ylabel('Processing Time (s)')
plt.title('Processing Time per Frame')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, 'processing_time_per_frame.png'))

# 4. Speed distributions
plt.figure(figsize=(12, 6))
sns.kdeplot(metrics['yolo']['speeds'], label='YOLOv8', fill=True, alpha=0.5)
sns.kdeplot(metrics['frcnn']['speeds'], label='Faster R-CNN', fill=True, alpha=0.5)
sns.kdeplot(metrics['ssd']['speeds'], label='SSD', fill=True, alpha=0.5)
plt.axvline(speed_limit, color='red', linestyle='--', label=f'Speed Limit ({speed_limit} km/h)')
plt.xlabel('Speed (km/h)')
plt.ylabel('Density')
plt.title('Speed Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plots_dir, 'speed_distribution.png'))

# 5. Performance comparison bar chart
plt.figure(figsize=(12, 6))
model_names = ['YOLOv8', 'Faster R-CNN', 'SSD']
fps_vals = [summary['yolo']['fps'], summary['frcnn']['fps'], summary['ssd']['fps']]
detection_counts = [summary['yolo']['total_unique_vehicles'], 
                    summary['frcnn']['total_unique_vehicles'],
                    summary['ssd']['total_unique_vehicles']]

x = np.arange(len(model_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
rects1 = ax1.bar(x - width/2, fps_vals, width, label='FPS', color='skyblue')
rects2 = ax2.bar(x + width/2, detection_counts, width, label='Vehicle Count', color='orange')

ax1.set_xlabel('Model')
ax1.set_ylabel('FPS')
ax2.set_ylabel('Vehicle Count')
ax1.set_title('Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'))

# 6. Speeding analysis
plt.figure(figsize=(10, 6))
speeder_counts = [summary['yolo']['total_speeders'],
                  summary['frcnn']['total_speeders'],
                  summary['ssd']['total_speeders']]
speeder_percentages = [summary['yolo']['speeder_percentage'],
                       summary['frcnn']['speeder_percentage'],
                       summary['ssd']['speeder_percentage']]

x = np.arange(len(model_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
rects1 = ax1.bar(x - width/2, speeder_counts, width, label='Speeders', color='crimson')
rects2 = ax2.bar(x + width/2, speeder_percentages, width, label='Speeder %', color='gold')

ax1.set_xlabel('Model')
ax1.set_ylabel('Speeder Count')
ax2.set_ylabel('Speeder Percentage (%)')
ax1.set_title('Speeder Analysis')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'speeder_analysis.png'))

# 7. Speed violation comparison from comparison.py
plt.figure(figsize=(15, 7))

# Calculate speed violation statistics
yolo_speeders_count = len([s for s in metrics['yolo']['speeds'] if s > speed_limit])
frcnn_speeders_count = len([s for s in metrics['frcnn']['speeds'] if s > speed_limit])
ssd_speeders_count = len([s for s in metrics['ssd']['speeds'] if s > speed_limit])

# Create speed violation detection comparison
violation_data = {
    'Total Speeders': [len(yolo_speeders), len(frcnn_speeders), len(ssd_speeders)],
    'Speeding %': [summary['yolo']['speeder_percentage'], 
                  summary['frcnn']['speeder_percentage'], 
                  summary['ssd']['speeder_percentage']],
    'Total Speed Readings': [len(metrics['yolo']['speeds']), 
                           len(metrics['frcnn']['speeds']), 
                           len(metrics['ssd']['speeds'])],
    'Speeding Readings': [yolo_speeders_count, frcnn_speeders_count, ssd_speeders_count]
}

# Plot speed violation comparison
plt.subplot(1, 2, 1)
plt.bar(['YOLOv8', 'Faster R-CNN', 'SSD'], violation_data['Total Speeders'], color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Total Speeding Vehicles Detected')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.subplot(1, 2, 2)
plt.bar(['YOLOv8', 'Faster R-CNN', 'SSD'], violation_data['Speeding %'], color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Percentage of Speeding Vehicles')
plt.ylabel('Percent (%)')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'speed_violation_comparison.png'))

plt.close('all')  # Close all figures

# Write frame-by-frame data to CSV
df.to_csv(csv_output_path, index=False)

# Write comparison report
with open(stats_output_path, 'w') as f:
    f.write("===== UNIFIED MODEL COMPARISON FOR VEHICLE SPEED DETECTION =====\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Video: {os.path.basename(video_path)}\n")
    f.write(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}\n")
    f.write(f"Frames Processed: {frame_count}\n")
    f.write(f"Speed Limit: {speed_limit} km/h\n")
    f.write(f"Confidence Threshold: {confidence_threshold}\n\n")
    
    f.write("===== DETECTION PERFORMANCE =====\n")
    f.write(f"{'Model':<15} {'Detection (s)':<15} {'Tracking (s)':<15} {'Total (s)':<15} {'FPS':<10} {'Avg Detections':<15}\n")
    f.write(f"YOLOv8         {summary['yolo']['avg_detection_time']:.4f}       {summary['yolo']['avg_tracking_time']:.4f}       {summary['yolo']['total_avg_time']:.4f}       {summary['yolo']['fps']:.2f}    {summary['yolo']['avg_detections']:.2f}\n")
    f.write(f"Faster R-CNN   {summary['frcnn']['avg_detection_time']:.4f}       {summary['frcnn']['avg_tracking_time']:.4f}       {summary['frcnn']['total_avg_time']:.4f}       {summary['frcnn']['fps']:.2f}    {summary['frcnn']['avg_detections']:.2f}\n")
    f.write(f"SSD            {summary['ssd']['avg_detection_time']:.4f}       {summary['ssd']['avg_tracking_time']:.4f}       {summary['ssd']['total_avg_time']:.4f}       {summary['ssd']['fps']:.2f}    {summary['ssd']['avg_detections']:.2f}\n\n")
    
    f.write("===== TRACKING RESULTS =====\n")
    f.write(f"{'Model':<15} {'Unique Vehicles':<20} {'Avg Tracked/Frame':<20} {'Speeders':<10} {'% Speeders':<15}\n")
    
    f.write(f"YOLOv8         {summary['yolo']['total_unique_vehicles']:<20} {summary['yolo']['avg_tracked']:.2f}{' '*14} {summary['yolo']['total_speeders']:<10} {summary['yolo']['speeder_percentage']:.2f}%\n")
    f.write(f"Faster R-CNN   {summary['frcnn']['total_unique_vehicles']:<20} {summary['frcnn']['avg_tracked']:.2f}{' '*14} {summary['frcnn']['total_speeders']:<10} {summary['frcnn']['speeder_percentage']:.2f}%\n")
    f.write(f"SSD            {summary['ssd']['total_unique_vehicles']:<20} {summary['ssd']['avg_tracked']:.2f}{' '*14} {summary['ssd']['total_speeders']:<10} {summary['ssd']['speeder_percentage']:.2f}%\n\n")
    
    f.write("===== SPEED ESTIMATION =====\n")
    f.write(f"{'Model':<15} {'Avg Speed (km/h)':<20} {'Max Speed (km/h)':<20}\n")
    
    f.write(f"YOLOv8         {summary['yolo']['avg_speed']:.2f}{' '*14} {summary['yolo']['max_speed']:.2f}\n")
    f.write(f"Faster R-CNN   {summary['frcnn']['avg_speed']:.2f}{' '*14} {summary['frcnn']['max_speed']:.2f}\n")
    f.write(f"SSD            {summary['ssd']['avg_speed']:.2f}{' '*14} {summary['ssd']['max_speed']:.2f}\n\n")
    
    # Speed violation details from comparison.py
    f.write("===== SPEED VIOLATION DETECTION CAPABILITIES =====\n")
    f.write(f"{'Model':<15} {'Total Vehicles':<15} {'Total Speeders':<15} {'% Speeding':<12} {'Avg Speed When Speeding':<25}\n")
    
    # Calculate average speeds of speeders for each model
    yolo_speeder_speeds = [speed for speed in metrics['yolo']['speeds'] if speed > speed_limit]
    frcnn_speeder_speeds = [speed for speed in metrics['frcnn']['speeds'] if speed > speed_limit]
    ssd_speeder_speeds = [speed for speed in metrics['ssd']['speeds'] if speed > speed_limit]
    
    yolo_avg_speeder_speed = np.mean(yolo_speeder_speeds) if yolo_speeder_speeds else 0
    frcnn_avg_speeder_speed = np.mean(frcnn_speeder_speeds) if frcnn_speeder_speeds else 0
    ssd_avg_speeder_speed = np.mean(ssd_speeder_speeds) if ssd_speeder_speeds else 0
    
    f.write(f"YOLOv8         {summary['yolo']['total_unique_vehicles']:<15} {summary['yolo']['total_speeders']:<15} {summary['yolo']['speeder_percentage']:.2f}%{' '*8} {yolo_avg_speeder_speed:.2f} km/h\n")
    f.write(f"Faster R-CNN   {summary['frcnn']['total_unique_vehicles']:<15} {summary['frcnn']['total_speeders']:<15} {summary['frcnn']['speeder_percentage']:.2f}%{' '*8} {frcnn_avg_speeder_speed:.2f} km/h\n")
    f.write(f"SSD            {summary['ssd']['total_unique_vehicles']:<15} {summary['ssd']['total_speeders']:<15} {summary['ssd']['speeder_percentage']:.2f}%{' '*8} {ssd_avg_speeder_speed:.2f} km/h\n\n")
    
    f.write("===== ALERT SYSTEM EFFECTIVENESS =====\n")
    f.write("Speed readings exceeding the limit:\n")
    f.write(f"YOLOv8: {yolo_speeders_count} out of {len(metrics['yolo']['speeds'])} readings ({(yolo_speeders_count/len(metrics['yolo']['speeds'])*100) if len(metrics['yolo']['speeds']) > 0 else 0:.2f}%)\n")
    f.write(f"Faster R-CNN: {frcnn_speeders_count} out of {len(metrics['frcnn']['speeds'])} readings ({(frcnn_speeders_count/len(metrics['frcnn']['speeds'])*100) if len(metrics['frcnn']['speeds']) > 0 else 0:.2f}%)\n")
    f.write(f"SSD: {ssd_speeders_count} out of {len(metrics['ssd']['speeds'])} readings ({(ssd_speeders_count/len(metrics['ssd']['speeds'])*100) if len(metrics['ssd']['speeds']) > 0 else 0:.2f}%)\n\n")
    
    # Add detailed speed violations log
    f.write("===== DETAILED SPEED VIOLATIONS =====\n")
    
    f.write("\nYOLOv8 Speed Violations:\n")
    if yolo_speed_violations:
        f.write("ID\tSpeed (km/h)\tTimestamp\tFrame\n")
        for v in yolo_speed_violations:
            f.write(f"{v['id']}\t{v['speed']:.2f}\t\t{v['time']}\t\t{v['frame']}\n")
    else:
        f.write("No speed violations detected.\n")
    
    f.write("\nFaster R-CNN Speed Violations:\n")
    if frcnn_speed_violations:
        f.write("ID\tSpeed (km/h)\tTimestamp\tFrame\n")
        for v in frcnn_speed_violations:
            f.write(f"{v['id']}\t{v['speed']:.2f}\t\t{v['time']}\t\t{v['frame']}\n")
    else:
        f.write("No speed violations detected.\n")
    
    f.write("\nSSD Speed Violations:\n")
    if ssd_speed_violations:
        f.write("ID\tSpeed (km/h)\tTimestamp\tFrame\n")
        for v in ssd_speed_violations:
            f.write(f"{v['id']}\t{v['speed']:.2f}\t\t{v['time']}\t\t{v['frame']}\n")
    else:
        f.write("No speed violations detected.\n")
        
    # Determine which model is most effective at detecting speeders
    effectiveness_ranking = sorted(['yolo', 'frcnn', 'ssd'], 
                                key=lambda x: (violation_data['Total Speeders'][['yolo', 'frcnn', 'ssd'].index(x)], 
                                            violation_data['Speeding %'][['yolo', 'frcnn', 'ssd'].index(x)]),
                                reverse=True)
    
    model_names_dict = {'yolo': 'YOLOv8', 'frcnn': 'Faster R-CNN', 'ssd': 'SSD'}
    
    f.write("\nAlert System Ranking (based on speeder detection):\n")
    f.write(" > ".join([model_names_dict[name] for name in effectiveness_ranking]) + "\n\n")
    
    f.write("===== SPEEDER ID LIST =====\n")
    yolo_speeder_list = ", ".join(map(str, sorted(yolo_speeders)))
    frcnn_speeder_list = ", ".join(map(str, sorted(frcnn_speeders)))
    ssd_speeder_list = ", ".join(map(str, sorted(ssd_speeders)))
    
    f.write(f"YOLOv8 Speeders: {yolo_speeder_list if yolo_speeders else 'None'}\n\n")
    f.write(f"Faster R-CNN Speeders: {frcnn_speeder_list if frcnn_speeders else 'None'}\n\n")
    f.write(f"SSD Speeders: {ssd_speeder_list if ssd_speeders else 'None'}\n\n")
    
    # Calculate agreement between models
    yolo_frcnn_agreement = len(yolo_speeders.intersection(frcnn_speeders))
    yolo_ssd_agreement = len(yolo_speeders.intersection(ssd_speeders))
    frcnn_ssd_agreement = len(frcnn_speeders.intersection(ssd_speeders))
    all_models_agreement = len(yolo_speeders.intersection(frcnn_speeders).intersection(ssd_speeders))
    
    f.write("===== INTER-MODEL AGREEMENT =====\n")
    f.write(f"YOLOv8 and Faster R-CNN agreement: {yolo_frcnn_agreement} speeders\n")
    f.write(f"YOLOv8 and SSD agreement: {yolo_ssd_agreement} speeders\n")
    f.write(f"Faster R-CNN and SSD agreement: {frcnn_ssd_agreement} speeders\n")
    f.write(f"All models agree on: {all_models_agreement} speeders\n\n")
    
    f.write("===== OVERALL PERFORMANCE SUMMARY =====\n")
    f.write(f"Total processing time: {total_time:.2f} seconds\n")
    f.write(f"Average processing speed: {frame_count/total_time:.2f} frames/sec\n\n")
    
    f.write("===== CONCLUSION =====\n")
    
    # Determine best performer for speed (FPS)
    fps_models = ['YOLOv8', 'Faster R-CNN', 'SSD']
    fps_values = [summary['yolo']['fps'], summary['frcnn']['fps'], summary['ssd']['fps']]
    fastest_model = fps_models[fps_values.index(max(fps_values))]
    
    # Determine best performer for detection count
    detection_values = [summary['yolo']['total_unique_vehicles'], summary['frcnn']['total_unique_vehicles'], summary['ssd']['total_unique_vehicles']]
    most_detections_model = fps_models[detection_values.index(max(detection_values))]
    
    f.write(f"Fastest model: {fastest_model} ({max(fps_values):.2f} FPS)\n")
    f.write(f"Most detected vehicles: {most_detections_model} ({max(detection_values)} vehicles)\n")
    
    # General conclusion
    f.write("\nRecommendation based on results:\n")
    if fastest_model == most_detections_model:
        f.write(f"{fastest_model} provides the best balance of speed and detection performance.\n")
    else:
        f.write(f"If prioritizing speed: {fastest_model}\n")
        f.write(f"If prioritizing detection accuracy: {most_detections_model}\n")

print(f"Unified comparison complete!")
print(f"Individual videos saved to:")
print(f"- YOLOv8: {yolo_video_path}")
print(f"- Faster R-CNN: {frcnn_video_path}")
print(f"- SSD: {ssd_video_path}")
print(f"Side-by-side comparison: {side_by_side_video_path}")
print(f"Statistics saved to: {stats_output_path}")
print(f"CSV data saved to: {csv_output_path}")
print(f"Plots saved to: {plots_dir}")