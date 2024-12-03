import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import random
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
import yaml
from pathlib import Path
import threading
import queue
from src.DetectionAnalytics import DetectionAnalytics
from src.detection_types import Detection
from src.core.control_panel import ControlPanel
from .detector_config import DetectorConfig

class DetectionZone:
    def __init__(self, points: List[Tuple[int, int]]):
        self.points = points
        self.active = True
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(np.array(self.points), point, False) >= 0

class TrackerFactory:
    @staticmethod
    def create_tracker():
        # First try modern OpenCV tracking implementations
        modern_trackers = [
            (lambda: cv2.TrackerCSRT_create(), 'CSRT'),
            (lambda: cv2.TrackerKCF_create(), 'KCF')
        ]
        
        # Then try legacy implementations
        legacy_trackers = [
            (lambda: cv2.legacy.TrackerCSRT_create(), 'Legacy CSRT'),
            (lambda: cv2.legacy.TrackerKCF_create(), 'Legacy KCF')
        ]
        
        # Try modern implementations first
        for creator, name in modern_trackers:
            try:
                tracker = creator()
                logging.info(f"Successfully created {name} tracker")
                return tracker
            except (AttributeError, cv2.error):
                continue
        
        # If modern implementations fail, try legacy
        for creator, name in legacy_trackers:
            try:
                tracker = creator()
                logging.info(f"Successfully created {name} tracker")
                return tracker
            except (AttributeError, cv2.error):
                continue
        
        # If all attempts fail, log error and return None
        logging.error("No compatible tracker found")
        print("Warning: No compatible tracker found. Tracking functionality will be disabled.")
        return None
        for creator, name in trackers:
            try:
                tracker = creator()
                logging.info(f"Successfully created {name} tracker")
                return tracker
            except AttributeError:
                continue
        
        logging.error("No compatible tracker found")
        return None

class DetectionHistory:
    def __init__(self, max_length: int = 50):
        self.history: Dict[str, deque] = {}
        self.max_length = max_length
    
    def add_detection(self, detection: Detection):
        if detection.class_name not in self.history:
            self.history[detection.class_name] = deque(maxlen=self.max_length)
        self.history[detection.class_name].append(detection.confidence)
    
    def get_average_confidence(self, class_name: str) -> float:
        if class_name not in self.history or not self.history[class_name]:
            return 0.0
        return sum(self.history[class_name]) / len(self.history[class_name])

class VideoWriter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.writer = None
        self.recording = False
    
    def start_recording(self, frame: np.ndarray):
        if self.recording:
            return
        
        filename = self.output_dir / f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (frame.shape[1], frame.shape[0])
        self.writer = cv2.VideoWriter(str(filename), fourcc, 20.0, frame_size)
        self.recording = True
        logging.info(f"Started recording: {filename}")
    
    def stop_recording(self):
        if not self.recording:
            return
        
        self.writer.release()
        self.writer = None
        self.recording = False
        logging.info("Recording stopped")
    
    def write_frame(self, frame: np.ndarray):
        if self.recording and self.writer is not None:
            self.writer.write(frame)

class ObjectDetector:
    def __init__(self, config: DetectorConfig):
        print("Initializing ObjectDetector...")
        self.config = config
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Initialize state variables first
        self.current_model_index = 0
        self.tracking_enabled = False
        self.tracking_box = None
        self.show_fps = True
        self.show_help = False
        self.detection_zones: List[DetectionZone] = []
        self.drawing_zone = False
        self.temp_zone: List[Tuple[int, int]] = []
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize components
        self.setup_logging()
        print("Setting up video writer...")
        self.video_writer = VideoWriter(config.output_dir)
        print("Initializing detection history...")
        self.history = DetectionHistory(config.history_length)
        print("Creating tracker...")
        self.tracker = TrackerFactory.create_tracker()
        
        # Load model
        print(f"Loading initial model: {config.available_models[0]}...")
        self.model = self.load_model()
        print("Model loaded successfully!")
        
        # Add UI components last
        print("Initializing analytics and control panel...")
        self.analytics = DetectionAnalytics(Path(config.output_dir))
        self.control_panel = ControlPanel(self)
        
        # Threading setup for async processing
        self.detection_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_detections, daemon=True)
        self.processing_thread.start()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'detector.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_model(self) -> YOLO:
        model_path = self.config.available_models[self.current_model_index]
        logging.info(f"Loading model: {model_path}")
        return YOLO(model_path)
    
    def switch_model(self) -> YOLO:
        self.current_model_index = (self.current_model_index + 1) % len(self.config.available_models)
        self.model = self.load_model()
        return self.model
    
    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, stream=True)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.config.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_name=class_name,
                    class_id=cls
                )
                
                # Check detection zones if any exist
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                if not self.detection_zones or any(zone.contains_point(center_point) for zone in self.detection_zones):
                    detections.append(detection)
                    self.history.add_detection(detection)
        
        return detections
    
    def process_detections(self):
        while True:
            frame, detections = self.detection_queue.get()
            if frame is None:
                break
            
            # Process detections asynchronously
            # Add any heavy processing here
            
            self.detection_queue.task_done()
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]):
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.get_color(detection.class_name)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.class_name}: {detection.confidence*100:.1f}%"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        if not hasattr(self, 'colors'):
            self.colors = {}
        
        if class_name not in self.colors:
            self.colors[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.colors[class_name]
    
    @property
    def current_model(self) -> str:
        return self.config.available_models[self.current_model_index]
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def main():
    # Load configuration
    print("Initializing object detector...")
    config = DetectorConfig()
    detector = ObjectDetector(config)
    
    # Start UI in separate thread
    print("Starting UI control panel...")
    ui_thread = threading.Thread(target=detector.control_panel.run, daemon=True)
    ui_thread.start()
    
    # Setup video capture
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    print("Camera opened successfully! Starting detection loop...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera!")
                break
            
            # Process frame and get detections
            detections = detector.process_frame(frame)
            
            # Add to async processing queue
            detector.detection_queue.put((frame.copy(), detections))
            
            # Update analytics
            for detection in detections:
                detector.analytics.add_event(detection)
            
            # Draw detections
            detector.draw_detections(frame, detections)
            
            # Handle video recording
            if detector.video_writer.recording:
                detector.video_writer.write_frame(frame)
            
            # Display frame
            cv2.imshow('Enhanced Object Detection', frame)
            
            # Handle keyboard input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit command received. Shutting down...")
                break
            
    except Exception as e:
        print(f"Critical error in main loop: {e}")
        logging.error(f"Error in main loop: {e}")
        
    finally:
        # Cleanup
        print("Cleaning up resources...")
        detector.detection_queue.put((None, None))  # Signal processing thread to stop
        detector.video_writer.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. Program terminated.")

if __name__ == "__main__":
    main()