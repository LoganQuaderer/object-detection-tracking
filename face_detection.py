import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import random
import json
import os
from collections import deque

class ObjectDetector:
    def __init__(self):
        self.recording = False
        self.video_writer = None
        self.show_fps = True
        self.confidence_threshold = 0.5
        self.colors = {}
        self.tracking_enabled = False
        self.object_history = {}
        self.history_length = 50
        self.current_model = 'yolov8n.pt'
        self.available_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        self.model_index = 0
        self.show_help = False
        self.detection_zones = []
        self.drawing_zone = False
        self.temp_zone = []
        self.tracker = None
        self.tracking_box = None
        
        # Create output directory
        self.output_dir = 'detection_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load detection history
        self.load_detection_history()
        
        # Initialize tracker
        self.init_tracker()
    
    def init_tracker(self):
        """Initialize the appropriate tracker based on OpenCV version"""
        if hasattr(cv2, 'TrackerCSRT_create'):
            self.tracker = cv2.TrackerCSRT_create()
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            self.tracker = cv2.legacy.TrackerCSRT_create()
        else:
            # Fallback to KCF tracker
            if hasattr(cv2, 'TrackerKCF_create'):
                self.tracker = cv2.TrackerKCF_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
                self.tracker = cv2.legacy.TrackerKCF_create()
            else:
                print("Warning: No compatible tracker found. Tracking disabled.")
                self.tracker = None
    
    def load_detection_history(self):
        history_file = os.path.join(self.output_dir, 'detection_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.object_history = json.load(f)
            except:
                self.object_history = {}
    
    def save_detection_history(self):
        history_file = os.path.join(self.output_dir, 'detection_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.object_history, f)
    
    def get_color(self, class_name):
        if class_name not in self.colors:
            self.colors[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.colors[class_name]
    
    def toggle_recording(self, frame):
        if not self.recording:
            filename = os.path.join(self.output_dir, 
                                  f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
            self.recording = True
            print(f"Started recording: {filename}")
        else:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped")
    
    def start_tracking(self, frame, bbox):
        """Start tracking an object"""
        if self.tracker is None:
            self.init_tracker()
            if self.tracker is None:
                return False
        
        self.tracking_box = bbox
        try:
            return self.tracker.init(frame, bbox)
        except Exception as e:
            print(f"Error initializing tracker: {e}")
            return False

    def update_tracking(self, frame):
        """Update tracking and return the new bounding box"""
        if self.tracker is None or not self.tracking_enabled or self.tracking_box is None:
            return None, False
        
        try:
            success, bbox = self.tracker.update(frame)
            return bbox if success else None, success
        except Exception as e:
            print(f"Error updating tracker: {e}")
            return None, False
    
    def draw_detection_zones(self, frame):
        # Draw existing zones
        for zone in self.detection_zones:
            points = np.array(zone, np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 255), 2)
        
        # Draw zone being created
        if self.drawing_zone and len(self.temp_zone) > 0:
            points = np.array(self.temp_zone, np.int32)
            cv2.polylines(frame, [points], False, (0, 255, 255), 2)
    
    def point_in_zones(self, point):
        for zone in self.detection_zones:
            if cv2.pointPolygonTest(np.array(zone), point, False) >= 0:
                return True
        return False

def main():
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)
    model = YOLO(detector.current_model)
    
    print("\nEnhanced Object Detection v2.0")
    print("\nControls:")
    print("'q' - Quit")
    print("'s' - Screenshot")
    print("'r' - Toggle recording")
    print("'f' - Toggle FPS display")
    print("'t' - Toggle object tracking")
    print("'m' - Switch model (n->s->m)")
    print("'h' - Toggle help overlay")
    print("'z' - Start/stop drawing detection zone")
    print("'c' - Clear all detection zones")
    print("'+/-' - Adjust confidence threshold")
    print("\nClick on an object to start tracking it")
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if detector.drawing_zone:
                detector.temp_zone.append((x, y))
            else:
                bbox = (x-30, y-30, 60, 60)
                if detector.start_tracking(frame, bbox):
                    detector.tracking_enabled = True
                    print("Started tracking object")
                else:
                    print("Failed to initialize tracking")
    
    cv2.namedWindow('Enhanced Object Detection')
    cv2.setMouseCallback('Enhanced Object Detection', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Update object tracking
        if detector.tracking_enabled:
            bbox, tracking_success = detector.update_tracking(frame)
            if tracking_success and bbox is not None:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Tracking", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                detector.tracking_enabled = False
                print("Lost tracking")
        
        # Run YOLO detection
        results = model(frame, stream=True)
        
        # Object counter for current frame
        object_count = {}
        
        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < detector.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Check if object is in any detection zone
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                if len(detector.detection_zones) > 0 and not detector.point_in_zones(center_point):
                    continue
                
                # Update counters and history
                object_count[class_name] = object_count.get(class_name, 0) + 1
                if class_name not in detector.object_history:
                    detector.object_history[class_name] = deque(maxlen=detector.history_length)
                detector.object_history[class_name].append(conf)
                
                # Draw detection
                color = detector.get_color(class_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{class_name}: {conf*100:.1f}%"
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw detection zones
        detector.draw_detection_zones(frame)
        
        # Display object counts and history
        y_pos = 30
        cv2.putText(frame, "Objects Detected:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for obj, count in object_count.items():
            y_pos += 25
            avg_conf = 0
            if obj in detector.object_history and len(detector.object_history[obj]) > 0:
                avg_conf = sum(detector.object_history[obj]) / len(detector.object_history[obj])
            text = f"{obj}: {count} (avg conf: {avg_conf*100:.1f}%)"
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, detector.get_color(obj), 2)
        
        # Show current settings
        if detector.show_fps:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cv2.putText(frame, f'FPS: {fps}', (frame.shape[1]-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Model: {detector.current_model}', (frame.shape[1]-200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Conf: {detector.confidence_threshold:.2f}',
                   (frame.shape[1]-120, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show recording indicator
        if detector.recording:
            cv2.circle(frame, (30, frame.shape[0]-30), 10, (0, 0, 255), -1)
            detector.video_writer.write(frame)
        
        # Show help overlay
        if detector.show_help:
            help_text = [
                "Controls:",
                "q - Quit",
                "s - Screenshot",
                "r - Toggle recording",
                "f - Toggle FPS",
                "t - Toggle tracking",
                "m - Switch model",
                "h - Toggle help",
                "z - Draw zone",
                "c - Clear zones",
                "+/- - Adjust confidence"
            ]
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (frame.shape[1]-200, 150+i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Enhanced Object Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(detector.output_dir,
                                  f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('r'):
            detector.toggle_recording(frame)
        elif key == ord('f'):
            detector.show_fps = not detector.show_fps
        elif key == ord('t'):
            detector.tracking_enabled = not detector.tracking_enabled
            if not detector.tracking_enabled:
                detector.tracking_box = None
        elif key == ord('m'):
            model = detector.switch_model()
            print(f"Switched to model: {detector.current_model}")
        elif key == ord('h'):
            detector.show_help = not detector.show_help
        elif key == ord('z'):
            if not detector.drawing_zone:
                detector.drawing_zone = True
                detector.temp_zone = []
            else:
                if len(detector.temp_zone) >= 3:
                    detector.detection_zones.append(detector.temp_zone)
                detector.drawing_zone = False
                detector.temp_zone = []
        elif key == ord('c'):
            detector.detection_zones = []
            print("Cleared all detection zones")
        elif key == ord('+'):
            detector.confidence_threshold = min(1.0, detector.confidence_threshold + 0.1)
        elif key == ord('-'):
            detector.confidence_threshold = max(0.1, detector.confidence_threshold - 0.1)
    
    # Cleanup
    detector.save_detection_history()
    if detector.recording:
        detector.video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 