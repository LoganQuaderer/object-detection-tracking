import tkinter as tk
from tkinter import ttk
import cv2
import threading
from PIL import Image, ImageTk

class ControlPanel:
    def __init__(self, detector):
        self.root = tk.Tk()
        self.root.title("Object Detection Control Panel")
        self.detector = detector
        
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.stats_frame = ttk.Frame(self.root)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self._init_controls()
        self._init_stats()
        self._setup_update_thread()
    
    def _init_controls(self):
        # Model selection
        ttk.Label(self.control_frame, text="Model Selection").pack()
        self.model_var = tk.StringVar(value=self.detector.current_model)
        model_menu = ttk.OptionMenu(
            self.control_frame,
            self.model_var,
            self.detector.current_model,
            *self.detector.config.available_models,
            command=self._on_model_change
        )
        model_menu.pack(fill=tk.X, pady=5)
        
        # Confidence threshold slider
        ttk.Label(self.control_frame, text="Confidence Threshold").pack()
        self.conf_scale = ttk.Scale(
            self.control_frame,
            from_=0.0,
            to=1.0,
            value=self.detector.config.confidence_threshold,
            command=self._on_confidence_change
        )
        self.conf_scale.pack(fill=tk.X, pady=5)
        
        # Recording controls
        self.record_btn = ttk.Button(
            self.control_frame,
            text="Start Recording",
            command=self._toggle_recording
        )
        self.record_btn.pack(fill=tk.X, pady=5)
        
        # Zone controls
        self.zone_btn = ttk.Button(
            self.control_frame,
            text="Add Detection Zone",
            command=self._toggle_zone_drawing
        )
        self.zone_btn.pack(fill=tk.X, pady=5)
        
        # Clear zones button
        self.clear_zones_btn = ttk.Button(
            self.control_frame,
            text="Clear All Zones",
            command=self._clear_zones
        )
        self.clear_zones_btn.pack(fill=tk.X, pady=5)
    
    def _init_stats(self):
        # Detection statistics
        self.stats_text = tk.Text(self.stats_frame, height=20, width=40)
        self.stats_text.pack(padx=5, pady=5)
    
    def _setup_update_thread(self):
        self.update_thread = threading.Thread(target=self._update_stats, daemon=True)
        self.update_thread.start()
    
    def _update_stats(self):
        while True:
            stats = self._get_current_stats()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats)
            self.root.after(1000)  # Update every second
    
    def _get_current_stats(self):
        stats = "Detection Statistics:\n\n"
        for class_name, detections in self.detector.history.history.items():
            avg_conf = self.detector.history.get_average_confidence(class_name)
            stats += f"{class_name}:\n"
            stats += f"  Count: {len(detections)}\n"
            stats += f"  Avg Confidence: {avg_conf:.2%}\n\n"
        return stats
    
    def _on_model_change(self, selection):
        self.detector.switch_model()
    
    def _on_confidence_change(self, value):
        self.detector.config.confidence_threshold = float(value)
    
    def _toggle_recording(self):
        if not self.detector.video_writer.recording:
            self.detector.video_writer.start_recording(self.detector.current_frame)
            self.record_btn.config(text="Stop Recording")
        else:
            self.detector.video_writer.stop_recording()
            self.record_btn.config(text="Start Recording")
    
    def _toggle_zone_drawing(self):
        self.detector.drawing_zone = not self.detector.drawing_zone
        self.zone_btn.config(
            text="Finish Zone" if self.detector.drawing_zone else "Add Detection Zone"
        )
    
    def _clear_zones(self):
        self.detector.detection_zones.clear()
        self.detector.temp_zone.clear()
        self.detector.drawing_zone = False
        self.zone_btn.config(text="Add Detection Zone")
    
    def run(self):
        while True:
            frame = self.detector.get_frame()
            if frame is None:
                print("Error: Could not get frame from camera")
                break
            
            # Process frame and get detections
            detections = self.detector.process_frame(frame)
            
            # Draw detections
            self.detector.draw_detections(frame, detections)
            
            # Display frame
            cv2.imshow('Object Detection Control Panel', frame)
            
            # Handle keyboard input (q to quit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.detector.release()
        cv2.destroyAllWindows()