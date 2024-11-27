# Advanced Object Detection and Tracking

A real-time object detection and tracking application using YOLOv8 and OpenCV. This application provides advanced features for object detection, tracking, and zone-based monitoring.

## Features

- Real-time object detection using YOLOv8
- Object tracking with CSRT/KCF tracker
- Multiple model support (nano, small, medium)
- Custom detection zones
- Recording and screenshot capabilities
- Detection history tracking
- FPS monitoring
- Confidence threshold adjustment
- Interactive help overlay

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-tracking.git
cd object-detection-tracking
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python face_detection.py
```

### Controls

- 'q' - Quit
- 's' - Take screenshot
- 'r' - Toggle recording
- 'f' - Toggle FPS display
- 't' - Toggle object tracking
- 'm' - Switch model (nano->small->medium)
- 'h' - Toggle help overlay
- 'z' - Start/stop drawing detection zone
- 'c' - Clear all detection zones
- '+/-' - Adjust confidence threshold

### Mouse Controls

- Click on an object to start tracking it
- When drawing zones (after pressing 'z'), click to add points

## Output

- Screenshots and recordings are saved in the `detection_output` directory
- Detection history is automatically saved and loaded between sessions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.