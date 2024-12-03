
# src/main.py
import argparse
from pathlib import Path
import yaml
from src.core.detector import ObjectDetector
from src.ui.control_panel import ControlPanel
from src.core.detector_config import DetectorConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection System')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--source', type=str, default='0',
                       help='Source (0 for webcam, or video/image path)')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    yaml_config = load_config(args.config)
    
    # Convert YAML config to DetectorConfig
    detector_config = DetectorConfig(
        confidence_threshold=yaml_config['detection']['confidence_threshold'],
        history_length=yaml_config['tracking']['history_length'],
        output_dir=yaml_config['output']['output_dir'],
        available_models=yaml_config['detection']['available_models']
    )
    
    try:
        detector = ObjectDetector(detector_config)
        control_panel = ControlPanel(detector)
        control_panel.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    main()