# CaptureManager: Multi-Source Object Detection

## Overview

`CaptureManager` is a Python class for robust, configurable multi-source image/video streaming and object detection using YOLO (Ultralytics) or the RPi AI Hat. It supports webcam, Raspberry Pi camera, and TCP streams, with flexible output options and logging. All configuration is handled via a single JSON file.

## Features
- **Multi-source input:**
  - Webcam (local camera)
  - Raspberry Pi camera (simulated)
  - TCP stream (remote image/video)
- **Image processing modes:**
  - Local (CPU/GPU, using YOLO)
  - AI hat (edge device, simulated)
- **Output options:**
  - Video display (`cv2.imshow`)
  - Console (tabular output)
  - ROS message publishing (DetectionArray, Detection)
- **Logging:**
  - All print statements can be logged to `tcp_stream_ros.log` if enabled in config
- **Configurable via JSON:**
  - Sources, processing mode, output, YOLO parameters, logging, and model path
- **Model selection:**
  - YOLO model path specified in JSON; loaded from `models/` directory

## Quick Start

1. **Install dependencies:**
   - Python 3.8+
   - `ultralytics`, `opencv-python`, `torch`, `numpy`, (optional: ROS Python packages)

2. **Prepare your YOLO model:**
   - Place your YOLO model file (e.g., `yolov8n.pt`) in the `models/` directory next to `tcp_stream_ros.py`.
   - Specify the model path in your config JSON (see below).

3. **Configure your sources and options:**
   - Edit `stream_config.json` to select input sources, processing mode, output, logging, and YOLO parameters.

4. **Run the script:**
   ```bash
   python tcp_stream_ros.py
   ```
   This will start all configured streams and begin detection/output as specified.

## Example: `stream_config.json`
```json
{
  "sources": [
    { "type": "web", "device": 0, "id": "webcam" },
    { "type": "tcp", "host": "127.0.0.1", "port": 1756, "id": "remote" }
  ],
  "processing": "local", // or "hat"
  "model": { "path": "models/yolov8n.pt" },
  "output": ["video", "console", "rosmsg"],
  "logging": true,
  "yolo_params": {
    "threshold": 0.25,
    "iou": 0.45,
    "imgsz_height": 640,
    "imgsz_width": 640,
    "half": false,
    "max_det": 100,
    "augment": false,
    "agnostic_nms": false,
    "retina_masks": false
  }
}
```

## How to Use the Class

### Starting the Manager
You can start the manager as shown in `main()`:
```python
from tcp_stream_ros import CaptureManager
manager = CaptureManager('stream_config.json')
manager.run()
```
This will launch all threads and handle all processing, output, and shutdown internally.

### Selecting Sources
- **Webcam:**
  - `{ "type": "web", "device": 0 }` (device index)
- **Raspberry Pi camera:**
  - `{ "type": "rpi", "device": "/dev/video0" }` (simulated)
- **TCP stream:**
  - `{ "type": "tcp", "host": "IP", "port": PORT }`

### Selecting Processing Mode
- **Local:**
  - Uses YOLO on CPU or GPU (auto-detected)
  - Set `"processing": "local"` in config
- **AI hat:**
  - Simulated edge device processing
  - Set `"processing": "hat"` in config

### Output Options
- **Video:**
  - `"video"` (shows image stream in OpenCV window)
- **Console:**
  - `"console"` (prints detection table to terminal)
- **ROS Message:**
  - `"rosmsg"` (publishes DetectionArray/Detection messages if ROS is available)

### Logging
- Enable logging by setting `"logging": true` in config
- All print statements are duplicated to `tcp_stream_ros.log`

### Configuring YOLO Parameters
- All YOLO detection parameters (threshold, iou, image size, etc.) are set in the `yolo_params` section of the config

### Model Specification
- Place your YOLO model file in the `models/` directory
- Specify the path in the config: `"model": { "path": "models/yolov8n.pt" }`

## Advanced Usage
- You can combine multiple sources and outputs as needed
- All runtime logic and event handling is managed inside the `CaptureManager` class
- For ROS integration, ensure the required message types are available

## Troubleshooting
- If the YOLO model file is missing or cannot be loaded, a clear error will be printed and the program will exit
- If ROS messages are not available, console and video output will still work
- Logging can be disabled by setting `"logging": false`

## License
MIT


