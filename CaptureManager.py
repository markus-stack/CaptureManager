from tcp_stream import TcpStream
from ultralytics import YOLO
import torch
import cv2
import threading
import json
import os
import numpy as np
import sys
# Reconfigure stdout to flush immediately
class FlushingStream:
    def write(self, msg):
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
    def flush(self):
        pass
sys.stdout = FlushingStream()

try:
    from yolo_ros.msg import DetectionArray, Detection
except ImportError:
    DetectionArray = None
    Detection = None
    print("[WARN] Could not import DetectionArray or Detection from yolo_ros.msg. ROS message publishing will be disabled.")

# Global event to signal all threads to stop
stop_event = threading.Event()

class CaptureManager:
    # Initialize the CaptureManager.
    # Loads configuration filename and optional ROS node, sets up thread management, sources, processing mode, output options, YOLO model config, and logging.
    def __init__(self, config_filename, node=None):
        self.config_filename = config_filename
        self.node = node
        self.threads = []
        self.stop_event = stop_event
        self.sources = []
        self.processing = 'local'
        self.output = []
        self.model_cfg = {}
        self.yolo_infer = None
        self.yolo_raw_model = None
        self.log_enabled = False
        self.log_file = None

    # Print a message to stdout and, if logging is enabled, also append it to the log file.
    # Used for all status, error, and info messages throughout the class.
    def log_print(self, *args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        print(msg, **kwargs)
        if self.log_enabled and self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')
            except Exception:
                pass

    # Create and return a thread function for capturing frames from a webcam device.
    # Captures frames, applies resolution settings, and processes each frame using process_and_output.
    def start_webcam_stream(self, device, processing, output, node=None, yolo_infer=None, yolo_raw_model=None):
        def webcam_thread(resolution=None):
            cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
            self.log_print(f"Starting webcam stream on device {device}")
            # Set resolution if provided
            if resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.log_print(f"Webcam device {device} not available.")
                    break
                self.process_and_output(frame, processing, output, node, yolo_infer, yolo_raw_model)
            cap.release()
            cv2.destroyAllWindows()
        return webcam_thread

    # Create and return a thread function for simulating a Raspberry Pi camera stream.
    # Generates synthetic images and processes each frame using process_and_output.
    def start_rpi_stream(self, device, processing, output, node=None, yolo_infer=None, yolo_raw_model=None):
        def rpi_thread(resolution=None):
            self.log_print(f"Starting dummy Raspberry Pi camera stream on device {device}")
            # Dummy: generate synthetic image
            w, h = 640, 480
            if resolution:
                w, h = resolution
            while not self.stop_event.is_set():
                img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.circle(img, (w//2, h//2), min(w,h)//8, (255,255,255), -1)
                self.process_and_output(img, processing, output, node, yolo_infer, yolo_raw_model)
                cv2.waitKey(100)
        return rpi_thread

    # Create and return a thread function for receiving frames from a TCP stream.
    # Handles connection, frame reception, resizing, and processing via process_and_output.
    def start_tcp_stream(self, host, port, processing, output, node=None, yolo_infer=None, yolo_raw_model=None):
        def tcp_thread(resolution=None):
            self.log_print(f"Starting TCP stream thread for host={host}, port={port}...")
            stream = TcpStream(host=host, port=port)
            try:
                while not self.stop_event.is_set():
                    frame = None
                    try:
                        frame = next(stream.listenToStream())
                    except StopIteration:
                        break
                    except Exception as e:
                        self.log_print(f"Stream for host={host}, port={port} lost or broken: {e}")
                        break
                    if frame is not None:
                        # Resize if requested
                        if resolution:
                            frame = cv2.resize(frame, resolution)
                        self.process_and_output(frame, processing, output, node, yolo_infer, yolo_raw_model)
            except Exception as e:
                self.log_print(f"Stream for host={host}, port={port} lost or broken: {e}")
        return tcp_thread

    # Start all configured input streams (webcam, rpi, tcp) as threads based on the sources in the config.
    # Each stream runs in its own thread and is appended to self.threads.
    def start_streams(self):
        for src in self.sources:
            typ = src.get('type')
            resolution = tuple(src.get('resolution', (640, 480)))
            if typ == 'web':
                thread_func = self.start_webcam_stream(src.get('device', 0), self.processing, self.output, self.node, self.yolo_infer, self.yolo_raw_model)
            elif typ == 'rpi':
                thread_func = self.start_rpi_stream(src.get('device', '/dev/video0'), self.processing, self.output, self.node, self.yolo_infer, self.yolo_raw_model)
            elif typ == 'tcp':
                thread_func = self.start_tcp_stream(src.get('host', '127.0.0.1'), src.get('port', 1756), self.processing, self.output, self.node, self.yolo_infer, self.yolo_raw_model)
            else:
                continue
            self.resolution = resolution
            thread = threading.Thread(target=thread_func, args=(resolution,), daemon=True)
            thread.start()
            self.threads.append(thread)

    # Signal all threads to stop and wait for them to finish. Cleans up OpenCV windows.
    # Used for graceful shutdown of all input streams.
    def close_streams(self):
        self.stop_event.set()
        # Wait for all threads to finish
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        cv2.destroyAllWindows()

    # Initialize and return the YOLO model and inference function for local processing.
    # Loads the model from the specified path, sets device (GPU/CPU), and handles errors.
    # Returns a callable for inference and the raw YOLO model object.
    def initialize_local_model(self, model_cfg):
        if not (model_cfg and model_cfg.get('path')):
            self.log_print("[ERROR] No YOLO model path specified in config. Exiting.")
            import sys
            sys.exit(1)
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_filename = os.path.basename(model_cfg['path'])
            model_path = os.path.join(models_dir, model_filename)
            if not os.path.exists(model_path):
                self.log_print(f"[ERROR] YOLO model file not found: {model_path}")
                self.log_print("[ERROR] Please ensure the model file exists at the specified path.")
                import sys
                sys.exit(1)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = device
            self.log_print(f"[YOLO] torch.cuda.is_available(): {torch.cuda.is_available()}")
            self.log_print(f"[YOLO] Initializing model on device: {device}")
            try:
                yolo_raw_model = YOLO(model_path, verbose=False)
                yolo_raw_model.to(device)
                self.log_print(f"[YOLO] Model loaded. Model device: {next(yolo_raw_model.model.parameters()).device}")
            except Exception as e:
                self.log_print(f"[ERROR] Failed to load YOLO model: {e}")
                import sys
                sys.exit(1)
            def yolo_infer(image):
                self.log_print(f"[YOLO] Inference device: {self.device}")
                results = yolo_raw_model.predict(
                    source=image,
                    verbose=False,
                    stream=False,
                    device=self.device,
                )
                dets = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        dets.append([x1, y1, x2, y2, conf, cls_id])
                return dets
            self.log_print(f"[YOLO] Model loaded from {model_path}")
            return yolo_infer, yolo_raw_model
        except Exception as e:
            self.log_print(f"[ERROR] Unexpected error during YOLO model initialization: {e}")
            import sys
            sys.exit(1)

    # Dummy initialization for an AI hat (edge device) model.
    # Returns a function that simply calls process_local for simulation purposes.
    def initialize_hat_model(self, model_cfg):
        def dummy_hat_infer(image):
            # Dummy: just call process_local without a model
            return self.process_local(image)
        self.log_print("Initialized dummy AI hat model (no real model loaded)")
        return dummy_hat_infer

    # Run YOLO object detection on the given image using the loaded model.
    # Returns a list of detected objects with class id, name, score, bounding box, mask, and keypoints.
    def process_local(self, image, yolo_model=None):
        # Use YOLO model for object detection if provided
        results = []
        if yolo_model is not None:
            yolo_results = self.yolo_raw_model.predict(
                source=image,
                verbose=False,
                stream=False,
                conf=self.yolo_threshold,
                iou=self.yolo_iou,
                imgsz=(self.yolo_imgsz_height, self.yolo_imgsz_width),
                half=self.yolo_half,
                max_det=self.yolo_max_det,
                augment=self.yolo_augment,
                agnostic_nms=self.yolo_agnostic_nms,
                retina_masks=self.yolo_retina_masks,
                device=getattr(self, 'device', self.device if hasattr(self, 'device') else 'cpu'),
            )
            yolo_results = yolo_results[0].cpu() if yolo_results else None
            if yolo_results is not None:
                hypothesis = []
                if hasattr(yolo_results, 'boxes') and yolo_results.boxes:
                    for box in yolo_results.boxes:
                        hypothesis.append({
                            'class_id': int(box.cls[0]),
                            'class_name': self.yolo_raw_model.model.names[int(box.cls[0])] if self.yolo_raw_model and hasattr(self.yolo_raw_model.model, 'names') else str(box.cls[0]),
                            'score': float(box.conf[0]),
                        })
                boxes = []
                if hasattr(yolo_results, 'boxes') and yolo_results.boxes:
                    for box in yolo_results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes.append([x1, y1, x2, y2])
                masks = []
                if hasattr(yolo_results, 'masks') and yolo_results.masks:
                    for mask in yolo_results.masks:
                        masks.append(mask)
                keypoints = []
                if hasattr(yolo_results, 'keypoints') and yolo_results.keypoints:
                    for kp in yolo_results.keypoints:
                        keypoints.append(kp)
                for i in range(len(hypothesis)):
                    obj = {
                        'class_id': hypothesis[i]['class_id'],
                        'class_name': hypothesis[i]['class_name'],
                        'score': hypothesis[i]['score'],
                        'bbox': boxes[i] if i < len(boxes) else None,
                        'mask': masks[i] if i < len(masks) else None,
                        'keypoints': keypoints[i] if i < len(keypoints) else None,
                    }
                    results.append(obj)
        return results

    # Dummy processing for AI hat mode. Simply calls process_local.
    def process_hat(self, image):
        # Dummy: just call process_local
        return self.process_local(image)

    # Process an image using the selected processing mode (local/hat), then output results.
    # Handles display (OpenCV window), console output (table), and ROS message publishing if enabled.
    def process_and_output(self, image, processing, output, node=None, yolo_infer=None, yolo_raw_model=None):
        image_for_yolo = image
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image_for_yolo = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        expected_h, expected_w = image.shape[:2]
        if hasattr(self, 'resolution'):
            expected_w, expected_h = self.resolution
        if image_for_yolo.shape[0] != expected_h or image_for_yolo.shape[1] != expected_w:
            image_for_yolo = cv2.resize(image_for_yolo, (expected_w, expected_h))

        if processing == 'local':
            results = self.process_local(image_for_yolo, yolo_infer)
        elif processing == 'hat':
            results = self.process_hat(image_for_yolo)
        else:
            results = []

        # For display/output, use grayscale if requested
        display_image = image
        if 'video' in output:
            cv2.imshow('Image Stream', display_image)
            cv2.waitKey(1)
        if 'console' in output:
            os.system('cls')
            if not results:
                print("No objects detected.")
            else:
                # Print header
                print(f"{'Idx':<4} {'Class':<7} {'Name':<15} {'Score':<7} {'BBox':<28} {'Mask':<10} {'Keypoints':<10}")
                print('-' * 90)
                for idx, obj in enumerate(results):
                    bbox = obj.get('bbox')
                    bbox_str = f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]" if bbox else "[----]"
                    mask_str = str(obj.get('mask'))[:8] if obj.get('mask') is not None else "-"
                    keypoints_str = str(obj.get('keypoints'))[:8] if obj.get('keypoints') is not None else "-"
                    print(f"{idx:<4} {obj.get('class_id', ''):<7} {str(obj.get('class_name', '')):<15} {obj.get('score', 0):<7.2f} {bbox_str:<28} {mask_str:<10} {keypoints_str:<10}")
        if 'rosmsg' in output and node is not None and DetectionArray is not None and Detection is not None:
            # Construct and publish DetectionArray message
            detections_msg = DetectionArray()
            for obj in results:
                aux_msg = Detection()
                aux_msg.class_id = obj.get('class_id')
                aux_msg.class_name = obj.get('class_name')
                aux_msg.score = obj.get('score')
                aux_msg.bbox = obj.get('bbox')
                aux_msg.mask = obj.get('mask')
                aux_msg.keypoints = obj.get('keypoints')
                detections_msg.detections.append(aux_msg)
            # Set header if available
            if hasattr(node, 'cv_bridge') and hasattr(node.cv_bridge, 'cv2_to_imgmsg'):
                detections_msg.header = getattr(node, 'header', None)
            node._pub.publish(detections_msg)

    # Main entry point for CaptureManager.
    # Loads configuration, sets up logging, YOLO parameters, sources, and starts all streams.
    # Handles graceful shutdown on KeyboardInterrupt.
    def run(self):
        config_path = os.path.join(os.path.dirname(__file__), self.config_filename)
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Logging activation from config
        self.log_enabled = bool(config.get('logging', False))
        if self.log_enabled:
            self.log_file = os.path.join(os.path.dirname(__file__), 'capture_manager.log')
            # Clear log file at start
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('')

            self.log_print("[INFO] Logging enabled. All print statements will be written to log file.")

        yolo_params = config.get('yolo_params', {})
        self.yolo_threshold = yolo_params.get('threshold', 0.25)
        self.yolo_iou = yolo_params.get('iou', 0.45)
        self.yolo_imgsz_height = yolo_params.get('imgsz_height', 640)
        self.yolo_imgsz_width = yolo_params.get('imgsz_width', 640)
        self.yolo_half = yolo_params.get('half', False)
        self.yolo_max_det = yolo_params.get('max_det', 100)
        self.yolo_augment = yolo_params.get('augment', False)
        self.yolo_agnostic_nms = yolo_params.get('agnostic_nms', False)
        self.yolo_retina_masks = yolo_params.get('retina_masks', False)

        self.sources = config.get('sources', [])
        self.processing = config.get('processing', 'local')
        self.output = config.get('output', [])
        self.model_cfg = config.get('model', {})

        if self.processing == 'local':
            self.yolo_infer, self.yolo_raw_model = self.initialize_local_model(self.model_cfg)
        elif self.processing == 'hat':
            self.yolo_infer, self.yolo_raw_model = self.initialize_hat_model(self.model_cfg), None
        self.start_streams()
        try:
            while True:
                cv2.waitKey(100)
        except KeyboardInterrupt:
            self.log_print("[INFO] KeyboardInterrupt received. Closing streams...")
            self.close_streams()
        finally:
            # Ensure all windows are closed even if an error occurs
            cv2.destroyAllWindows()