import argparse  # Library for command-line option parsing
from datetime import datetime  # Library to handle date and time objects
import logging  # Library for logging system
import os  # Library for interacting with the operating system
import threading  # Library for concurrent threads
import time  # Library to handle time-related functions
import cv2  # OpenCV for computer vision
import torch  # PyTorch for machine learning model operations
from huggingface_hub import snapshot_download  # To download models from Hugging Face
from PIL import Image  # Pillow library for image manipulation
from transformers import AutoProcessor, AutoModelForCausalLM  # HF Transformers: model + processor
from transformers import BitsAndBytesConfig  # HF Transformers quantization config
import sys  # System-specific parameters and functions
import hid  # Library for accessing HID devices
import msvcrt  # Windows-specific console keyboard reading

def setup_logging(log_to_file, log_file_path="alerts.log"):
    """
    Sets up the logging configuration for the entire application.
    If log_to_file is True, messages will also be written to a specified file.

    :param log_to_file: Boolean indicating whether to also log to a file.
    :param log_file_path: The path where the log file will be written.
    """
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)

class PTZTracker:
    """
    Autonomous PTZ tracking class. Keeps a specified object centered and at a desired size.

    This class adjusts camera pan, tilt, and zoom automatically to keep
    the detected object within a certain bounding box ratio, or "zoom level."
    It smooths the bounding box width and height to reduce jitter and only
    sends commands at certain intervals to prevent overwhelming the PTZ camera.
    """

    def __init__(self, camera,
                 desired_ratio=0.20,
                 zoom_tolerance=0.4,
                 pan_tilt_tolerance=25,
                 pan_tilt_interval=0.75,
                 zoom_interval=0.5,
                 smoothing_factor=0.2,
                 max_consecutive_errors=5):
        """
        Initializes the PTZTracker with various parameters controlling behavior.

        :param camera: A camera object that supports PTZ commands.
        :param desired_ratio: Desired fraction of the frame the object should occupy.
        :param zoom_tolerance: The tolerance around the desired_ratio before zooming in/out.
        :param pan_tilt_tolerance: Pixel difference from center before panning/tilting.
        :param pan_tilt_interval: Minimum time (in seconds) between pan/tilt commands.
        :param zoom_interval: Minimum time (in seconds) between zoom commands.
        :param smoothing_factor: Weight for exponential smoothing of bounding box size.
        :param max_consecutive_errors: Maximum camera command errors before deactivation.
        """
        if not (0 < smoothing_factor < 1):
            raise ValueError("smoothing_factor must be between 0 and 1.")
        if desired_ratio <= 0 or desired_ratio >= 1:
            raise ValueError("desired_ratio should be between 0 and 1.")
        if zoom_tolerance < 0:
            raise ValueError("zoom_tolerance must be >= 0.")
        if pan_tilt_tolerance < 0:
            raise ValueError("pan_tilt_tolerance must be >= 0.")
        if pan_tilt_interval <= 0 or zoom_interval <= 0:
            raise ValueError("Intervals must be positive.")
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1.")

        self.camera = camera
        self.desired_ratio = desired_ratio
        self.zoom_tolerance = zoom_tolerance
        self.pan_tilt_tolerance = pan_tilt_tolerance
        self.pan_tilt_interval = pan_tilt_interval
        self.zoom_interval = zoom_interval
        self.smoothing_factor = smoothing_factor
        self.max_consecutive_errors = max_consecutive_errors

        self.last_pan_tilt_adjust = 0.0
        self.last_zoom_adjust = 0.0
        self.smoothed_width = None
        self.smoothed_height = None
        self.active = False
        self.consecutive_errors = 0

    def activate(self, active=True):
        """
        Activate or deactivate PTZ tracking. When deactivated, tracking resets smoothing and error counters.

        :param active: Boolean indicating whether tracking should be active.
        """
        self.active = active
        if not active:
            self.smoothed_width = None
            self.smoothed_height = None
            self.consecutive_errors = 0

    def adjust_camera(self, bbox, frame_width, frame_height):
        """
        Adjusts camera pan, tilt, and zoom to keep the object bounding box centered and sized per desired_ratio.

        :param bbox: A tuple (x1, y1, x2, y2) representing the object bounding box coordinates.
        :param frame_width: The width of the current frame in pixels.
        :param frame_height: The height of the current frame in pixels.
        """
        if not self.active:
            return

        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            print("Invalid bbox coordinates; skipping camera adjustment.")
            return

        bbox_width = (x2 - x1)
        bbox_height = (y2 - y1)

        if self.smoothed_width is None:
            self.smoothed_width = bbox_width
            self.smoothed_height = bbox_height
        else:
            self.smoothed_width = (self.smoothing_factor * bbox_width
                                   + (1 - self.smoothing_factor) * self.smoothed_width)
            self.smoothed_height = (self.smoothing_factor * bbox_height
                                    + (1 - self.smoothing_factor) * self.smoothed_height)

        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        desired_width = frame_width * self.desired_ratio
        desired_height = frame_height * self.desired_ratio

        min_width = desired_width * (1 - self.zoom_tolerance)
        max_width = desired_width * (1 + self.zoom_tolerance)
        min_height = desired_height * (1 - self.zoom_tolerance)
        max_height = desired_height * (1 + self.zoom_tolerance)

        current_time = time.time()

        if (current_time - self.last_pan_tilt_adjust) >= self.pan_tilt_interval:
            dx = bbox_center_x - frame_center_x
            dy = bbox_center_y - frame_center_y

            pan_tilt_moved = False
            if abs(dx) > self.pan_tilt_tolerance:
                pan_tilt_moved = self._safe_camera_command('pan_left' if dx < 0 else 'pan_right') or pan_tilt_moved
            if abs(dy) > self.pan_tilt_tolerance:
                pan_tilt_moved = self._safe_camera_command('tilt_up' if dy < 0 else 'tilt_down') or pan_tilt_moved

            if pan_tilt_moved:
                self.last_pan_tilt_adjust = current_time

        if (current_time - self.last_zoom_adjust) >= self.zoom_interval:
            width_too_small = self.smoothed_width < min_width
            height_too_small = self.smoothed_height < min_height
            width_too_large = self.smoothed_width > max_width
            height_too_large = self.smoothed_height > max_height

            zoom_moved = False
            if width_too_small or height_too_small:
                zoom_moved = self._safe_camera_command('zoom_in')
            elif width_too_large or height_too_large:
                zoom_moved = self._safe_camera_command('zoom_out')

            if zoom_moved:
                self.last_zoom_adjust = current_time

        if self.consecutive_errors >= self.max_consecutive_errors:
            print("Too many consecutive camera errors, deactivating PTZ tracking.")
            self.activate(False)

    def _safe_camera_command(self, command):
        """
        Safely invokes a camera command, handling exceptions and counting errors.

        :param command: A string specifying the method name to call on the camera.
        :return: Boolean indicating whether the command executed successfully.
        """
        if not hasattr(self.camera, command):
            print(f"Camera does not support command '{command}'.")
            return False
        try:
            method = getattr(self.camera, command)
            method()
            self.consecutive_errors = 0
            return True
        except Exception as e:
            self.consecutive_errors += 1
            print(f"Error executing camera command '{command}': {e}")
            return False

class ModelManager:
    """
    Class responsible for loading and managing a Hugging Face Transformer model and processor,
    with optional quantization settings.
    """

    def __init__(self, device, quantization=None):
        """
        Initialize the ModelManager with a torch device and an optional quantization setting.

        :param device: Torch device, e.g., 'cuda' or 'cpu'.
        :param quantization: A string (e.g., "4bit") indicating which quantization scheme to apply.
        """
        self.device = device
        self.model = None
        self.processor = None
        self.quantization = quantization

    def load_local_model(self, model_path):
        """
        Loads a local model from the specified directory path. Optionally applies quantization.

        :param model_path: Filesystem path to the local pre-trained model directory.
        :return: Boolean indicating whether the model was successfully loaded.
        """
        if not os.path.exists(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} does not exist.")
            return False
        if not os.path.isdir(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} is not a directory.")
            return False

        try:
            logging.info(f"Attempting to load model from {os.path.abspath(model_path)}")
            quant_config = self._get_quant_config()

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quant_config,
            ).eval()

            if not self.quantization:
                self.model.to(self.device)
                if torch.cuda.is_available():
                    self.model = self.model.half()
                    logging.info("Using FP16 precision for the model.")

            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            logging.info(f"Model loaded successfully from {os.path.abspath(model_path)}")
            return True
        except (OSError, ValueError, ModuleNotFoundError) as e:
            logging.error(f"Error initializing model: {e}")
        except Exception as e:
            logging.error(f"Unexpected error initializing model: {e}")
        return False

    def download_and_load_model(self, repo_id="microsoft/Florence-2-base-ft"):
        """
        Downloads a model from the Hugging Face Hub using its repository ID, then loads it locally.

        :param repo_id: The Hugging Face model repository ID to download from.
        :return: Boolean indicating whether the model was successfully downloaded and loaded.
        """
        try:
            local_model_dir = "model"
            snapshot_download(repo_id=repo_id, local_dir=local_model_dir)
            if not os.path.exists(local_model_dir):
                logging.error(f"Model download failed, directory {os.path.abspath(local_model_dir)} does not exist.")
                return False
            if not os.path.isdir(local_model_dir):
                logging.error(f"Model download failed, path {os.path.abspath(local_model_dir)} is not a directory.")
                return False
            logging.info(f"Model downloaded and initialized at {os.path.abspath(local_model_dir)}")
            return self.load_local_model(local_model_dir)
        except OSError as e:
            logging.error(f"OS error during model download: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
        return False

    def _get_quant_config(self):
        if self.quantization == "4bit":
            logging.info("Using 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None

class RecordingManager:
    """
    Class that manages video recording. Can record continuously or by detection/inference triggers.
    """

    def __init__(self, record_mode=None):
        """
        Initializes the recording manager with a specified mode.

        :param record_mode: The mode for starting/stopping recording:
            None - no recording,
            "od" - based on object detections,
            "infy"/"infn" - based on inference results (yes/no).
        """
        self.record_mode = record_mode
        self.recording = False
        self.video_writer = None
        self.video_out_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        self.last_detection_time = time.time()

    def start_recording(self, frame):
        """
        Starts video recording given an initial frame (to set up dimensions, codec, etc.).

        :param frame: An OpenCV image frame used to determine recording dimensions and color format.
        """
        if not self.recording and self.record_mode:
            height, width, _ = frame.shape
            self.video_writer = cv2.VideoWriter(
                self.video_out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                (width, height),
            )
            self.recording = True
            logging.info(f"Started recording video: {self.video_out_path}")

    def stop_recording(self):
        """
        Stops video recording and releases the VideoWriter resource.
        """
        if self.recording:
            self.video_writer.release()
            self.recording = False
            logging.info(f"Stopped recording video: {self.video_out_path}")

    def write_frame(self, frame):
        """
        Writes a single frame to the open video file if currently recording.

        :param frame: The OpenCV image frame to be written to the video.
        """
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def handle_recording_by_detection(self, detections, frame):
        """
        Starts or stops recording based on whether object detections are present.

        :param detections: A list of detections, each of which is typically (bbox, label).
        :param frame: The current OpenCV image frame.
        """
        if not self.record_mode:
            return
        current_time = time.time()
        if detections:
            self.start_recording(frame)
            self.last_detection_time = current_time
        else:
            if (current_time - self.last_detection_time) > 1:
                self.stop_recording()
                logging.info("Recording stopped due to no detection for 1+ second.")

    def handle_recording_by_inference(self, inference_result, frame):
        """
        Starts or stops recording based on inference (yes/no) results.

        :param inference_result: A string, typically "yes" or "no" from some model inference.
        :param frame: The current OpenCV image frame.
        """
        if self.record_mode == "infy" and inference_result == "yes":
            self.start_recording(frame)
        elif self.record_mode == "infy" and inference_result == "no":
            self.stop_recording()
        elif self.record_mode == "infn" and inference_result == "no":
            self.start_recording(frame)
        elif self.record_mode == "infn" and inference_result == "yes":
            self.stop_recording()

class ImageUtils:
    """
    Utility class for image-related operations such as drawing bounding boxes and saving screenshots.
    """

    @staticmethod
    def plot_bbox(image, detections):
        """
        Draws bounding boxes and labels on an image using OpenCV.

        :param image: The OpenCV image (numpy array).
        :param detections: A list of (bbox, label) tuples, where bbox=(x1, y1, x2, y2).
        :return: The image with bounding boxes drawn.
        """
        try:
            for bbox, label in detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            return image
        except cv2.error as e:
            logging.error(f"OpenCV error plotting bounding boxes: {e}")
        except Exception as e:
            logging.error(f"Error plotting bounding boxes: {e}")
        return image

    @staticmethod
    def save_screenshot(frame):
        """
        Saves a screenshot of the current frame with a timestamped filename.

        :param frame: The OpenCV image (numpy array) to save.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
            print(f"[{timestamp}] Screenshot saved: {filename}")
        except cv2.error as e:
            logging.error(f"OpenCV error saving screenshot: {e}")
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error saving screenshot: {e}")
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error saving screenshot: {e}")

class AlertLogger:
    """
    A simple class to log alerts both to a dedicated file (alerts.log) and to the console.
    """

    @staticmethod
    def log_alert(message):
        """
        Appends an alert message to a log file with a timestamp, and also prints to console.

        :param message: The alert message to be logged.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open("alerts.log", "a") as log_file:
                log_file.write(f"{timestamp} - {message}\n")
            logging.info(f"{timestamp} - {message}")
            print(f"[{timestamp}] Log entry written: {message}")
        except IOError as e:
            logging.error(f"IO error logging alert: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] IO error logging alert: {e}")
        except Exception as e:
            logging.error(f"Error logging alert: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Error logging alert: {e}")

class PTZController:
    """
    Class to control PTZ camera movements via HID commands.
    It locates a suitable HID device for the PTZ camera based on given vendor and product IDs.
    """

    def __init__(self, vendor_id=0x046D, product_id=0x085F, usage_page=65280, usage=1):
        """
        Initializes the PTZController by attempting to open a HID device matching the given parameters.

        :param vendor_id: The USB vendor ID of the PTZ device.
        :param product_id: The USB product ID of the PTZ device.
        :param usage_page: The HID usage page number.
        :param usage: The HID usage number.
        """
        self.device = None
        try:
            ptz_path = None
            for d in hid.enumerate(vendor_id, product_id):
                if d['usage_page'] == usage_page and d['usage'] == usage:
                    ptz_path = d['path']
                    break
            if ptz_path:
                self.device = hid.device()
                self.device.open_path(ptz_path)
                print("PTZ HID interface opened successfully.")
            else:
                print("No suitable PTZ HID interface found. PTZ commands may not work.")
        except IOError as e:
            print(f"Error opening PTZ device: {e}")
        except Exception as e:
            print(f"Unexpected error during PTZ device initialization: {e}")

    def send_command(self, report_id, value):
        """
        Sends a command to the PTZ device via HID write.

        :param report_id: The report ID for the PTZ control.
        :param value: The value that represents the specific command (e.g., pan left/right, tilt up/down).
        """
        if not self.device:
            print("PTZ Device not initialized.")
            return
        command = [report_id & 0xFF, value] + [0x00]*30
        try:
            self.device.write(command)
            print(f"Command sent: report_id={report_id}, value={value}")
            time.sleep(0.2)
        except IOError as e:
            print(f"Error sending PTZ command: {e}")
        except Exception as e:
            print(f"Unexpected error sending PTZ command: {e}")

    def pan_right(self):
        """Pans the camera to the right."""
        self.send_command(0x0B, 0x02)

    def pan_left(self):
        """Pans the camera to the left."""
        self.send_command(0x0B, 0x03)

    def tilt_up(self):
        """Tilts the camera upward."""
        self.send_command(0x0B, 0x00)

    def tilt_down(self):
        """Tilts the camera downward."""
        self.send_command(0x0B, 0x01)

    def zoom_in(self):
        """Zooms the camera in."""
        self.send_command(0x0B, 0x04)

    def zoom_out(self):
        """Zooms the camera out."""
        self.send_command(0x0B, 0x05)

    def close(self):
        """
        Closes the HID device handle, if open, to release system resources.
        """
        if self.device:
            try:
                self.device.close()
                print("PTZ device closed successfully.")
            except Exception as e:
                print(f"Error closing PTZ device: {e}")

class YOFLO:
    """
    Main class to run object detection and/or expression comprehension using a loaded model.
    Handles webcam or RTSP streams, optional PTZ tracking, screenshot capturing, logging, and more.
    """

    def __init__(
        self,
        model_path=None,
        display_inference_rate=False,
        pretty_print=False,
        inference_limit=None,
        class_names=None,
        webcam_indices=None,
        rtsp_urls=None,
        record=None,
        quantization=None,
        ptz_tracker=None,
        track_object_name=None
    ):
        """
        Initializes the YOFLO system with various configuration options.

        :param model_path: Local filesystem path to the pre-trained model directory, if not downloading.
        :param display_inference_rate: If True, logs the inference rate (inferences per second).
        :param pretty_print: If True, logs detections in a 'pretty' format with boundaries.
        :param inference_limit: Numeric limit on how many inferences can be made per second.
        :param class_names: Optional list of class names for filtering object detections.
        :param webcam_indices: List of integer indices for local webcams to open.
        :param rtsp_urls: List of RTSP URLs for network stream sources.
        :param record: Recording mode ("od", "infy", "infn", or None).
        :param quantization: Quantization mode ("4bit" or None).
        :param ptz_tracker: Optional PTZTracker object for autonomous camera movement.
        :param track_object_name: The name of the object class to track using PTZ, if ptz_tracker is active.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_start_time = None
        self.inference_count = 0
        self.class_names = class_names if class_names else []
        self.phrase = None
        self.object_detection_active = False
        self.screenshot_active = False
        self.log_to_file_active = False
        self.headless = True

        self.display_inference_rate = display_inference_rate
        self.stop_webcam_flag = threading.Event()
        self.webcam_threads = []
        self.pretty_print = pretty_print
        self.inference_limit = inference_limit
        self.last_inference_time = 0
        self.inference_phrases = []
        self.webcam_indices = webcam_indices if webcam_indices else [0]
        self.rtsp_urls = rtsp_urls if rtsp_urls else []
        self.quantization = quantization
        self.record = record

        self.track_object_name = track_object_name
        self.recording_manager = RecordingManager(record)
        self.model_manager = ModelManager(self.device, self.quantization)
        self.ptz_tracker = ptz_tracker

        if model_path:
            self.model_manager.load_local_model(model_path)

    @property
    def model(self):
        return self.model_manager.model

    @property
    def processor(self):
        return self.model_manager.processor

    def update_inference_rate(self):
        try:
            if self.inference_start_time is None:
                self.inference_start_time = time.time()
            else:
                elapsed_time = time.time() - self.inference_start_time
                if elapsed_time > 0:
                    inferences_per_second = self.inference_count / elapsed_time
                    if self.display_inference_rate:
                        logging.info(f"IPS: {inferences_per_second:.2f}")
        except Exception as e:
            logging.error(f"Error updating inference rate: {e}")

    def run_object_detection(self, image):
        try:
            task_prompt = "<OD>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            dtype = next(self.model.parameters()).dtype
            inputs = {
                k: v.to(self.device, dtype=dtype) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.device),
                    pixel_values=inputs.get("pixel_values").to(self.device),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task=task_prompt, image_size=image.size
                )
            return parsed_answer
        except (torch.cuda.CudaError, ModuleNotFoundError) as e:
            logging.error(f"CUDA error during object detection: {e}")
        except Exception as e:
            logging.error(f"Error during object detection: {e}")
        return None

    def run_expression_comprehension(self, image, phrase):
        try:
            task_prompt = "<CAPTION_TO_EXPRESSION_COMPREHENSION>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            inputs["input_ids"] = self.processor.tokenizer(phrase, return_tensors="pt").input_ids

            dtype = next(self.model.parameters()).dtype
            inputs = {
                k: v.to(self.device, dtype=dtype) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.device),
                    pixel_values=inputs.get("pixel_values").to(self.device),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            return generated_text
        except (torch.cuda.CudaError, ModuleNotFoundError) as e:
            logging.error(f"CUDA error during expression comprehension: {e}")
        except Exception as e:
            logging.error(f"Error during expression comprehension: {e}")
        return None

    def filter_detections(self, detections):
        try:
            if not self.class_names:
                return detections
            filtered_detections = [
                (bbox, label)
                for bbox, label in detections
                if label.lower() in [name.lower() for name in self.class_names]
            ]
            return filtered_detections
        except Exception as e:
            logging.error(f"Error filtering detections: {e}")
        return detections

    def pretty_print_detections(self, detections):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info("\n" + "=" * 50)
            for bbox, label in detections:
                bbox_str = f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
                logging.info(f"- {label}: {bbox_str} at {timestamp}")
            logging.info("=" * 50 + "\n")
        except Exception as e:
            logging.error(f"Error in pretty_print_detections: {e}")

    def pretty_print_expression(self, clean_result):
        """
        Prints expression comprehension results in a nicely formatted block
        rather than just raw text. This helps if the model outputs tokens like
        '</s><s>yes</s>' which we can clean up or highlight.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tidy_result = clean_result.replace("</s>", "").replace("<s>", "").strip()

            logging.info("\n" + "=" * 50)
            logging.info(f"Expression Comprehension: {tidy_result} at {timestamp}")
            logging.info("=" * 50 + "\n")
        except Exception as e:
            logging.error(f"Error in pretty_print_expression: {e}")

    def evaluate_inference_chain(self, image):
        try:
            if not self.inference_phrases:
                logging.error("No inference phrases set.")
                return "FAIL", []
            results = []
            for phrase in self.inference_phrases:
                result = self.run_expression_comprehension(image, phrase)
                if result:
                    results.append("yes" in result.lower())
            overall_result = "PASS" if results.count(True) >= 2 else "FAIL"
            return overall_result, results
        except Exception as e:
            logging.error(f"Error evaluating inference chain: {e}")
            return "FAIL", []

    def set_inference_phrases(self, phrases):
        self.inference_phrases = phrases
        logging.info(f"Inference phrases set: {self.inference_phrases}")

    def download_model(self):
        return self.model_manager.download_and_load_model()

    def start_webcam_detection(self):
        try:
            if self.webcam_threads:
                logging.warning("Webcam detection is already running.")
                return
            self.stop_webcam_flag.clear()

            sources = self.rtsp_urls if self.rtsp_urls else self.webcam_indices
            for source in sources:
                thread = threading.Thread(
                    target=self._webcam_detection_thread, args=(source,)
                )
                thread.start()
                self.webcam_threads.append(thread)
        except Exception as e:
            logging.error(f"Error starting webcam detection: {e}")

    def stop_webcam_detection(self):
        try:
            self.object_detection_active = False
            self.stop_webcam_flag.set()
            for thread in self.webcam_threads:
                thread.join()
            self.webcam_threads = []
            logging.info("Webcam detection stopped")
            if self.recording_manager.recording:
                self.recording_manager.stop_recording()
        except Exception as e:
            logging.error(f"Error stopping webcam detection: {e}")

    def _webcam_detection_thread(self, source):
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logging.error(f"Could not open video source {source}.")
                return
            window_name = f"Object Detection Source {source}"

            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.error(f"Failed to capture image from source {source}.")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Failed to capture image from source {source}.")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image)
                current_time = time.time()

                if self.inference_limit:
                    time_since_last_inference = current_time - self.last_inference_time
                    if time_since_last_inference < 1 / self.inference_limit:
                        time.sleep(1 / self.inference_limit - time_since_last_inference)
                    current_time = time.time()

                self._process_frame(frame, image_pil, source)

                if not self.headless:
                    if self.recording_manager.recording:
                        self.recording_manager.write_frame(frame)
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.last_inference_time = current_time

            cap.release()
            if not self.headless:
                cv2.destroyWindow(window_name)
            if self.recording_manager.recording:
                self.recording_manager.stop_recording()

        except cv2.error as e:
            logging.error(f"OpenCV error in detection thread {source}: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] OpenCV error in detection thread {source}: {e}")
        except ModuleNotFoundError as e:
            logging.error(f"ModuleNotFoundError in detection thread {source}: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ModuleNotFoundError in detection thread {source}: {e}")
        except Exception as e:
            logging.error(f"Error in detection thread {source}: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in detection thread {source}: {e}")

    def _pick_tracked_object(self, detections):
        if not self.track_object_name:
            return None
        candidate_detections = [(bbox, label) for bbox, label in detections
                                if label.lower() == self.track_object_name.lower()]
        if not candidate_detections:
            return None
        def bbox_area(bbox):
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        largest_bbox = max(candidate_detections, key=lambda x: bbox_area(x[0]))[0]
        return largest_bbox

    def _process_frame(self, frame, image_pil, source):
        primary_bbox = None

        if self.object_detection_active:
            results = self.run_object_detection(image_pil)
            if results and "<OD>" in results:
                detections = [
                    (bbox, label)
                    for bbox, label in zip(
                        results["<OD>"]["bboxes"], results["<OD>"]["labels"]
                    )
                ]
                filtered_detections = self.filter_detections(detections)
                if self.pretty_print:
                    self.pretty_print_detections(filtered_detections)
                else:
                    logging.info(f"Detections from source {source}: {filtered_detections}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detections from source {source}: {filtered_detections}")

                if not self.headless:
                    frame = ImageUtils.plot_bbox(frame, filtered_detections)
                self.inference_count += 1
                self.update_inference_rate()

                if filtered_detections:
                    if self.screenshot_active:
                        ImageUtils.save_screenshot(frame)
                    if self.log_to_file_active:
                        AlertLogger.log_alert(f"Detections from source {source}: {filtered_detections}")

                self.recording_manager.handle_recording_by_detection(filtered_detections, frame)
                if self.ptz_tracker and self.ptz_tracker.active:
                    primary_bbox = self._pick_tracked_object(filtered_detections)

        if self.phrase:
            results = self.run_expression_comprehension(image_pil, self.phrase)
            if results:
                clean_result = results.lower()

                if self.pretty_print:
                    self.pretty_print_expression(clean_result)
                else:
                    logging.info(f"Single phrase inference from source {source}: {clean_result}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Single-phrase result: {clean_result}")
                self.inference_count += 1
                self.update_inference_rate()

                if "yes" in clean_result:
                    if self.log_to_file_active:
                        AlertLogger.log_alert(f"Expression from source {source}: yes")
                    if self.record:
                        self.recording_manager.handle_recording_by_inference("yes", frame)
                elif "no" in clean_result:
                    if self.log_to_file_active:
                        AlertLogger.log_alert(f"Expression from source {source}: no")
                    if self.record:
                        self.recording_manager.handle_recording_by_inference("no", frame)

        if self.inference_phrases:
            inference_result, phrase_results = self.evaluate_inference_chain(image_pil)
            logging.info(f"Inference Chain result from source {source}: {inference_result}, Details: {phrase_results}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inference Chain result from source {source}: {inference_result}, Details: {phrase_results}")

            if self.pretty_print:
                for idx, result in enumerate(phrase_results):
                    logging.info(f"Inference {idx + 1} from source {source}: {'PASS' if result else 'FAIL'}")
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inference {idx + 1} from source {source}: {'PASS' if result else 'FAIL'}")
            self.inference_count += 1
            self.update_inference_rate()

        if self.ptz_tracker and self.ptz_tracker.active and primary_bbox:
            frame_height, frame_width, _ = frame.shape
            self.ptz_tracker.adjust_camera(primary_bbox, frame_width, frame_height)

def ptz_control_thread(ptz_camera):
    """
    A simple thread function for interactive PTZ control using arrow keys and +/- zoom on Windows.

    :param ptz_camera: A PTZController instance to control.
    """
    print("PTZ control started. Use arrow keys to pan/tilt, +/- to zoom, q to quit.")
    while True:
        ch = msvcrt.getch()
        if ch == b'\xe0':
            arrow = msvcrt.getch()
            if arrow == b'H':
                ptz_camera.tilt_up()
            elif arrow == b'P':
                ptz_camera.tilt_down()
            elif arrow == b'K':
                ptz_camera.pan_left()
            elif arrow == b'M':
                ptz_camera.pan_right()
        elif ch == b'+':
            ptz_camera.zoom_in()
        elif ch == b'-':
            ptz_camera.zoom_out()
        elif ch == b'q':
            print("Quitting PTZ control.")
            break
    ptz_camera.close()

def main():
    """
    Main function to parse command-line arguments, configure and run the YO-FLO system,
    including optional model download, PTZ camera setup, and webcam detection loops.
    """
    parser = argparse.ArgumentParser(
        description="YO-FLO: A proof-of-concept vision-language model as a YOLO alternative."
    )
    parser.add_argument("-od", nargs="*", help='Enable object detection with optional class names.')
    parser.add_argument("-ph", type=str, help="Yes/No question for expression comprehension.")
    parser.add_argument("-hl", action="store_true", help="Run in headless mode (no video display).")
    parser.add_argument("-ss", action="store_true", help="Enable screenshot on detection.")
    parser.add_argument("-lf", action="store_true", help="Enable logging alerts to file.")
    parser.add_argument("-ir", action="store_true", help="Display inference rate.")
    parser.add_argument("-pp", action="store_true", help="Enable pretty print for detections.")
    parser.add_argument("-il", type=float, help="Limit the inference rate (inferences per second).")
    parser.add_argument("-ic", nargs="+", help="Enable inference chain with specified phrases.")
    parser.add_argument("-wi", nargs="+", type=int, help="Indices of webcams to use.")
    parser.add_argument("-rtsp", nargs="+", type=str, help="RTSP URLs for video streams.")
    parser.add_argument("-r", choices=["od", "infy", "infn"], help="Video recording mode.")
    parser.add_argument("-4bit", action="store_true", help="Enable 4-bit quantization.")
    parser.add_argument("-ptz", nargs='?', const='0',
                        help="Enable PTZ control. If 'track' is supplied, autonomous tracking is enabled.")
    parser.add_argument("-to", "--track-object", type=str,
                        help="Specify the object class name to track when PTZ tracking is active.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-mp", type=str, help="Path to the local pre-trained model directory.")
    group.add_argument("-dm", action="store_true", help="Download the model from Hugging Face.")

    args = parser.parse_args()
    quantization_mode = "4bit" if getattr(args, '4bit', False) else None

    try:
        setup_logging(args.lf)
        webcam_indices = args.wi if args.wi else [0]
        rtsp_urls = args.rtsp if args.rtsp else []

        if args.dm:
            yo_flo = YOFLO(
                display_inference_rate=args.ir,
                pretty_print=args.pp,
                inference_limit=args.il,
                class_names=args.od,
                webcam_indices=webcam_indices,
                rtsp_urls=rtsp_urls,
                record=args.r,
                quantization=quantization_mode,
                track_object_name=args.track_object
            )
            if not yo_flo.download_model():
                return
        else:
            if not os.path.exists(args.mp):
                logging.error(f"Model path {args.mp} does not exist.")
                return
            if not os.path.isdir(args.mp):
                logging.error(f"Model path {args.mp} is not a directory.")
                return
            yo_flo = YOFLO(
                model_path=args.mp,
                display_inference_rate=args.ir,
                pretty_print=args.pp,
                inference_limit=args.il,
                class_names=args.od,
                webcam_indices=webcam_indices,
                rtsp_urls=rtsp_urls,
                record=args.r,
                quantization=quantization_mode,
                track_object_name=args.track_object
            )

        if args.ph:
            yo_flo.phrase = args.ph
        if args.ic:
            yo_flo.set_inference_phrases(args.ic)

        yo_flo.headless = args.hl
        yo_flo.object_detection_active = args.od is not None
        yo_flo.screenshot_active = args.ss
        yo_flo.log_to_file_active = args.lf

        yo_flo.start_webcam_detection()

        ptz_thread = None
        ptz_camera = None
        if args.ptz is not None:
            if args.ptz.lower() == 'track':
                ptz_camera = PTZController()
                ptz_tracker = PTZTracker(ptz_camera)
                ptz_tracker.activate(True)
                yo_flo.ptz_tracker = ptz_tracker
            else:
                try:
                    ptz_index = int(args.ptz)
                except ValueError:
                    ptz_index = 0
                print(f"Initializing PTZ control for camera index: {ptz_index}")
                ptz_camera = PTZController()
                ptz_thread = threading.Thread(target=ptz_control_thread, args=(ptz_camera,))
                ptz_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            yo_flo.stop_webcam_detection()
            if ptz_thread and ptz_thread.is_alive():
                print("Press 'q' to quit PTZ mode if still active.")
        finally:
            if ptz_thread and ptz_thread.is_alive():
                ptz_thread.join()

    except Exception as e:
        logging.error(f"An error occurred during main loop: {e}")
    else:
        input("Press Enter to stop...")
        yo_flo.stop_webcam_detection()

if __name__ == "__main__":
    main()
