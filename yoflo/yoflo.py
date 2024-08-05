import argparse
from datetime import datetime
import logging
import os
import threading
import time
import cv2
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def setup_logging(log_to_file, log_file_path="alerts.log"):
    """Set up logging to file and/or console.

    Args:
        log_to_file (bool): Whether to log to a file.
        log_file_path (str, optional): Path to the log file. Defaults to "alerts.log".
    """
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)

class YOFLO:
    def __init__(
        self,
        model_path=None,
        display_inference_speed=False,
        pretty_print=False,
        inference_limit=None,
        class_names=None,
        webcam_indices=None,
        rtsp_urls=None,
        record=None,
        quantization=None
    ):
        """Initialize the YO-FLO class with configuration options.

        Args:
            model_path (str, optional): Path to the pre-trained model directory. Defaults to None.
            display_inference_speed (bool, optional): Whether to display inference speed. Defaults to False.
            pretty_print (bool, optional): Whether to pretty print detections. Defaults to False.
            inference_limit (float, optional): Limit the inference rate to X inferences per second. Defaults to None.
            class_names (list, optional): List of class names to detect. Defaults to None.
            webcam_indices (list, optional): Indices of the webcams to use. Defaults to None.
            rtsp_urls (list, optional): RTSP URLs for the video streams. Defaults to None.
            record (str, optional): Mode for video recording. Defaults to None.
            quantization (str, optional): Quantization mode ("8bit" or "4bit"). Defaults to None.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.inference_start_time = None
        self.inference_count = 0
        self.class_names = class_names if class_names else []
        self.phrase = None
        self.object_detection_active = False
        self.screenshot_active = False
        self.log_to_file_active = False
        self.headless = True
        self.display_inference_speed = display_inference_speed
        self.stop_webcam_flag = threading.Event()
        self.webcam_threads = []
        self.pretty_print = pretty_print
        self.inference_limit = inference_limit
        self.last_inference_time = 0
        self.inference_phrases = []
        self.webcam_indices = webcam_indices if webcam_indices else [0]
        self.rtsp_urls = rtsp_urls if rtsp_urls else []
        self.record = record
        self.recording = False
        self.video_writer = None
        self.video_out_path = "output.avi"
        self.quantization = quantization
        if model_path:
            self.init_model(model_path)

    def init_model(self, model_path):
        """Initialize the model and processor from the given model path.

        Args:
            model_path (str): Path to the pre-trained model directory.
        """
        if not os.path.exists(model_path):
            logging.error(f"Model path {os.path.abspath(model_path)} does not exist.")
            return
        if not os.path.isdir(model_path):
            logging.error(
                f"Model path {os.path.abspath(model_path)} is not a directory."
            )
            return
        try:
            logging.info(f"Attempting to load model from {os.path.abspath(model_path)}")

            quantization_config = None
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                logging.info("Using 4-bit quantization.")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config
            ).eval()

            if not self.quantization:
                self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            logging.info(
                f"Model loaded successfully from {os.path.abspath(model_path)}"
            )
        except (OSError, ValueError, ModuleNotFoundError) as e:
            logging.error(f"Error initializing model: {e}")
        except Exception as e:
            logging.error(f"Unexpected error initializing model: {e}")

    def update_inference_rate(self):
        """Calculate and log the inference rate (inferences per second)."""
        try:
            if self.inference_start_time is None:
                self.inference_start_time = time.time()
            else:
                elapsed_time = time.time() - self.inference_start_time
                if elapsed_time > 0:
                    inferences_per_second = self.inference_count / elapsed_time
                    if self.display_inference_speed:
                        logging.info(f"IPS: {inferences_per_second:.2f}")
        except Exception as e:
            logging.error(f"Error updating inference rate: {e}")

    def run_object_detection(self, image):
        """Perform object detection on the given image.

        Args:
            image (PIL.Image): The image to perform object detection on.

        Returns:
            dict: The parsed detection results.
        """
        try:
            task_prompt = "<OD>"
            inputs = self.processor(
                text=task_prompt, images=image, return_tensors="pt"
            )

            if not self.quantization:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.quantization:
                dtype = next(self.model.parameters()).dtype
                inputs = {k: v.to(self.model.device, dtype=dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.model.device),
                    pixel_values=inputs.get("pixel_values").to(self.model.device),
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

    def filter_detections(self, detections):
        """Filter detections to include only specified class names.

        Args:
            detections (list): List of detections.

        Returns:
            list: Filtered detections.
        """
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

    def run_expression_comprehension(self, image, phrase):
        """Run expression comprehension on the given image and phrase.

        Args:
            image (PIL.Image): The image to run expression comprehension on.
            phrase (str): The phrase to evaluate.

        Returns:
            str: The generated text.
        """
        try:
            task_prompt = "<CAPTION_TO_EXPRESSION_COMPREHENSION>"
            inputs = self.processor(
                text=task_prompt, images=image, return_tensors="pt"
            )
            inputs["input_ids"] = self.processor.tokenizer(
                phrase, return_tensors="pt"
            ).input_ids

            if not self.quantization:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.quantization:
                dtype = next(self.model.parameters()).dtype
                inputs = {k: v.to(dtype=dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.model.device),
                    pixel_values=inputs.get("pixel_values").to(self.model.device),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
            return generated_text
        except (torch.cuda.CudaError, ModuleNotFoundError) as e:
            logging.error(f"CUDA error during expression comprehension: {e}")
        except Exception as e:
            logging.error(f"Error during expression comprehension: {e}")
        return None

    def plot_bbox(self, image, detections):
        """Draw bounding boxes on the image based on detections.

        Args:
            image (numpy.ndarray): The image to draw bounding boxes on.
            detections (list): List of detections to draw.

        Returns:
            numpy.ndarray: The image with bounding boxes drawn.
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

    def download_model(self):
        """Download the model and processor from Hugging Face Hub."""
        try:
            local_model_dir = "model"
            snapshot_download(
                repo_id="microsoft/Florence-2-base-ft", local_dir=local_model_dir
            )
            if not os.path.exists(local_model_dir):
                logging.error(
                    f"Model download failed, directory {os.path.abspath(local_model_dir)} does not exist."
                )
                return False
            if not os.path.isdir(local_model_dir):
                logging.error(
                    f"Model download failed, path {os.path.abspath(local_model_dir)} is not a directory."
                )
                return False
            logging.info(
                f"Model and associated files downloaded and initialized at {os.path.abspath(local_model_dir)}"
            )
            self.init_model(local_model_dir)
            return True
        except OSError as e:
            logging.error(f"OS error during model download: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
        return False

    def start_webcam_detection(self):
        """Start separate threads for each specified webcam or RTSP stream."""
        try:
            if self.webcam_threads:
                logging.warning("Webcam detection is already running.")
                return
            self.stop_webcam_flag.clear()
            if self.rtsp_urls:
                for rtsp_url in self.rtsp_urls:
                    thread = threading.Thread(
                        target=self._webcam_detection_thread, args=(rtsp_url,)
                    )
                    thread.start()
                    self.webcam_threads.append(thread)
            else:
                for index in self.webcam_indices:
                    thread = threading.Thread(
                        target=self._webcam_detection_thread, args=(index,)
                    )
                    thread.start()
                    self.webcam_threads.append(thread)
        except Exception as e:
            logging.error(f"Error starting webcam detection: {e}")

    def _webcam_detection_thread(self, source):
        """Run the webcam detection loop in a separate thread for a specific webcam or RTSP stream.

        Args:
            source (str or int): The source index or RTSP URL for the webcam.
        """
        try:
            if isinstance(source, str):
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logging.error(f"Error: Could not open video source {source}.")
                return
            window_name = f"Object Detection Source {source}"
            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.error(
                        f"Error: Failed to capture image from source {source}."
                    )
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image)
                current_time = time.time()
                if self.inference_limit:
                    time_since_last_inference = current_time - self.last_inference_time
                    if time_since_last_inference < 1 / self.inference_limit:
                        time.sleep(1 / self.inference_limit - time_since_last_inference)
                    current_time = time.time()
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
                            logging.info(
                                f"Detections from source {source}: {filtered_detections}"
                            )
                        if not self.headless:
                            frame = self.plot_bbox(frame, filtered_detections)
                        self.inference_count += 1
                        self.update_inference_rate()
                        if filtered_detections:
                            if self.screenshot_active:
                                self.save_screenshot(frame)
                            if self.log_to_file_active:
                                self.log_alert(
                                    f"Detections from source {source}: {filtered_detections}"
                                )
                            if self.record:
                                self.start_recording(frame)
                        else:
                            if self.record and self.record == "od":
                                self.stop_recording()
                elif self.phrase:
                    results = self.run_expression_comprehension(image_pil, self.phrase)
                    if results:
                        clean_result = (
                            results.replace("<s>", "")
                            .replace("</s>", "")
                            .strip()
                            .lower()
                        )
                        self.pretty_print_expression(clean_result)
                        self.inference_count += 1
                        self.update_inference_rate()
                        if clean_result in ["yes", "no"]:
                            if self.log_to_file_active:
                                self.log_alert(
                                    f"Expression Comprehension from source {source}: {clean_result} at {datetime.now()}"
                                )
                            if self.record:
                                self.handle_recording_by_inference(clean_result, frame)
                if self.inference_phrases:
                    inference_result, phrase_results = self.evaluate_inference_chain(
                        image_pil
                    )
                    logging.info(
                        f"Inference Chain result from source {source}: {inference_result}, Details: {phrase_results}"
                    )
                    if self.pretty_print:
                        for idx, result in enumerate(phrase_results):
                            logging.info(
                                f"Inference {idx + 1} from source {source}: {'PASS' if result else 'FAIL'}"
                            )
                    self.inference_count += 1
                    self.update_inference_rate()
                if not self.headless:
                    if self.recording:
                        self.video_writer.write(frame)
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                self.last_inference_time = current_time
            cap.release()
            if not self.headless:
                cv2.destroyWindow(window_name)
            if self.recording:
                self.stop_recording()
        except cv2.error as e:
            logging.error(f"OpenCV error in detection thread {source}: {e}")
        except ModuleNotFoundError as e:
            logging.error(f"ModuleNotFoundError in detection thread {source}: {e}")
        except Exception as e:
            logging.error(f"Error in detection thread {source}: {e}")

    def stop_webcam_detection(self):
        """Stop all webcam detection threads."""
        try:
            self.object_detection_active = False
            self.stop_webcam_flag.set()
            for thread in self.webcam_threads:
                thread.join()
            self.webcam_threads = []
            logging.info("Webcam detection stopped")
            if self.recording:
                self.stop_recording()
        except Exception as e:
            logging.error(f"Error stopping webcam detection: {e}")

    def save_screenshot(self, frame):
        """Save a screenshot of the current frame.

        Args:
            frame (numpy.ndarray): The frame to save as a screenshot.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
        except cv2.error as e:
            logging.error(f"OpenCV error saving screenshot: {e}")
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")

    def log_alert(self, message):
        """Log an alert message to a file.

        Args:
            message (str): The alert message to log.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open("alerts.log", "a") as log_file:
                log_file.write(f"{timestamp} - {message}\n")
            logging.info(f"{timestamp} - {message}")
        except IOError as e:
            logging.error(f"IO error logging alert: {e}")
        except Exception as e:
            logging.error(f"Error logging alert: {e}")

    def pretty_print_detections(self, detections):
        """Pretty print the detections to the console.

        Args:
            detections (list): List of detections to print.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info("\n" + "=" * 50)
            for bbox, label in detections:
                bbox_str = (
                    f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
                )
                logging.info(f"- {label}: {bbox_str} at {timestamp}")
            logging.info("=" * 50 + "\n")
        except Exception as e:
            logging.error(f"Error in pretty_print_detections: {e}")

    def pretty_print_expression(self, clean_result):
        """Pretty print the expression comprehension result to the console.

        Args:
            clean_result (str): The clean result to print.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.pretty_print:
                logging.info("\n" + "=" * 50)
                logging.info(f"Expression Comprehension: {clean_result} at {timestamp}")
                logging.info("=" * 50 + "\n")
            else:
                logging.info(f"Expression Comprehension: {clean_result} at {timestamp}")
        except Exception as e:
            logging.error(f"Error in pretty_print_expression: {e}")

    def set_inference_phrases(self, phrases):
        """Set the phrases for the inference chain.

        Args:
            phrases (list): List of phrases for the inference chain.
        """
        self.inference_phrases = phrases
        logging.info(f"Inference phrases set: {self.inference_phrases}")

    def evaluate_inference_chain(self, image):
        """Evaluate the inference chain based on the set phrases.

        Args:
            image (PIL.Image): The image to evaluate.

        Returns:
            tuple: Overall result and individual phrase results.
        """
        try:
            if not self.inference_phrases:
                logging.error("No inference phrases set.")
                return "FAIL", []
            results = []
            for phrase in self.inference_phrases:
                result = self.run_expression_comprehension(image, phrase)
                if result:
                    if "yes" in result.lower():
                        results.append(True)
                    else:
                        results.append(False)
            overall_result = "PASS" if results.count(True) >= 2 else "FAIL"
            return overall_result, results
        except Exception as e:
            logging.error(f"Error evaluating inference chain: {e}")
            return "FAIL", []

    def start_recording(self, frame):
        """Start recording the video.

        Args:
            frame (numpy.ndarray): The frame to use for setting the video writer.
        """
        try:
            if not self.recording:
                height, width, _ = frame.shape
                self.video_writer = cv2.VideoWriter(
                    self.video_out_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    20.0,
                    (width, height)
                )
                self.recording = True
                logging.info(f"Started recording video: {self.video_out_path}")
        except Exception as e:
            logging.error(f"Error starting video recording: {e}")

    def stop_recording(self):
        """Stop recording the video."""
        try:
            if self.recording:
                self.video_writer.release()
                self.recording = False
                logging.info(f"Stopped recording video: {self.video_out_path}")
        except Exception as e:
            logging.error(f"Error stopping video recording: {e}")

    def handle_recording_by_inference(self, inference_result, frame):
        """Handle recording based on inference result.

        Args:
            inference_result (str): The inference result ("yes" or "no").
            frame (numpy.ndarray): The frame to use for recording.
        """
        try:
            if self.record == "infy" and inference_result == "yes":
                self.start_recording(frame)
            elif self.record == "infy" and inference_result == "no":
                self.stop_recording()
            elif self.record == "infn" and inference_result == "no":
                self.start_recording(frame)
            elif self.record == "infn" and inference_result == "yes":
                self.stop_recording()
        except Exception as e:
            logging.error(f"Error handling recording by inference: {e}")

def main():
    """Parse command-line arguments and run the YO-FLO application."""
    parser = argparse.ArgumentParser(
        description="YO-FLO: A proof-of-concept in using advanced vision-language models as a YOLO alternative."
    )
    parser.add_argument(
        "-od",
        "--object_detection",
        nargs="*",
        help='Enable object detection with optional class names to detect (e.g., "cat", "dog"). Specify class names in quotes.',
    )
    parser.add_argument(
        "-ph",
        "--phrase",
        type=str,
        help="Yes/No question for expression comprehension (e.g., 'Is the person smiling?'). This will check the presence of specific expressions in the captured images.",
    )
    parser.add_argument(
        "-hl",
        "--headless",
        action="store_true",
        help="Run in headless mode without displaying video. Useful for running on servers without a display.",
    )
    parser.add_argument(
        "-ss",
        "--screenshot",
        action="store_true",
        help="Enable screenshot on detection. Saves an image file when detections are made.",
    )
    parser.add_argument(
        "-lf",
        "--log_to_file",
        action="store_true",
        help="Enable logging alerts to file. Logs will be saved in 'alerts.log'.",
    )
    parser.add_argument(
        "-is",
        "--display_inference_speed",
        action="store_true",
        help="Display inference speed (inferences per second) in the console output.",
    )
    parser.add_argument(
        "-pp",
        "--pretty_print",
        action="store_true",
        help="Enable pretty print for detections. Formats and prints detection results nicely in the console.",
    )
    parser.add_argument(
        "-il",
        "--inference_limit",
        type=float,
        help="Limit the inference rate to X inferences per second. Useful for controlling the load on the system.",
        required=False,
    )
    parser.add_argument(
        "-ic",
        "--inference_chain",
        nargs="+",
        help="Enable inference chain with specified phrases. Provide phrases in quotes, separated by spaces (e.g., 'Is it sunny?' 'Is it raining?').",
    )
    parser.add_argument(
        "-wi",
        "--webcam_indices",
        nargs="+",
        type=int,
        help="Specify the indices of the webcams to use (e.g., 0 1 2).",
    )
    parser.add_argument(
        "-rtsp",
        "--rtsp_urls",
        nargs="+",
        type=str,
        help="Specify the RTSP URLs for the video streams.",
    )
    parser.add_argument(
        "-r",
        "--record",
        choices=["od", "infy", "infn"],
        help="Enable video recording and specify the recording mode: 'od' to start/stop based on object detection, 'infy' to start on 'yes' inference and stop on 'no', and 'infn' to start on 'no' inference and stop on 'yes'.",
    )

    parser.add_argument(
        "-4bit",
        "--four_bit",
        action="store_true",
        help="Enable 4-bit quantization for model loading.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-mp",
        "--model_path",
        type=str,
        help="Path to the pre-trained model directory. Use this if you have a local copy of the model.",
    )
    group.add_argument(
        "-dm",
        "--download_model",
        action="store_true",
        help="Download model from Hugging Face. Use this if you want to download the model files automatically.",
    )

    args = parser.parse_args()
    if not args.model_path and not args.download_model:
        parser.error("You must specify either --model_path or --download_model.")

    quantization_mode = "4bit" if args.four_bit else None

    try:
        setup_logging(args.log_to_file)
        webcam_indices = args.webcam_indices if args.webcam_indices else [0]
        rtsp_urls = args.rtsp_urls if args.rtsp_urls else []
        if args.download_model:
            yo_flo = YOFLO(
                display_inference_speed=args.display_inference_speed,
                pretty_print=args.pretty_print,
                inference_limit=args.inference_limit,
                class_names=args.object_detection,
                webcam_indices=webcam_indices,
                rtsp_urls=rtsp_urls,
                record=args.record,
                quantization=quantization_mode
            )
            if not yo_flo.download_model():
                return
        else:
            if not os.path.exists(args.model_path):
                logging.error(f"Model path {args.model_path} does not exist.")
                return
            if not os.path.isdir(args.model_path):
                logging.error(f"Model path {args.model_path} is not a directory.")
                return
            yo_flo = YOFLO(
                model_path=args.model_path,
                display_inference_speed=args.display_inference_speed,
                pretty_print=args.pretty_print,
                inference_limit=args.inference_limit,
                class_names=args.object_detection,
                webcam_indices=webcam_indices,
                rtsp_urls=rtsp_urls,
                record=args.record,
                quantization=quantization_mode
            )
        if args.phrase:
            yo_flo.phrase = args.phrase
        if args.inference_chain:
            yo_flo.set_inference_phrases(args.inference_chain)
        yo_flo.headless = args.headless
        yo_flo.object_detection_active = args.object_detection is not None
        yo_flo.screenshot_active = args.screenshot
        yo_flo.log_to_file_active = args.log_to_file
        yo_flo.start_webcam_detection()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            yo_flo.stop_webcam_detection()
    except Exception as e:
        logging.error(f"An error occurred during main loop: {e}")
    else:
        input("Press Enter to stop...")
        yo_flo.stop_webcam_detection()

if __name__ == "__main__":
    main()
