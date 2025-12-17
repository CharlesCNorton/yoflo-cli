"""Core YOFLO class for object detection and visual Q&A."""

import time
import logging
import threading
from datetime import datetime

import cv2
import torch
from PIL import Image

from .model import ModelManager
from .recording import RecordingManager
from .utils import ImageUtils, AlertLogger, get_youtube_stream_url


class YOFLO:
    """
    Main class for object detection and expression comprehension using Florence-2.
    Handles webcam/RTSP streams, optional PTZ tracking, screenshots, logging, and recording.
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or []
        self.phrase = None
        self.inference_phrases = []

        self.object_detection_active = False
        self.screenshot_active = False
        self.log_to_file_active = False
        self.headless = True

        self.display_inference_rate = display_inference_rate
        self.pretty_print = pretty_print
        self.inference_limit = inference_limit
        self.quantization = quantization

        self.webcam_indices = webcam_indices or [0]
        self.rtsp_urls = rtsp_urls or []
        self.track_object_name = track_object_name
        self.ptz_tracker = ptz_tracker

        self.inference_start_time = None
        self.inference_count = 0
        self.last_inference_time = 0

        self.stop_webcam_flag = threading.Event()
        self.webcam_threads = []

        self.recording_manager = RecordingManager(record)
        self.model_manager = ModelManager(self.device, quantization)
        self.alert_logger = AlertLogger()

        if model_path:
            self.model_manager.load_local_model(model_path)

    @property
    def model(self):
        return self.model_manager.model

    @property
    def processor(self):
        return self.model_manager.processor

    def download_model(self, repo_id=None):
        """Download model from HuggingFace."""
        return self.model_manager.download_and_load_model(repo_id)

    def run_object_detection(self, image):
        """
        Run object detection on a PIL image.

        :param image: PIL Image.
        :return: Detection results dict or None.
        """
        if not self.model_manager.is_loaded:
            logging.error("Model not loaded.")
            return None

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
                    pixel_values=inputs["pixel_values"].to(self.device),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                return self.processor.post_process_generation(
                    generated_text, task=task_prompt, image_size=image.size
                )
        except Exception as e:
            logging.error(f"Error during object detection: {e}")
            return None

    def run_expression_comprehension(self, image, phrase):
        """
        Run yes/no question answering on a PIL image.

        :param image: PIL Image.
        :param phrase: Question to answer.
        :return: Generated text response or None.
        """
        if not self.model_manager.is_loaded:
            logging.error("Model not loaded.")
            return None

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
                    pixel_values=inputs["pixel_values"].to(self.device),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )
                return self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
        except Exception as e:
            logging.error(f"Error during expression comprehension: {e}")
            return None

    def filter_detections(self, detections):
        """Filter detections by class names if specified."""
        if not self.class_names:
            return detections
        return [
            (bbox, label) for bbox, label in detections
            if label.lower() in [name.lower() for name in self.class_names]
        ]

    def set_inference_phrases(self, phrases):
        """Set multiple phrases for inference chain."""
        self.inference_phrases = phrases
        logging.info(f"Inference phrases set: {phrases}")

    def evaluate_inference_chain(self, image):
        """Evaluate multiple yes/no queries, return pass/fail."""
        if not self.inference_phrases:
            return "FAIL", []

        results = []
        for phrase in self.inference_phrases:
            result = self.run_expression_comprehension(image, phrase)
            if result:
                results.append("yes" in result.lower())

        overall = "PASS" if results.count(True) >= 2 else "FAIL"
        return overall, results

    def _update_inference_rate(self):
        """Update and optionally display inference rate."""
        if self.inference_start_time is None:
            self.inference_start_time = time.time()
        elif self.display_inference_rate:
            elapsed = time.time() - self.inference_start_time
            if elapsed > 0:
                rate = self.inference_count / elapsed
                logging.info(f"IPS: {rate:.2f}")

    def _pretty_print_detections(self, detections):
        """Log detections in formatted output."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info("\n" + "=" * 50)
        for bbox, label in detections:
            bbox_str = f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
            logging.info(f"- {label}: {bbox_str} at {timestamp}")
        logging.info("=" * 50 + "\n")

    def _pretty_print_expression(self, result):
        """Log expression result in formatted output."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean = result.replace("</s>", "").replace("<s>", "").strip()
        logging.info("\n" + "=" * 50)
        logging.info(f"Expression Comprehension: {clean} at {timestamp}")
        logging.info("=" * 50 + "\n")

    def _pick_tracked_object(self, detections):
        """Choose largest bounding box matching track_object_name."""
        if not self.track_object_name:
            return None

        candidates = [
            (bbox, label) for bbox, label in detections
            if label.lower() == self.track_object_name.lower()
        ]
        if not candidates:
            return None

        def area(bbox):
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return max(candidates, key=lambda x: area(x[0]))[0]

    def _process_frame(self, frame, image_pil, source):
        """Process a single frame for detection/inference."""
        primary_bbox = None
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.object_detection_active:
            results = self.run_object_detection(image_pil)
            if results and "<OD>" in results:
                detections = list(zip(results["<OD>"]["bboxes"], results["<OD>"]["labels"]))
                filtered = self.filter_detections(detections)

                if self.pretty_print:
                    self._pretty_print_detections(filtered)
                else:
                    logging.info(f"Detections from {source}: {filtered}")

                if not self.headless:
                    frame = ImageUtils.draw_bboxes(frame, filtered)

                self.inference_count += 1
                self._update_inference_rate()

                if filtered:
                    if self.screenshot_active:
                        ImageUtils.save_screenshot(frame)
                    if self.log_to_file_active:
                        self.alert_logger.log_alert(f"Detections from {source}: {filtered}")

                self.recording_manager.handle_detection_trigger(filtered, frame)

                if self.ptz_tracker and self.ptz_tracker.active:
                    primary_bbox = self._pick_tracked_object(filtered)

        if self.phrase:
            result = self.run_expression_comprehension(image_pil, self.phrase)
            if result:
                clean = result.lower()
                if self.pretty_print:
                    self._pretty_print_expression(clean)
                else:
                    logging.info(f"Expression from {source}: {clean}")

                self.inference_count += 1
                self._update_inference_rate()

                if "yes" in clean:
                    if self.log_to_file_active:
                        self.alert_logger.log_alert(f"Expression from {source}: yes")
                    self.recording_manager.handle_inference_trigger("yes", frame)
                elif "no" in clean:
                    if self.log_to_file_active:
                        self.alert_logger.log_alert(f"Expression from {source}: no")
                    self.recording_manager.handle_inference_trigger("no", frame)

        if self.inference_phrases:
            overall, phrase_results = self.evaluate_inference_chain(image_pil)
            logging.info(f"Inference chain from {source}: {overall}, Details: {phrase_results}")
            self.inference_count += 1
            self._update_inference_rate()

        if self.ptz_tracker and self.ptz_tracker.active and primary_bbox:
            h, w = frame.shape[:2]
            self.ptz_tracker.adjust_camera(primary_bbox, w, h)

        return frame

    def _webcam_detection_thread(self, source):
        """Thread function for processing a video source."""
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logging.error(f"Could not open video source {source}.")
                return

            window_name = f"YOFLO - Source {source}"

            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.error(f"Failed to capture from {source}.")
                    break

                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if self.inference_limit:
                    elapsed = time.time() - self.last_inference_time
                    if elapsed < 1 / self.inference_limit:
                        time.sleep(1 / self.inference_limit - elapsed)

                frame = self._process_frame(frame, image_pil, source)
                self.last_inference_time = time.time()

                if not self.headless:
                    if self.recording_manager.recording:
                        self.recording_manager.write_frame(frame)
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            cap.release()
            if not self.headless:
                cv2.destroyWindow(window_name)
            self.recording_manager.cleanup()

        except Exception as e:
            logging.error(f"Error in detection thread {source}: {e}")

    def start_webcam_detection(self):
        """Start detection threads for all configured sources."""
        if self.webcam_threads:
            logging.warning("Detection already running.")
            return

        self.stop_webcam_flag.clear()
        sources = self.rtsp_urls if self.rtsp_urls else self.webcam_indices

        for source in sources:
            thread = threading.Thread(target=self._webcam_detection_thread, args=(source,))
            thread.start()
            self.webcam_threads.append(thread)

    def stop_webcam_detection(self):
        """Stop all detection threads."""
        self.object_detection_active = False
        self.stop_webcam_flag.set()

        for thread in self.webcam_threads:
            thread.join()

        self.webcam_threads = []
        self.recording_manager.cleanup()
        logging.info("Detection stopped.")
