import argparse
import logging
import os
import threading
import time
from datetime import datetime
import cv2
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def setup_logging(log_to_file, log_file_path="alerts.log"):
    """Set up logging to file and/or console."""
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file_path))
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)

class YOFLO:
    def __init__(self, model_path=None, display_inference_speed=False, pretty_print=False, inference_limit=None, class_names=None):
        """Initialize the YO-FLO class with configuration options."""
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
        self.last_beep_time = 0
        self.webcam_thread = None
        self.pretty_print = pretty_print
        self.inference_limit = inference_limit
        self.last_inference_time = 0
        self.last_detection = None
        self.last_detection_count = 0
        if model_path:
            self.init_model(model_path)

    def init_model(self, model_path):
        """Initialize the model and processor from the given model path."""
        if not os.path.exists(model_path):
            logging.error(f"Model path {model_path} does not exist.")
            return
        if not os.path.isdir(model_path):
            logging.error(f"Model path {model_path} is not a directory.")
            return

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device).half()
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logging.info("Model loaded successfully")
        except (OSError, ValueError, torch.nn.ModuleNotFoundError) as e:
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
        """Perform object detection on the given image."""
        try:
            task_prompt = '<OD>'
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
            with torch.amp.autocast('cuda'):
                generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs.get("pixel_values"), max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=1)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
            return parsed_answer
        except (torch.cuda.CudaError, torch.nn.ModuleNotFoundError) as e:
            logging.error(f"CUDA error during object detection: {e}")
        except Exception as e:
            logging.error(f"Error during object detection: {e}")
        return None

    def filter_detections(self, detections):
        """Filter detections to include only specified class names."""
        try:
            if not self.class_names:
                return detections
            return [(bbox, label) for bbox, label in detections if label.lower() in [name.lower() for name in self.class_names]]
        except Exception as e:
            logging.error(f"Error filtering detections: {e}")
        return detections

    def run_expression_comprehension(self, image, phrase):
        """Run expression comprehension on the given image and phrase."""
        try:
            task_prompt = '<CAPTION_TO_EXPRESSION_COMPREHENSION>'
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            inputs["input_ids"] = self.processor.tokenizer(phrase, return_tensors="pt").input_ids.to(self.device)
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
            with torch.amp.autocast('cuda'):
                generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs.get("pixel_values"), max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=1)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            return generated_text
        except (torch.cuda.CudaError, torch.nn.ModuleNotFoundError) as e:
            logging.error(f"CUDA error during expression comprehension: {e}")
        except Exception as e:
            logging.error(f"Error during expression comprehension: {e}")
        return None

    def plot_bbox(self, image, detections):
        """Draw bounding boxes on the image based on detections."""
        try:
            for bbox, label in detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image
        except cv2.error as e:
            logging.error(f"OpenCV error plotting bounding boxes: {e}")
        except Exception as e:
            logging.error(f"Error plotting bounding boxes: {e}")
        return image

    def download_model(self):
        """Download the model and processor from Hugging Face Hub."""
        file_urls = {
            "config.json": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/config.json",
            "tokenizer.json": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/tokenizer.json",
            "tokenizer_config.json": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/tokenizer_config.json",
            "vocab.json": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/vocab.json",
            "pytorch_model.bin": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/pytorch_model.bin",
            "processing_florence2.py": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/processing_florence2.py",
            "preprocessor_config.json": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/preprocessor_config.json",
            "modeling_florence2.py": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/modeling_florence2.py",
            "configuration_florence2.py": "https://huggingface.co/microsoft/Florence-2-base-ft/resolve/main/configuration_florence2.py"
        }
        try:
            local_model_dir = "model"
            os.makedirs(local_model_dir, exist_ok=True)
            for filename, url in file_urls.items():
                local_path = os.path.join(local_model_dir, filename)
                if not os.path.exists(local_path):
                    hf_hub_download(repo_id="microsoft/Florence-2-base-ft", filename=filename, local_dir=local_model_dir)
            self.init_model(local_model_dir)
            logging.info("Model and associated files downloaded and initialized")
        except OSError as e:
            logging.error(f"OS error during model download: {e}")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")

    def start_webcam_detection(self):
        """Start a separate thread for webcam detection."""
        try:
            if self.webcam_thread and self.webcam_thread.is_alive():
                logging.warning("Webcam detection is already running.")
                return
            self.stop_webcam_flag.clear()
            self.webcam_thread = threading.Thread(target=self._webcam_detection_thread)
            self.webcam_thread.start()
        except Exception as e:
            logging.error(f"Error starting webcam detection: {e}")

    def _webcam_detection_thread(self):
        """Run the webcam detection loop in a separate thread."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Error: Could not open webcam.")
                return
            while not self.stop_webcam_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.error("Error: Failed to capture image from webcam.")
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
                    if results and '<OD>' in results:
                        detections = [(bbox, label) for bbox, label in zip(results['<OD>']['bboxes'], results['<OD>']['labels'])]
                        filtered_detections = self.filter_detections(detections)
                        if self.pretty_print:
                            self.pretty_print_detections(filtered_detections)
                        else:
                            logging.info(f"Detections: {filtered_detections}")
                        if not self.headless:
                            frame = self.plot_bbox(frame, filtered_detections)
                        self.inference_count += 1
                        self.update_inference_rate()
                        if filtered_detections:
                            if self.screenshot_active:
                                self.save_screenshot(frame)
                            if self.log_to_file_active:
                                self.log_alert(f"Detections: {filtered_detections}")
                elif self.phrase:
                    results = self.run_expression_comprehension(image_pil, self.phrase)
                    if results:
                        clean_result = results.replace('<s>', '').replace('</s>', '').strip().lower()
                        logging.info(f"Expression Comprehension: {clean_result}")
                        if self.pretty_print:
                            self.pretty_print_expression(clean_result)
                        self.inference_count += 1
                        self.update_inference_rate()
                        if clean_result in ['yes', 'no']:
                            if self.log_to_file_active:
                                self.log_alert(f"Expression Comprehension: {clean_result} at {datetime.now()}")
                if not self.headless:
                    cv2.imshow('Object Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                self.last_inference_time = current_time
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
        except cv2.error as e:
            logging.error(f"OpenCV error in webcam detection thread: {e}")
        except Exception as e:
            logging.error(f"Error in webcam detection thread: {e}")

    def stop_webcam_detection(self):
        """Stop the webcam detection thread."""
        try:
            self.object_detection_active = False
            self.stop_webcam_flag.set()
            if self.webcam_thread:
                self.webcam_thread.join()
            logging.info("Webcam detection stopped")
        except Exception as e:
            logging.error(f"Error stopping webcam detection: {e}")

    def save_screenshot(self, frame):
        """Save a screenshot of the current frame."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
        except cv2.error as e:
            logging.error(f"OpenCV error saving screenshot: {e}")
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")

    def log_alert(self, message):
        """Log an alert message to a file."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            with open("alerts.log", "a") as log_file:
                log_file.write(f"{timestamp} - {message}\n")
            logging.info(f"{timestamp} - {message}")
        except IOError as e:
            logging.error(f"IO error logging alert: {e}")
        except Exception as e:
            logging.error(f"Error logging alert: {e}")

    def pretty_print_detections(self, detections):
        """Pretty print the detections to the console."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info("\n" + "="*50)
            for bbox, label in detections:
                bbox_str = f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
                logging.info(f"- {label}: {bbox_str} at {timestamp}")
            logging.info("="*50 + "\n")
        except Exception as e:
            logging.error(f"Error in pretty_print_detections: {e}")

    def pretty_print_expression(self, clean_result):
        """Pretty print the expression comprehension result to the console."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Expression Comprehension: {clean_result} at {timestamp}")
        except Exception as e:
            logging.error(f"Error in pretty_print_expression: {e}")

def main():
    """Parse command-line arguments and run the YO-FLO application."""
    parser = argparse.ArgumentParser(description="YO-FLO: A proof-of-concept in using advanced vision-language models as a YOLO alternative.")
    parser.add_argument("-od", "--object_detection", nargs='*', help='Enable object detection with optional class names in quotes to detect (e.g., `"cat"`, `"dog"`)')
    parser.add_argument("-ph", "--phrase", type=str, help="Yes/No question for expression comprehension (e.g., 'Is the person smiling?')")
    parser.add_argument("-hl", "--headless", action='store_true', help="Run in headless mode without displaying video")
    parser.add_argument("-ss", "--screenshot", action='store_true', help="Enable screenshot on detection")
    parser.add_argument("-lf", "--log_to_file", action='store_true', help="Enable logging alerts to file")
    parser.add_argument("-is", "--display_inference_speed", action='store_true', help="Display inference speed")
    parser.add_argument("-pp", "--pretty_print", action='store_true', help="Enable pretty print for detections")
    parser.add_argument("-il", "--inference_limit", type=float, help="Limit the inference rate to X inferences per second", required=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-mp", "--model_path", type=str, help="Path to the pre-trained model directory")
    group.add_argument("-dm", "--download_model", action='store_true', help="Download model from Hugging Face")

    args = parser.parse_args()
    if not args.model_path and not args.download_model:
        parser.error("You must specify either --model_path or --download_model.")

    try:
        setup_logging(args.log_to_file)
        if args.download_model:
            yo_flo = YOFLO(display_inference_speed=args.display_inference_speed, pretty_print=args.pretty_print, inference_limit=args.inference_limit, class_names=args.object_detection)
            yo_flo.download_model()
        else:
            if not os.path.exists(args.model_path):
                logging.error(f"Model path {args.model_path} does not exist.")
                return
            if not os.path.isdir(args.model_path):
                logging.error(f"Model path {args.model_path} is not a directory.")
                return
            yo_flo = YOFLO(model_path=args.model_path, display_inference_speed=args.display_inference_speed, pretty_print=args.pretty_print, inference_limit=args.inference_limit, class_names=args.object_detection)

        if args.phrase:
            yo_flo.phrase = args.phrase
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
