import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
import threading
import os
import time
from datetime import datetime
from huggingface_hub import hf_hub_download
import argparse
import logging

class YO_FLO:
    def __init__(self, model_path=None, debug=False, display_inference_speed=False, pretty_print=False, alert_on="yes"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.inference_start_time = None
        self.inference_count = 0
        self.class_name = None
        self.phrase = None
        self.debug = debug
        self.object_detection_active = False
        self.screenshot_active = False
        self.log_to_file_active = False
        self.headless = True
        self.display_inference_speed = display_inference_speed
        self.stop_webcam_flag = threading.Event()
        self.last_beep_time = 0
        self.webcam_thread = None
        self.pretty_print = pretty_print
        self.alert_on = alert_on.lower()

        if model_path:
            self.init_model(model_path)

    def init_model(self, model_path):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval().to(self.device).half()
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            print("Model loaded successfully in fp16")
        except Exception as e:
            print(f"Error initializing model: {e}")

    def update_inference_rate(self):
        if self.inference_start_time is None:
            self.inference_start_time = time.time()
        else:
            elapsed_time = time.time() - self.inference_start_time
            if elapsed_time > 0:
                inferences_per_second = self.inference_count / elapsed_time
                if self.display_inference_speed:
                    print(f"Inferences/sec: {inferences_per_second:.2f}")

    def run_object_detection(self, image):
        try:
            task_prompt = '<OD>'
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
            with torch.amp.autocast('cuda'):
                generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs.get("pixel_values"), max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=1)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
            return parsed_answer
        except Exception as e:
            print(f"Error during object detection: {e}")
            return None

    def run_expression_comprehension(self, image, phrase):
        try:
            task_prompt = '<CAPTION_TO_EXPRESSION_COMPREHENSION>'
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
            inputs["input_ids"] = self.processor.tokenizer(phrase, return_tensors="pt").input_ids.to(self.device)
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
            with torch.amp.autocast('cuda'):
                generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs.get("pixel_values"), max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=1)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            return generated_text
        except Exception as e:
            print(f"Error during expression comprehension: {e}")
            return None

    def plot_bbox(self, image, detections):
        try:
            for bbox, label in detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image
        except Exception as e:
            print(f"Error plotting bounding boxes: {e}")
            return image

    def download_model(self):
        try:
            model_name = "microsoft/Florence-2-base-ft"
            model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            processor_path = hf_hub_download(repo_id=model_name, filename="preprocessor_config.json")
            local_model_dir = os.path.dirname(model_path)
            self.init_model(local_model_dir)
            print("Model downloaded and initialized")
        except Exception as e:
            print(f"Error downloading model: {e}")

    def start_webcam_detection(self):
        if self.webcam_thread and self.webcam_thread.is_alive():
            print("Webcam detection is already running.")
            return
        self.stop_webcam_flag.clear()
        self.webcam_thread = threading.Thread(target=self._webcam_detection_thread)
        self.webcam_thread.start()

    def _webcam_detection_thread(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while not self.stop_webcam_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)

            if self.object_detection_active:
                results = self.run_object_detection(image_pil)
                if results and '<OD>' in results:
                    detections = [(bbox, label) for bbox, label in zip(results['<OD>']['bboxes'], results['<OD>']['labels'])]
                    if self.headless:
                        if self.pretty_print:
                            self.pretty_print_detections(detections)
                        else:
                            print("Detections:", detections)
                    else:
                        frame = self.plot_bbox(frame, detections)
                    self.inference_count += 1
                    self.update_inference_rate()

                    if detections:
                        if self.screenshot_active and not self.headless:
                            self.save_screenshot(frame)

                        if self.log_to_file_active:
                            self.log_alert("Detections: " + str(detections))

            elif self.phrase:
                results = self.run_expression_comprehension(image_pil, self.phrase)
                if results:
                    clean_result = results.replace('<s>', '').replace('</s>', '').strip().lower()
                    if self.headless:
                        if self.pretty_print:
                            self.pretty_print_expression(results)
                        else:
                            print("Expression Comprehension:", results)
                    self.inference_count += 1
                    self.update_inference_rate()

                        if clean_result == self.alert_on:
                        if self.screenshot_active and not self.headless:
                            self.save_screenshot(frame)

                        if self.log_to_file_active:
                            self.log_alert(f"Expression Comprehension: {clean_result} at {datetime.now()}")

            if not self.headless:
                cv2.imshow('Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if not self.headless:
            cv2.destroyAllWindows()

    def stop_webcam_detection(self):
        self.object_detection_active = False
        self.stop_webcam_flag.set()
        if self.webcam_thread:
            self.webcam_thread.join()
        print("Webcam detection stopped")

    def save_screenshot(self, frame):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")

    def log_alert(self, message):
        try:
            with open("alerts.log", "a") as log_file:
                log_file.write(f"{datetime.now()} - {message}\n")
            print(f"Logged alert: {message}")
        except Exception as e:
            print(f"Error logging alert: {e}")

    def toggle_screenshot(self):
        self.screenshot_active = not self.screenshot_active
        print(f"Screenshot on detection is now {'enabled' if self.screenshot_active else 'disabled'}")

    def toggle_log_to_file(self):
        self.log_to_file_active = not self.log_to_file_active
        print(f"Log to file on detection is now {'enabled' if self.log_to_file_active else 'disabled'}")

    def toggle_object_detection(self):
        self.object_detection_active = not self.object_detection_active
        print(f"Object detection is now {'enabled' if self.object_detection_active else 'disabled'}")

    def toggle_headless(self):
        self.headless = not self.headless
        print(f"Headless mode is now {'enabled' if self.headless else 'disabled'}")

    def pretty_print_detections(self, detections):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Detections:")
        for bbox, label in detections:
            bbox_str = f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
            print(f"- {label}: {bbox_str} at {timestamp}")

    def pretty_print_expression(self, results):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        clean_result = results.replace('<s>', '').replace('</s>', '').strip()
        print(f"Expression Comprehension: {clean_result} at {timestamp}")

def main():
    parser = argparse.ArgumentParser(description="YO-FLO: A proof-of-concept in using advanced vision models as a YOLO alternative.")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model directory", required=False)
    parser.add_argument("--class_name", type=str, help="Class name to detect (e.g., 'cat', 'dog')")
    parser.add_argument("--phrase", type=str, help="Yes/No question for expression comprehension (e.g., 'Is the person smiling?')")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode without displaying video")
    parser.add_argument("--screenshot", action='store_true', help="Enable screenshot on detection")
    parser.add_argument("--log_to_file", action='store_true', help="Enable logging alerts to file")
    parser.add_argument("--inference_speed", action='store_true', help="Display inference speed")
    parser.add_argument("--object_detection", action='store_true', help="Enable object detection")
    parser.add_argument("--download_model", action='store_true', help="Download model from Hugging Face")
    parser.add_argument("--pretty_print", action='store_true', help="Enable pretty print for detections")
    parser.add_argument("--alert_on", type=str, choices=["yes", "no"], default="yes", help="Trigger alert on 'yes' or 'no' result")

    args = parser.parse_args()

    if not args.model_path and not args.download_model:
        parser.error("You must specify either --model_path or --download_model.")

    try:
        if args.download_model:
            yo_flo = YO_FLO(debug=args.debug, display_inference_speed=args.inference_speed, pretty_print=args.pretty_print, alert_on=args.alert_on)
            yo_flo.download_model()
        else:
            yo_flo = YO_FLO(model_path=args.model_path, debug=args.debug, display_inference_speed=args.inference_speed, pretty_print=args.pretty_print, alert_on=args.alert_on)

        if args.class_name:
            yo_flo.class_name = args.class_name

        if args.phrase:
            yo_flo.phrase = args.phrase

        yo_flo.headless = args.headless

        if args.object_detection:
            yo_flo.toggle_object_detection()

        if args.screenshot:
            yo_flo.toggle_screenshot()

        if args.log_to_file:
            yo_flo.toggle_log_to_file()

        yo_flo.start_webcam_detection()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            yo_flo.stop_webcam_detection()
        else:
            input("Press Enter to stop...")
            yo_flo.stop_webcam_detection()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
