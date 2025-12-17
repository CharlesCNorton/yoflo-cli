"""Video recording management with detection/inference triggers."""

import time
import logging
from datetime import datetime

import cv2


class RecordingManager:
    """Manages video recording with configurable trigger modes."""

    MODES = ('od', 'infy', 'infn')

    def __init__(self, record_mode=None, output_dir="."):
        """
        Initialize recording manager.

        :param record_mode: Trigger mode - 'od' (object detection),
                          'infy' (start on yes), 'infn' (start on no).
        :param output_dir: Directory for output files.
        """
        self.record_mode = record_mode
        self.output_dir = output_dir
        self.recording = False
        self.video_writer = None
        self.video_out_path = None
        self.last_detection_time = time.time()

    def _generate_filename(self):
        """Generate timestamped output filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.output_dir}/output_{timestamp}.avi"

    def start_recording(self, frame):
        """Start video recording."""
        if self.recording or not self.record_mode:
            return

        height, width = frame.shape[:2]
        self.video_out_path = self._generate_filename()
        self.video_writer = cv2.VideoWriter(
            self.video_out_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            20.0,
            (width, height),
        )
        self.recording = True
        logging.info(f"Started recording: {self.video_out_path}")

    def stop_recording(self):
        """Stop video recording."""
        if not self.recording:
            return

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
        logging.info(f"Stopped recording: {self.video_out_path}")

    def write_frame(self, frame):
        """Write a frame to the video file."""
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def handle_detection_trigger(self, detections, frame):
        """Handle recording based on object detections."""
        if self.record_mode != 'od':
            return

        current_time = time.time()
        if detections:
            self.start_recording(frame)
            self.last_detection_time = current_time
        elif (current_time - self.last_detection_time) > 1:
            self.stop_recording()

    def handle_inference_trigger(self, inference_result, frame):
        """Handle recording based on inference results (yes/no)."""
        if self.record_mode == "infy":
            if inference_result == "yes":
                self.start_recording(frame)
            elif inference_result == "no":
                self.stop_recording()
        elif self.record_mode == "infn":
            if inference_result == "no":
                self.start_recording(frame)
            elif inference_result == "yes":
                self.stop_recording()

    def cleanup(self):
        """Clean up resources."""
        self.stop_recording()
