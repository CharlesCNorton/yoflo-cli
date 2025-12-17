"""Utility functions for image processing and logging."""

import logging
from datetime import datetime

import cv2

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    yt_dlp = None


class ImageUtils:
    """Utility class for image operations."""

    @staticmethod
    def draw_bboxes(image, detections):
        """
        Draw bounding boxes and labels on an image.

        :param image: OpenCV image (numpy array).
        :param detections: List of (bbox, label) tuples.
        :return: Image with bounding boxes drawn.
        """
        try:
            for bbox, label in detections:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        except Exception as e:
            logging.error(f"Error drawing bounding boxes: {e}")
        return image

    @staticmethod
    def save_screenshot(frame, output_dir="."):
        """Save a timestamped screenshot."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logging.info(f"Screenshot saved: {filename}")
            return filename
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")
            return None


class AlertLogger:
    """Logger for detection alerts."""

    def __init__(self, log_file="alerts.log"):
        self.log_file = log_file

    def log_alert(self, message):
        """Append an alert message to the log file."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp} - {message}\n")
            logging.info(f"{timestamp} - {message}")
        except Exception as e:
            logging.error(f"Error logging alert: {e}")


def get_youtube_stream_url(youtube_url):
    """
    Extract direct stream URL from YouTube using yt-dlp.

    :param youtube_url: YouTube video URL.
    :return: Direct stream URL or None if extraction fails.
    """
    if not YT_DLP_AVAILABLE:
        logging.error("yt-dlp not installed. Cannot process YouTube streams.")
        return None

    ydl_opts = {
        'format': 'best[ext=m3u8]/best',
        'quiet': True,
        'skip_download': True,
        'simulate': True,
        'forceurl': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get('url')
    except Exception as e:
        logging.error(f"Error extracting YouTube stream URL: {e}")
        return None


def setup_logging(log_to_file=False, log_file="alerts.log"):
    """Configure application logging."""
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=handlers
    )
