"""PTZ autonomous tracking logic."""

import time


class PTZTracker:
    """
    Autonomous PTZ tracking class. Keeps a specified object centered and at a desired size.
    Adjusts camera pan, tilt, and zoom automatically to keep the detected object
    within a certain bounding box ratio. Smooths the bounding box to reduce jitter.
    """

    def __init__(
        self,
        camera,
        desired_ratio=0.20,
        zoom_tolerance=0.4,
        pan_tilt_tolerance=25,
        pan_tilt_interval=0.75,
        zoom_interval=0.5,
        smoothing_factor=0.2,
        max_consecutive_errors=5
    ):
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
        """Activate or deactivate PTZ tracking."""
        self.active = active
        if not active:
            self.smoothed_width = None
            self.smoothed_height = None
            self.consecutive_errors = 0

    def adjust_camera(self, bbox, frame_width, frame_height):
        """Adjusts camera pan, tilt, and zoom based on object bounding box."""
        if not self.active:
            return

        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            print("Invalid bbox coordinates; skipping camera adjustment.")
            return

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if self.smoothed_width is None:
            self.smoothed_width = bbox_width
            self.smoothed_height = bbox_height
        else:
            self.smoothed_width = (
                self.smoothing_factor * bbox_width
                + (1 - self.smoothing_factor) * self.smoothed_width
            )
            self.smoothed_height = (
                self.smoothing_factor * bbox_height
                + (1 - self.smoothing_factor) * self.smoothed_height
            )

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
        """Safely invokes a camera command, handling exceptions."""
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
