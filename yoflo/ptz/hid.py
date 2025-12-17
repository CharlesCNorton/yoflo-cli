"""HID-based PTZ camera controller (Logitech PTZ Pro, etc.)."""

import time
import logging

try:
    import hid
    HID_AVAILABLE = True
except ImportError:
    HID_AVAILABLE = False
    hid = None


class HIDPTZController:
    """
    Controls PTZ camera movements via HID commands.
    Works with Logitech PTZ Pro and similar HID-compatible cameras.
    """

    VENDOR_ID = 0x046D
    PRODUCT_ID = 0x085F
    USAGE_PAGE = 65280
    USAGE = 1

    COMMANDS = {
        'tilt_up': (0x0B, 0x00),
        'tilt_down': (0x0B, 0x01),
        'pan_right': (0x0B, 0x02),
        'pan_left': (0x0B, 0x03),
        'zoom_in': (0x0B, 0x04),
        'zoom_out': (0x0B, 0x05),
    }

    def __init__(self, vendor_id=None, product_id=None, usage_page=None, usage=None):
        if not HID_AVAILABLE:
            raise RuntimeError("HID library unavailable. Install with: pip install hid")

        self.vendor_id = vendor_id or self.VENDOR_ID
        self.product_id = product_id or self.PRODUCT_ID
        self.usage_page = usage_page or self.USAGE_PAGE
        self.usage = usage or self.USAGE
        self.device = None

        self._connect()

    def _connect(self):
        """Attempt to connect to the HID PTZ device."""
        try:
            ptz_path = None
            for d in hid.enumerate(self.vendor_id, self.product_id):
                if d['usage_page'] == self.usage_page and d['usage'] == self.usage:
                    ptz_path = d['path']
                    break

            if ptz_path:
                self.device = hid.device()
                self.device.open_path(ptz_path)
                logging.info("HID PTZ interface opened successfully.")
            else:
                logging.warning("No suitable HID PTZ interface found.")
        except IOError as e:
            logging.error(f"Error opening HID PTZ device: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during HID PTZ initialization: {e}")

    def _send_command(self, report_id, value):
        """Send a command to the PTZ device via HID."""
        if not self.device:
            logging.warning("HID PTZ device not initialized.")
            return False

        command = [report_id & 0xFF, value] + [0x00] * 30
        try:
            self.device.write(command)
            time.sleep(0.2)
            return True
        except IOError as e:
            logging.error(f"Error sending HID PTZ command: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error sending HID PTZ command: {e}")
            return False

    def pan_right(self):
        report_id, value = self.COMMANDS['pan_right']
        self._send_command(report_id, value)

    def pan_left(self):
        report_id, value = self.COMMANDS['pan_left']
        self._send_command(report_id, value)

    def tilt_up(self):
        report_id, value = self.COMMANDS['tilt_up']
        self._send_command(report_id, value)

    def tilt_down(self):
        report_id, value = self.COMMANDS['tilt_down']
        self._send_command(report_id, value)

    def zoom_in(self):
        report_id, value = self.COMMANDS['zoom_in']
        self._send_command(report_id, value)

    def zoom_out(self):
        report_id, value = self.COMMANDS['zoom_out']
        self._send_command(report_id, value)

    def close(self):
        """Close the HID device handle."""
        if self.device:
            try:
                self.device.close()
                logging.info("HID PTZ device closed.")
            except Exception as e:
                logging.error(f"Error closing HID PTZ device: {e}")
            self.device = None


# Backwards compatibility alias
PTZController = HIDPTZController
