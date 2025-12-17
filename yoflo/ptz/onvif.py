"""ONVIF-based PTZ camera controller for IP cameras."""

import time
import logging

try:
    from onvif import ONVIFCamera
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False
    ONVIFCamera = None


class ONVIFPTZController:
    """
    Controls PTZ camera movements via ONVIF protocol.
    Works with most IP cameras (Hikvision, Dahua, Axis, Hanwha, etc.).
    """

    def __init__(self, host, port=80, user="admin", password="", wsdl_dir=None):
        if not ONVIF_AVAILABLE:
            raise RuntimeError("onvif-zeep library unavailable. Install with: pip install onvif-zeep")

        self.camera = None
        self.ptz_service = None
        self.media_service = None
        self.profile_token = None
        self.move_speed = {"pan": 0.5, "tilt": 0.5, "zoom": 0.5}

        try:
            logging.info(f"Connecting to ONVIF camera at {host}:{port}...")
            if wsdl_dir:
                self.camera = ONVIFCamera(host, port, user, password, wsdl_dir)
            else:
                self.camera = ONVIFCamera(host, port, user, password)

            self.media_service = self.camera.create_media_service()
            profiles = self.media_service.GetProfiles()
            if not profiles:
                raise RuntimeError("No media profiles found on camera.")
            self.profile_token = profiles[0].token

            self.ptz_service = self.camera.create_ptz_service()
            logging.info(f"ONVIF PTZ connected. Profile: {self.profile_token}")

        except Exception as e:
            logging.error(f"Error connecting to ONVIF camera: {e}")
            self.camera = None

    def _continuous_move(self, pan=0, tilt=0, zoom=0, duration=0.3):
        """Perform a continuous move then stop."""
        if not self.ptz_service:
            logging.warning("ONVIF PTZ service not initialized.")
            return

        try:
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile_token
            request.Velocity = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }
            self.ptz_service.ContinuousMove(request)
            time.sleep(duration)
            self.stop()
        except Exception as e:
            logging.error(f"Error during ONVIF PTZ move: {e}")

    def stop(self):
        """Stop all PTZ movement."""
        if not self.ptz_service:
            return
        try:
            request = self.ptz_service.create_type('Stop')
            request.ProfileToken = self.profile_token
            request.PanTilt = True
            request.Zoom = True
            self.ptz_service.Stop(request)
        except Exception as e:
            logging.error(f"Error stopping ONVIF PTZ: {e}")

    def pan_right(self):
        self._continuous_move(pan=self.move_speed["pan"])

    def pan_left(self):
        self._continuous_move(pan=-self.move_speed["pan"])

    def tilt_up(self):
        self._continuous_move(tilt=self.move_speed["tilt"])

    def tilt_down(self):
        self._continuous_move(tilt=-self.move_speed["tilt"])

    def zoom_in(self):
        self._continuous_move(zoom=self.move_speed["zoom"])

    def zoom_out(self):
        self._continuous_move(zoom=-self.move_speed["zoom"])

    def go_to_preset(self, preset_token):
        """Move camera to a saved preset position."""
        if not self.ptz_service:
            logging.warning("ONVIF PTZ service not initialized.")
            return
        try:
            request = self.ptz_service.create_type('GotoPreset')
            request.ProfileToken = self.profile_token
            request.PresetToken = preset_token
            self.ptz_service.GotoPreset(request)
            logging.info(f"Moving to preset: {preset_token}")
        except Exception as e:
            logging.error(f"Error going to preset: {e}")

    def get_presets(self):
        """Return list of available preset positions."""
        if not self.ptz_service:
            return []
        try:
            presets = self.ptz_service.GetPresets({'ProfileToken': self.profile_token})
            return [(p.token, p.Name) for p in presets]
        except Exception as e:
            logging.error(f"Error getting presets: {e}")
            return []

    def set_move_speed(self, pan=None, tilt=None, zoom=None):
        """Set movement speed (0.0 to 1.0) for PTZ operations."""
        if pan is not None:
            self.move_speed["pan"] = max(0.0, min(1.0, pan))
        if tilt is not None:
            self.move_speed["tilt"] = max(0.0, min(1.0, tilt))
        if zoom is not None:
            self.move_speed["zoom"] = max(0.0, min(1.0, zoom))

    def get_stream_uri(self):
        """Return the RTSP stream URI for the camera."""
        if not self.media_service:
            return None
        try:
            request = self.media_service.create_type('GetStreamUri')
            request.ProfileToken = self.profile_token
            request.StreamSetup = {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            }
            response = self.media_service.GetStreamUri(request)
            return response.Uri
        except Exception as e:
            logging.error(f"Error getting stream URI: {e}")
            return None

    def close(self):
        """Close the ONVIF connection."""
        self.stop()
        self.camera = None
        self.ptz_service = None
        logging.info("ONVIF PTZ connection closed.")
