"""Command-line interface for YOFLO."""

import os
import sys
import time
import logging
import argparse
import threading

from .core import YOFLO
from .ptz import PTZTracker, HID_AVAILABLE, ONVIF_AVAILABLE
from .ptz.hid import HIDPTZController
from .ptz.onvif import ONVIFPTZController
from .utils import setup_logging, get_youtube_stream_url

# Windows keyboard input for PTZ manual control
if sys.platform == "win32":
    import msvcrt
    MSVCRT_AVAILABLE = True
else:
    MSVCRT_AVAILABLE = False


def ptz_keyboard_control(ptz_camera):
    """Thread for interactive PTZ control using keyboard (Windows only)."""
    if not MSVCRT_AVAILABLE:
        logging.error("Interactive PTZ control not available on this platform.")
        return

    print("PTZ control active. Arrow keys=pan/tilt, +/-=zoom, q=quit.")
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
            print("Exiting PTZ control.")
            break
    ptz_camera.close()


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="YOFLO: Object detection and visual Q&A using Florence-2."
    )

    parser.add_argument("-od", nargs="*", help="Enable object detection (optional class filter).")
    parser.add_argument("-ph", type=str, help="Yes/no question for expression comprehension.")
    parser.add_argument("-hl", action="store_true", help="Headless mode (no display).")
    parser.add_argument("-ss", action="store_true", help="Screenshot on detection.")
    parser.add_argument("-lf", action="store_true", help="Log alerts to file.")
    parser.add_argument("-ir", action="store_true", help="Display inference rate.")
    parser.add_argument("-pp", action="store_true", help="Pretty print output.")
    parser.add_argument("-il", type=float, help="Inference rate limit (inferences/sec).")
    parser.add_argument("-ic", nargs="+", help="Inference chain with multiple phrases.")
    parser.add_argument("-wi", nargs="+", type=int, help="Webcam indices to use.")
    parser.add_argument("-rtsp", nargs="+", type=str, help="RTSP stream URLs.")
    parser.add_argument("-r", choices=["od", "infy", "infn"], help="Recording trigger mode.")
    parser.add_argument("-4bit", action="store_true", help="Enable 4-bit quantization.")
    parser.add_argument("-yt", type=str, help="YouTube live stream URL.")

    parser.add_argument("-ptz", nargs='?', const='0', help="Enable HID PTZ control.")
    parser.add_argument("-onvif", type=str, metavar="HOST", help="ONVIF camera IP/hostname.")
    parser.add_argument("-onvif-port", type=int, default=80, help="ONVIF port (default: 80).")
    parser.add_argument("-onvif-user", type=str, default="admin", help="ONVIF username.")
    parser.add_argument("-onvif-pass", type=str, default="", help="ONVIF password.")
    parser.add_argument("-to", "--track-object", type=str, help="Object class to track.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-mp", type=str, help="Path to local model directory.")
    group.add_argument("-dm", action="store_true", help="Download model from HuggingFace.")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.lf)
    quantization = "4bit" if getattr(args, '4bit', False) else None

    webcam_indices = args.wi or [0]
    rtsp_urls = args.rtsp or []

    # Handle YouTube URL
    if args.yt:
        stream_url = get_youtube_stream_url(args.yt)
        if not stream_url:
            logging.error("Failed to get YouTube stream URL.")
            return
        rtsp_urls = [stream_url]

    try:
        # Create YOFLO instance
        if args.dm:
            yoflo = YOFLO(
                display_inference_rate=args.ir,
                pretty_print=args.pp,
                inference_limit=args.il,
                class_names=args.od,
                webcam_indices=webcam_indices,
                rtsp_urls=rtsp_urls,
                record=args.r,
                quantization=quantization,
                track_object_name=args.track_object
            )
            if not yoflo.download_model():
                return
        else:
            if not os.path.exists(args.mp):
                logging.error(f"Model path {args.mp} does not exist.")
                return
            if not os.path.isdir(args.mp):
                logging.error(f"Model path {args.mp} is not a directory.")
                return
            yoflo = YOFLO(
                model_path=args.mp,
                display_inference_rate=args.ir,
                pretty_print=args.pp,
                inference_limit=args.il,
                class_names=args.od,
                webcam_indices=webcam_indices,
                rtsp_urls=rtsp_urls,
                record=args.r,
                quantization=quantization,
                track_object_name=args.track_object
            )

        # Configure YOFLO
        if args.ph:
            yoflo.phrase = args.ph
        if args.ic:
            yoflo.set_inference_phrases(args.ic)

        yoflo.headless = args.hl
        yoflo.object_detection_active = args.od is not None
        yoflo.screenshot_active = args.ss
        yoflo.log_to_file_active = args.lf

        # Start detection
        yoflo.start_webcam_detection()

        # Setup PTZ
        ptz_thread = None
        ptz_camera = None

        if args.onvif:
            if not ONVIF_AVAILABLE:
                logging.error("ONVIF not available. Install with: pip install onvif-zeep")
            else:
                try:
                    ptz_camera = ONVIFPTZController(
                        host=args.onvif,
                        port=args.onvif_port,
                        user=args.onvif_user,
                        password=args.onvif_pass
                    )
                    if ptz_camera.camera:
                        if not rtsp_urls and not args.wi:
                            stream_uri = ptz_camera.get_stream_uri()
                            if stream_uri:
                                logging.info(f"Using ONVIF stream: {stream_uri}")
                                yoflo.rtsp_urls = [stream_uri]

                        if args.track_object:
                            tracker = PTZTracker(ptz_camera)
                            tracker.activate(True)
                            yoflo.ptz_tracker = tracker
                            logging.info(f"ONVIF tracking enabled for: {args.track_object}")
                except Exception as e:
                    logging.error(f"ONVIF error: {e}")

        elif args.ptz:
            if not HID_AVAILABLE:
                logging.error("HID not available. Install with: pip install hid")
            else:
                ptz_camera = HIDPTZController()
                if args.ptz.lower() == 'track':
                    tracker = PTZTracker(ptz_camera)
                    tracker.activate(True)
                    yoflo.ptz_tracker = tracker
                else:
                    ptz_thread = threading.Thread(target=ptz_keyboard_control, args=(ptz_camera,))
                    ptz_thread.start()

        # Main loop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            yoflo.stop_webcam_detection()
            if ptz_thread and ptz_thread.is_alive():
                print("Press 'q' to exit PTZ control.")
        finally:
            if ptz_thread and ptz_thread.is_alive():
                ptz_thread.join()

    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()
