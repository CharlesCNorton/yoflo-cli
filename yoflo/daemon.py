#!/usr/bin/env python3
"""
YOFLO Daemon - Keeps model loaded in memory for fast repeated queries.
Cross-platform compatible (Windows, Linux, Mac).

Usage:
    Start daemon:  python -m yoflo.daemon start [--model large] [--dir /path/to/workdir]
    Stop daemon:   python -m yoflo.daemon stop [--dir /path/to/workdir]
    Query:         python -m yoflo.daemon query --image <path_or_url> [--question "..."]
    Status:        python -m yoflo.daemon status [--dir /path/to/workdir]
"""

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path


def get_work_dir(custom_dir=None):
    if custom_dir:
        work_dir = Path(custom_dir)
    else:
        work_dir = Path(tempfile.gettempdir()) / "yoflo_daemon"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def get_paths(work_dir):
    return {
        "pid": work_dir / "daemon.pid",
        "query": work_dir / "query.json",
        "response": work_dir / "response.json",
        "ready": work_dir / "ready",
        "shutdown": work_dir / "shutdown",
    }


def cleanup_files(paths):
    for key in ["query", "response", "ready", "shutdown"]:
        if paths[key].exists():
            paths[key].unlink()


def is_daemon_running(paths):
    if not paths["pid"].exists():
        return False
    try:
        pid = int(paths["pid"].read_text().strip())
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (ProcessLookupError, ValueError, OSError):
        return False


def start_daemon(model_size="large", work_dir=None):
    work_dir = get_work_dir(work_dir)
    paths = get_paths(work_dir)

    if is_daemon_running(paths):
        print(f"Daemon already running (PID: {paths['pid'].read_text().strip()})")
        return

    cleanup_files(paths)

    pid = os.getpid()
    paths["pid"].write_text(str(pid))

    print(f"YOFLO Daemon starting (PID: {pid})")
    print(f"Work directory: {work_dir}")
    print(f"Model: Florence-2-{model_size}-ft")

    from yoflo.yoflo import YOFLO, get_youtube_live_url
    import cv2
    from PIL import Image

    yf = YOFLO()
    repo_id = f"microsoft/Florence-2-{model_size}-ft"
    print(f"Loading model: {repo_id}")
    yf.model_manager.download_and_load_model(repo_id)
    print("Model loaded successfully!")

    paths["ready"].write_text("ready")
    print("Daemon ready and waiting for queries...")
    print(f"To stop: python -m yoflo.daemon stop --dir \"{work_dir}\"")

    def handle_shutdown(signum=None, frame=None):
        print("\nShutting down daemon...")
        cleanup_files(paths)
        if paths["pid"].exists():
            paths["pid"].unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, handle_shutdown)

    while True:
        if paths["shutdown"].exists():
            handle_shutdown()

        if paths["query"].exists():
            try:
                query = json.loads(paths["query"].read_text())
                paths["query"].unlink()

                image_source = query.get("image")
                question = query.get("question")
                action = query.get("action", "detect")

                image_pil = None
                if image_source:
                    if image_source.startswith(("http://", "https://")):
                        if "youtube.com" in image_source or "youtu.be" in image_source:
                            stream_url = get_youtube_live_url(image_source)
                        else:
                            stream_url = image_source
                        cap = cv2.VideoCapture(stream_url)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            raise ValueError(f"Failed to capture frame from {image_source}")
                    else:
                        image_pil = Image.open(image_source)

                result = {"status": "ok"}

                if action == "detect" and image_pil:
                    result["detections"] = yf.run_object_detection(image_pil)
                elif action == "ask" and image_pil and question:
                    raw = yf.run_expression_comprehension(image_pil, question)
                    clean = raw.replace("</s>", "").replace("<s>", "").strip().lower()
                    result["answer"] = clean
                    result["raw"] = raw
                elif action == "ping":
                    result["message"] = "pong"
                else:
                    result = {"status": "error", "message": "Invalid action or missing parameters"}

                paths["response"].write_text(json.dumps(result, indent=2))

            except Exception as e:
                error_result = {"status": "error", "message": str(e)}
                paths["response"].write_text(json.dumps(error_result, indent=2))

        time.sleep(0.05)


def stop_daemon(work_dir=None):
    work_dir = get_work_dir(work_dir)
    paths = get_paths(work_dir)

    if not is_daemon_running(paths):
        print("Daemon is not running")
        cleanup_files(paths)
        if paths["pid"].exists():
            paths["pid"].unlink()
        return

    paths["shutdown"].write_text("shutdown")
    print("Shutdown signal sent...")

    for _ in range(50):
        if not is_daemon_running(paths):
            print("Daemon stopped")
            return
        time.sleep(0.1)

    print("Daemon did not stop gracefully, attempting force kill...")
    try:
        pid = int(paths["pid"].read_text().strip())
        if sys.platform == "win32":
            os.system(f"taskkill /F /PID {pid}")
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"Error killing process: {e}")

    cleanup_files(paths)
    if paths["pid"].exists():
        paths["pid"].unlink()


def query_daemon(image=None, question=None, action="detect", work_dir=None, timeout=60):
    work_dir = get_work_dir(work_dir)
    paths = get_paths(work_dir)

    if not is_daemon_running(paths):
        return {"status": "error", "message": "Daemon is not running. Start it with: python -m yoflo.daemon start"}

    if not paths["ready"].exists():
        return {"status": "error", "message": "Daemon is not ready yet"}

    if paths["response"].exists():
        paths["response"].unlink()

    query = {"action": action}
    if image:
        query["image"] = image
    if question:
        query["question"] = question

    paths["query"].write_text(json.dumps(query))

    start_time = time.time()
    while time.time() - start_time < timeout:
        if paths["response"].exists():
            result = json.loads(paths["response"].read_text())
            paths["response"].unlink()
            return result
        time.sleep(0.05)

    return {"status": "error", "message": "Query timed out"}


def status_daemon(work_dir=None):
    work_dir = get_work_dir(work_dir)
    paths = get_paths(work_dir)

    running = is_daemon_running(paths)
    ready = paths["ready"].exists()

    print(f"Work directory: {work_dir}")
    print(f"Running: {running}")
    if running:
        print(f"PID: {paths['pid'].read_text().strip()}")
    print(f"Ready: {ready}")

    return {"running": running, "ready": ready}


def main():
    parser = argparse.ArgumentParser(description="YOFLO Daemon")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    start_parser = subparsers.add_parser("start", help="Start the daemon")
    start_parser.add_argument("--model", choices=["base", "large"], default="large", help="Model size")
    start_parser.add_argument("--dir", type=str, help="Working directory")

    stop_parser = subparsers.add_parser("stop", help="Stop the daemon")
    stop_parser.add_argument("--dir", type=str, help="Working directory")

    query_parser = subparsers.add_parser("query", help="Query the daemon")
    query_parser.add_argument("--image", type=str, help="Image path or URL")
    query_parser.add_argument("--question", type=str, help="Question for expression comprehension")
    query_parser.add_argument("--action", choices=["detect", "ask", "ping"], default="detect")
    query_parser.add_argument("--dir", type=str, help="Working directory")
    query_parser.add_argument("--timeout", type=int, default=60, help="Query timeout in seconds")

    status_parser = subparsers.add_parser("status", help="Check daemon status")
    status_parser.add_argument("--dir", type=str, help="Working directory")

    args = parser.parse_args()

    if args.command == "start":
        start_daemon(model_size=args.model, work_dir=args.dir)
    elif args.command == "stop":
        stop_daemon(work_dir=args.dir)
    elif args.command == "query":
        result = query_daemon(
            image=args.image,
            question=args.question,
            action=args.action,
            work_dir=args.dir,
            timeout=args.timeout
        )
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        status_daemon(work_dir=args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
