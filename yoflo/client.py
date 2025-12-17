#!/usr/bin/env python3
"""
YOFLO Client - Simple interface to query the YOFLO daemon.

Usage:
    from yoflo.client import YofloClient

    client = YofloClient()

    # Object detection
    result = client.detect("path/to/image.jpg")
    result = client.detect("https://youtube.com/watch?v=...")

    # Yes/No questions
    answer = client.ask("path/to/image.jpg", "Do you see a car?")

    # Check if daemon is running
    if client.is_ready():
        ...
"""

from yoflo.daemon import query_daemon, status_daemon, get_work_dir, get_paths, is_daemon_running


class YofloClient:
    def __init__(self, work_dir=None):
        self.work_dir = work_dir

    def is_running(self):
        work_dir = get_work_dir(self.work_dir)
        paths = get_paths(work_dir)
        return is_daemon_running(paths)

    def is_ready(self):
        work_dir = get_work_dir(self.work_dir)
        paths = get_paths(work_dir)
        return is_daemon_running(paths) and paths["ready"].exists()

    def status(self):
        return status_daemon(self.work_dir)

    def ping(self, timeout=10):
        result = query_daemon(action="ping", work_dir=self.work_dir, timeout=timeout)
        return result.get("status") == "ok"

    def detect(self, image, timeout=60):
        result = query_daemon(image=image, action="detect", work_dir=self.work_dir, timeout=timeout)
        if result.get("status") == "ok":
            return result.get("detections")
        raise RuntimeError(result.get("message", "Unknown error"))

    def ask(self, image, question, timeout=60):
        result = query_daemon(
            image=image,
            question=question,
            action="ask",
            work_dir=self.work_dir,
            timeout=timeout
        )
        if result.get("status") == "ok":
            return result.get("answer")
        raise RuntimeError(result.get("message", "Unknown error"))

    def ask_raw(self, image, question, timeout=60):
        result = query_daemon(
            image=image,
            question=question,
            action="ask",
            work_dir=self.work_dir,
            timeout=timeout
        )
        if result.get("status") == "ok":
            return result
        raise RuntimeError(result.get("message", "Unknown error"))


def detect(image, work_dir=None, timeout=60):
    return YofloClient(work_dir).detect(image, timeout)


def ask(image, question, work_dir=None, timeout=60):
    return YofloClient(work_dir).ask(image, question, timeout)


def is_ready(work_dir=None):
    return YofloClient(work_dir).is_ready()
