import cv2
import threading
import time
import numpy as np
from .object_detector import ObjectDetector
from .pose_estimator import PoseEstimator
from .state_analyzer import StateAnalyzer


class MindLensPipeline:
    # The heavy models (YOLO + 2x MediaPipe) don't need to run every frame.
    # ~8 inferences/sec is more than enough for desk activity and keeps CPU/GPU
    # (and laptop battery) from being pinned at 100%.
    TARGET_AI_FPS = 8.0
    # If the AI thread hasn't produced fresh boxes in this long, stop drawing the
    # old ones so boxes don't linger after an object leaves the frame.
    BOX_STALE_SEC = 0.8

    def __init__(self):
        self.detector = ObjectDetector()
        self.pose_est = PoseEstimator()
        self.analyzer = StateAnalyzer()

        self.cached_state = "Idle"
        self.cached_objects = []
        self.cached_objects_ts = 0.0
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self._min_interval = 1.0 / self.TARGET_AI_FPS
        self.ai_thread = threading.Thread(target=self._process_ai_loop, daemon=True)
        self.ai_thread.start()

    def _process_ai_loop(self):
        while self.running:
            loop_start = time.time()
            frame_to_process = None

            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None  # consume so we don't re-process

            if frame_to_process is not None:
                objects = self.detector.detect(frame_to_process)
                hand_coords, is_pinching, head_down, chin_y = self.pose_est.process(frame_to_process)
                state = self.analyzer.analyze(objects, hand_coords, is_pinching, head_down, chin_y)

                with self.lock:
                    self.cached_objects = objects
                    self.cached_objects_ts = time.time()
                    self.cached_state = state

            # Throttle to the target rate (accounting for processing time).
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self._min_interval - elapsed) if frame_to_process is not None else 0.03)

    def warmup(self):
        """Run one dummy inference so the first real frame isn't slowed by the
        one-time graph build / weight upload cost of YOLO and MediaPipe."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            objects = self.detector.detect(dummy)
            hand_coords, is_pinching, head_down, chin_y = self.pose_est.process(dummy)
            self.analyzer.analyze(objects, hand_coords, is_pinching, head_down, chin_y)
        except Exception as e:
            print(f"[Pipeline] Warmup skipped: {e}")

    def get_state(self, frame, detect_frame=None):
        """Return (state, annotated_display_frame).

        `frame` is what the user sees and what boxes are drawn on; pass a
        separate `detect_frame` (e.g. a contrast-enhanced copy) to run the AI on
        without affecting the natural-looking preview. Both must share the same
        resolution so box coordinates line up.
        """
        now = time.time()
        with self.lock:
            self.latest_frame = (detect_frame if detect_frame is not None else frame).copy()
            current_state = self.cached_state
            fresh = (now - self.cached_objects_ts) <= self.BOX_STALE_SEC
            current_objects = self.cached_objects if fresh else []

        for obj in current_objects:
            x1, y1, x2, y2 = obj["bbox"]
            # Softer, semi-transparent accent box reads cleaner than a hard line.
            color = (255, 170, 80)  # BGR — a calm blue/cyan accent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f"{obj.get('name', obj['class'])} {obj['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv2.rectangle(frame, (x1, max(0, y1 - th - 9)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(th + 3, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        return current_state, frame

    def stop(self):
        self.running = False
        self.ai_thread.join(timeout=2)