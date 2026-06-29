import sys
import threading
import time

# Target rate for capturing/encoding the preview frame. The heavy AI inference
# is throttled separately inside the camera pipeline; this only governs how
# often we JPEG-encode a frame for the live feed, which is pure overhead above
# ~15 fps for this use case.
DISPLAY_INTERVAL = 1.0 / 15.0


def get_unified_state(camera_state, screen_state):
    if camera_state == "Actively Using Phone":
        return "Distracted"
    if camera_state in ("Studying / Writing", "Reading"):
        return "Productive"
    if screen_state == "Productive":
        return "Productive"
    if screen_state == "Distracted":
        return "Distracted"
    return "Neutral"


class _ModelCache:
    """Loads every heavy model once, in the background, at process startup.

    The whole point is that by the time the user clicks "Start Session" all of
    cv2, the camera pipeline (YOLO + MediaPipe) and the screen analyzer (easyocr
    + the zero-shot classifier) are already in memory, so the session starts in
    well under a second instead of stalling ~10s while easyocr loads on the
    click. The camera and screen models are independent, so they load on two
    threads concurrently and we warm up a dummy inference on each to absorb the
    one-time first-call cost.
    """

    def __init__(self):
        self._ready = threading.Event()
        self._camera_pipeline = None
        self._screen_analyzer = None
        self._cv2 = None
        self._error = None
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def _load(self):
        try:
            print("[Preload] Loading ML models in background...")
            t0 = time.time()

            import cv2
            self._cv2 = cv2

            errors = []

            def load_camera():
                try:
                    from CameraDetection.pipeline import MindLensPipeline
                    pipe = MindLensPipeline()
                    pipe.warmup()
                    self._camera_pipeline = pipe
                except Exception as e:  # noqa: BLE001 - surfaced via _error
                    errors.append(e)

            def load_screen():
                try:
                    from ScreenDetection.screen_analyzer import ScreenAnalyzer
                    analyzer = ScreenAnalyzer()
                    analyzer.warmup()
                    self._screen_analyzer = analyzer
                except Exception as e:  # noqa: BLE001 - surfaced via _error
                    errors.append(e)

            cam_thread = threading.Thread(target=load_camera, daemon=True)
            scr_thread = threading.Thread(target=load_screen, daemon=True)
            cam_thread.start()
            scr_thread.start()
            cam_thread.join()
            scr_thread.join()

            if errors:
                raise errors[0]

            elapsed = time.time() - t0
            print(f"[Preload] All models ready in {elapsed:.1f}s")
        except Exception as e:
            self._error = e
            print(f"[Preload] Failed to load models: {e}")
        finally:
            self._ready.set()

    def is_ready(self):
        return self._ready.is_set() and self._error is None

    def wait_and_get(self):
        self._ready.wait()
        if self._error:
            raise self._error
        return self._cv2, self._camera_pipeline, self._screen_analyzer


_model_cache = _ModelCache()


def _enhance_frame(cv2, frame):
    """Fast contrast enhancement. Runs in ~1-2ms per frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


class CameraStream:
    """Runs the camera hardware on a background thread to prevent buffer backups."""
    def __init__(self, cv2_module, src=0):
        self.cv2 = cv2_module
        self.cap = self._open_capture(src)
        if self.cap.isOpened():
            self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # A small buffer keeps the latest frame fresh instead of letting a
            # backlog build up; supported on most backends, harmless elsewhere.
            try:
                self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

        # Start the background reader
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _open_capture(self, src):
        """Open the webcam using the fastest backend for the current OS.

        On Windows the default MSMF backend can take several seconds to open a
        camera; DirectShow (CAP_DSHOW) is dramatically faster. macOS/Linux use
        the platform default (AVFoundation / V4L2), which is already quick.
        """
        if sys.platform.startswith("win"):
            cap = self.cv2.VideoCapture(src, self.cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return self.cv2.VideoCapture(src)

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    if ret:
                        self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            # Return a copy to prevent thread collisions while processing
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()


class MindLensTracker:
    def __init__(self, state):
        self.state = state
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def _run_loop(self):
        cap_stream = None
        screen_analyzer = None

        try:
            with self.state.lock:
                self.state.camera_state = "Initializing"
                self.state.screen_state = "Initializing"
                self.state.unified_state = "Neutral"

            cv2, camera_pipeline, screen_analyzer = _model_cache.wait_and_get()

            # The analyzer instance is preloaded once and reused across sessions;
            # (re)bind it to this session's live label/enable providers and start
            # its background capture thread.
            screen_analyzer.label_provider = self._screen_labels
            screen_analyzer.enabled_provider = self._screen_enabled
            screen_analyzer.start()
            last_time = time.time()

            while self._is_running():
                loop_start = time.time()
                dt = loop_start - last_time
                last_time = loop_start

                with self.state.lock:
                    camera_enabled = self.state.camera_enabled
                    screen_enabled = self.state.screen_enabled

                camera_state = "Idle"
                annotated_frame = None

                if camera_enabled:
                    if cap_stream is None:
                        with self.state.lock:
                            self.state.camera_state = "Initializing"
                        cap_stream = CameraStream(cv2, 0)

                    if cap_stream is not None and cap_stream.isOpened():
                        # This now returns instantly without waiting for hardware
                        ret, frame = cap_stream.read()

                        if ret and frame is not None:
                            # Detect on a contrast-enhanced copy (better accuracy),
                            # but show the user the natural frame so the preview
                            # doesn't look harshly over-processed.
                            enhanced_frame = _enhance_frame(cv2, frame)
                            camera_state, annotated_frame = camera_pipeline.get_state(
                                frame, detect_frame=enhanced_frame)
                        else:
                            camera_state = "Camera Unavailable"
                else:
                    if cap_stream is not None:
                        cap_stream.release()
                        cap_stream = None
                    camera_state = "Camera Off"

                screen_state = screen_analyzer.get_state() if screen_enabled else "Screen Off"
                detected_state = get_unified_state(camera_state, screen_state)

                if annotated_frame is not None:
                    ok, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        self.state.update_frame(buffer.tobytes())
                else:
                    self.state.update_frame(None)

                with self.state.lock:
                    self.state.camera_state = camera_state
                    self.state.screen_state = screen_state
                    # A manual override (if active) wins over the detected state
                    # for both the displayed status and the timer that ticks.
                    unified_state = self.state.effective_state(detected_state)
                    self.state.unified_state = unified_state

                    if unified_state == "Distracted":
                        self.state.distracted_timer.increment_time(dt)
                    elif unified_state == "Productive":
                        self.state.productive_timer.increment_time(dt)
                    else:
                        self.state.neutral_timer.increment_time(dt)
                    self.state.total_timer.increment_time(dt)

                # Consistent frame pacing (accounting for time already spent).
                spent = time.time() - loop_start
                time.sleep(max(0.0, DISPLAY_INTERVAL - spent))

        except Exception as exc:
            print(f"[Tracker] Error: {exc}")
            with self.state.lock:
                self.state.camera_state = "Error"
                self.state.screen_state = "Error"
                self.state.unified_state = "Neutral"
                self.state.is_running = False
        finally:
            if screen_analyzer is not None:
                screen_analyzer.stop()
            if cap_stream is not None:
                cap_stream.release()

    def _is_running(self):
        with self.state.lock:
            return self.state.is_running

    def _screen_labels(self):
        with self.state.lock:
            return list(self.state.custom_productive), list(self.state.custom_distracted)

    def _screen_enabled(self):
        with self.state.lock:
            return self.state.screen_enabled