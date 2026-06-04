import threading
import time

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
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._camera_pipeline = None
        self._screen_analyzer_cls = None
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

            from CameraDetection.pipeline import MindLensPipeline
            self._camera_pipeline = MindLensPipeline()

            from ScreenDetection.screen_analyzer import ScreenAnalyzer
            self._screen_analyzer_cls = ScreenAnalyzer

            elapsed = time.time() - t0
            print(f"[Preload] All models ready in {elapsed:.1f}s")
        except Exception as e:
            self._error = e
            print(f"[Preload] Failed to load models: {e}")
        finally:
            self._ready.set()

    def wait_and_get(self):
        self._ready.wait()
        if self._error:
            raise self._error
        return self._cv2, self._camera_pipeline, self._screen_analyzer_cls

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
        self.cap = self.cv2.VideoCapture(src)
        if self.cap.isOpened():
            self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        
        # Start the background reader
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

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

            cv2, camera_pipeline, ScreenAnalyzerCls = _model_cache.wait_and_get()

            screen_analyzer = ScreenAnalyzerCls(
                label_provider=self._screen_labels,
                enabled_provider=self._screen_enabled
            )
            screen_analyzer.start()
            last_time = time.time()

            while self._is_running():
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

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
                            enhanced_frame = _enhance_frame(cv2, frame)
                            camera_state, annotated_frame = camera_pipeline.get_state(enhanced_frame)
                        else:
                            camera_state = "Camera Unavailable"
                else:
                    if cap_stream is not None:
                        cap_stream.release()
                        cap_stream = None
                    camera_state = "Camera Off"

                screen_state = screen_analyzer.get_state() if screen_enabled else "Screen Off"
                unified_state = get_unified_state(camera_state, screen_state)

                if annotated_frame is not None:
                    ok, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        self.state.update_frame(buffer.tobytes())
                else:
                    self.state.update_frame(None)

                with self.state.lock:
                    self.state.camera_state = camera_state
                    self.state.screen_state = screen_state
                    self.state.unified_state = unified_state

                    if unified_state == "Distracted":
                        self.state.distracted_timer.increment_time(dt)
                    elif unified_state == "Productive":
                        self.state.productive_timer.increment_time(dt)
                    else:
                        self.state.neutral_timer.increment_time(dt)
                    self.state.total_timer.increment_time(dt)

                # Consistent frame pacing
                time.sleep(0.03)

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