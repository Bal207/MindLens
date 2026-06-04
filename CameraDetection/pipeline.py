import cv2
import threading
import time
from .object_detector import ObjectDetector
from .pose_estimator import PoseEstimator
from .state_analyzer import StateAnalyzer

class MindLensPipeline:
    def __init__(self):
        self.detector = ObjectDetector()
        self.pose_est = PoseEstimator()
        self.analyzer = StateAnalyzer()
        
        self.cached_state = "Idle"
        self.cached_objects = []
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.ai_thread = threading.Thread(target=self._process_ai_loop, daemon=True)
        self.ai_thread.start()

    def _process_ai_loop(self):
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                objects = self.detector.detect(frame_to_process)
                hand_coords, is_pinching, head_down, chin_y = self.pose_est.process(frame_to_process)
                state = self.analyzer.analyze(objects, hand_coords, is_pinching, head_down, chin_y)
                
                with self.lock:
                    self.cached_objects = objects
                    self.cached_state = state
                    
                time.sleep(0.01)
            else:
                time.sleep(0.05)

    def get_state(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            current_objects = self.cached_objects
            current_state = self.cached_state

        for obj in current_objects:
            x1, y1, x2, y2 = obj["bbox"]
            color = (0, 120, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{obj.get('name', obj['class'])} {obj['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(th + 4, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return current_state, frame

    def stop(self):
        self.running = False
        self.ai_thread.join()