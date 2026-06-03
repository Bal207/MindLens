import cv2
from object_detector import ObjectDetector
from pose_estimator import PoseEstimator
from state_analyzer import StateAnalyzer

class MindLensPipeline:
    def __init__(self):
        self.detector = ObjectDetector()
        self.pose_est = PoseEstimator()
        self.analyzer = StateAnalyzer()
        self.frame_counter = 0
        self.cached_state = "Idle"
        self.cached_objects = []

    def get_state(self, frame):
        self.frame_counter += 1

        if self.frame_counter % 5 == 0:
            self.cached_objects = self.detector.detect(frame)
            hand_coords, is_pinching, head_down = self.pose_est.process(frame)
            self.cached_state = self.analyzer.analyze(self.cached_objects, hand_coords, is_pinching, head_down)

        for obj in self.cached_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        cv2.putText(frame, f"State: {self.cached_state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return self.cached_state, frame