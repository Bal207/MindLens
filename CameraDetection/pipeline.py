import cv2
from .object_detector import ObjectDetector
from .pose_estimator import PoseEstimator
from .state_analyzer import StateAnalyzer

class MindLensPipeline:
    def __init__(self):
        self.detector = ObjectDetector()
        self.pose_est = PoseEstimator()
        self.analyzer = StateAnalyzer()
        self.frame_counter = 0
        self.cached_state = "Idle"
        self.cached_objects = []
        self.detect_every_frames = 8

    def get_state(self, frame):
        self.frame_counter += 1

        if self.frame_counter == 1 or self.frame_counter % self.detect_every_frames == 0:
            self.cached_objects = self.detector.detect(frame)
            hand_coords, is_pinching, head_down, chin_y = self.pose_est.process(frame)
            self.cached_state = self.analyzer.analyze(self.cached_objects, hand_coords, is_pinching, head_down, chin_y)

        for obj in self.cached_objects:
            x1, y1, x2, y2 = obj["bbox"]
            color = (0, 120, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.get('name', obj['class'])} {obj['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(th + 4, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return self.cached_state, frame
