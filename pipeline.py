import cv2
from object_detector import ObjectDetector
from pose_estimator import PoseEstimator
from state_analyzer import StateAnalyzer

class MindLensPipeline:
    def __init__(self):
        self.detector = ObjectDetector()
        self.pose_est = PoseEstimator()
        self.analyzer = StateAnalyzer()

    def get_state(self, frame):
        resized_frame = cv2.resize(frame, (640, 480))
        objects = self.detector.detect(resized_frame)
        hand_coords, is_pinching, head_down = self.pose_est.process(resized_frame)
        state = self.analyzer.analyze(objects, hand_coords, is_pinching, head_down)
        
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        cv2.putText(resized_frame, f"State: {state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return state, resized_frame