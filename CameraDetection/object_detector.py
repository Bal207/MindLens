import torch
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path="yolo26n.pt", conf_threshold=0.3):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.model = YOLO(model_path).to(self.device)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model.predict(source=frame, stream=False, verbose=False, classes=[63, 67, 73], conf=self.conf_threshold)
        boxes_data = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = self.model.names.get(cls_id, str(cls_id))
                boxes_data.append({"bbox": (x1, y1, x2, y2), "class": cls_id, "name": class_name, "conf": conf_val})
        return boxes_data
