from tensorflow.core import framework
from cameraCheck import checkCamera
from ultralytics import YOLO
import cv2
import os
import torch

if torch.backends.mps.is_available():
    device = "mps"      
elif torch.cuda.is_available():
    device = "cuda"   
else:
    device = "cpu"      

detectionModel = YOLO("yolo26n.pt").to(device)
classificationModel = YOLO("yolo26sls.pt").to(device)
cameraAvailable = checkCamera()
if cameraAvailable:
    detectionModel.predict(source = 0, show = True)
else:
    print("Camera not available. Exiting.")
    exit(1)

def startClassification():
    for detection in detectionModel:
        boxes = detection.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
            cropped_obj = cv2.frame[y1:y2, x1:x2]
            class_res = classificationModel.predict(source = cropped_obj, verbose = False)

            top_class_idx = class_res[0].probs.top1
            top_class_name = class_res[0].names[top_class_idx]
            confidence = class_res[0].probs.top1conf.item()

            cv2.rectangle(cv2.videoCapture(0), (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{top_class_name} ({confidence:.2f})"
            cv2.putText(cv2.videoCapture(0), label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Frame", cv2.videoCapture(0))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



startClassification()
cv2.videoCapture(0).release()
cv2.destroyAllWindows()
print("Pipeline closed successfully.")
