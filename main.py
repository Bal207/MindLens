import cv2
from CameraDetection.pipeline import MindLensPipeline
import time
from dataclasses import dataclass

def main():
    pipeline = MindLensPipeline()
    cap = cv2.VideoCapture(0)

    @dataclass
    class TimeTracker:
        distracted: float = 0.0
        studying: float = 0.0
        neutral: float = 0.0
        total: float = 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = TimeTracker()
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        state, annotated_frame = pipeline.get_state(frame)
        
        cv2.imshow("MindLens", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if state == "Actively Using Phone":
            tracker.distracted += dt
        elif state == "Studying / Writing" or state == "Reading":
            tracker.studying += dt
        elif state == "Idle":
            tracker.neutral += dt
            
        tracker.total += dt

    cap.release()
    cv2.destroyAllWindows()
    print("Distracted: ", tracker.distracted)
    print("Studying: ", tracker.studying)
    print("Neutral: ", tracker.neutral)
    print("Total: ", tracker.total)

if __name__ == "__main__":
    main()