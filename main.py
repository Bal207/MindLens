import cv2
from pipeline import MindLensPipeline
import time
from dataclasses import dataclass

def main():
    pipeline = MindLensPipeline()
    cap = cv2.VideoCapture(0)

    @dataclass
    class TimeTracker:
        distracted: int = 0
        studying: int = 0
        neutral: int = 0
        total: int = 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = TimeTracker()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        state, annotated_frame = pipeline.get_state(frame)
        
        cv2.imshow("MindLens", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if(state == "Actively Using Phone"):
            tracker.distracted += 0.05
        if(state == "Studying / Writing" or state == "Reading"):
            tracker.studying += 0.05
        if(state == "Idle"):
            tracker.neutral += 0.05
        tracker.total += 0.05
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    print("Distracted: ", tracker.distracted)
    print("Studying: ", tracker.studying)
    print("Neutral: ", tracker.neutral)
    print("Total: ", tracker.total)


if __name__ == "__main__":
    main()