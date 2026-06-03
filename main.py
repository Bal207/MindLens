import cv2
import time
from CameraDetection.pipeline import MindLensPipeline
from ScreenDetection.screen_analyzer import ScreenAnalyzer
from timeHandler import timeHandler


total_timer = timeHandler(0,0,0)
productive_timer = timeHandler(0,0,0)
distracted_timer = timeHandler(0,0,0)
neutral_timer = timeHandler(0,0,0)  

def get_unified_state(camera_state, screen_state):
    if camera_state == "Actively Using Phone":
        return "Distracted"
    elif camera_state == "Studying / Writing" or camera_state == "Reading":
        return "Productive"
    else:
        if screen_state == "Productive":
            return "Productive"
        elif screen_state == "Distracted":
            return "Distracted"
        else:
            return "Neutral"

def main():
    print("MindLens Tracker Starting...")
    print("Initializing Screen Analyzer models (this might take a moment)...")
    screen_analyzer = ScreenAnalyzer()
    camera_pipeline = MindLensPipeline()

    screen_analyzer.start()

    print("Initializing Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_time = time.time()

    print("MindLens Tracker Active! Press 'q' in the video window to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        camera_state, annotated_frame = camera_pipeline.get_state(frame)
        screen_state = screen_analyzer.get_state()

        unified_state = get_unified_state(camera_state, screen_state)

        cv2.putText(annotated_frame, f"Scr: {screen_state}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Overall: {unified_state}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("MindLens", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if unified_state == "Distracted":
            distracted_timer.increment_time(dt)
        elif unified_state == "Productive":
            productive_timer.increment_time(dt)
        else:
            neutral_timer.increment_time(dt)

        total_timer.increment_time(dt)

        time.sleep(0.01)

    print("\nExiting cleanly...")
    screen_analyzer.stop()
    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Session Summary ---")
    print(f"Distracted: {distracted_timer.get_time()}")
    print(f"Productive: {productive_timer.get_time()}")
    print(f"Neutral:    {neutral_timer.get_time()}")
    print(f"Total:      {total_timer.get_time()}")

if __name__ == "__main__":
    main()