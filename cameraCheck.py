import cv2

"""
Checks if the camera is avilable/connected to the system.
"""
def checkCamera():
    try:
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("Camera is available")
        else:
            print("Camera is not available")
            return False
        
        ret, frame = cap.read()
        if ret:
            print(f"Successfully captured a frame of size: {frame.shape}")
            return True
        else:
            print("Camera opened, but failed to read a frame.")
            return False
        return False
    except:
        print("Unexpected Error occured.")
        return False
    cap.release()