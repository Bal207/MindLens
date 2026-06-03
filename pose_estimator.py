import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)

        hand_coords = []
        is_pinching = False
        h, w, _ = frame.shape

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    hand_coords.append((int(lm.x * w), int(lm.y * h)))
                
                wrist = hand_landmarks.landmark[0]
                knuckle = hand_landmarks.landmark[9]
                hand_scale = np.sqrt((wrist.x - knuckle.x)**2 + (wrist.y - knuckle.y)**2)
                if hand_scale == 0: 
                    hand_scale = 1
                
                t = hand_landmarks.landmark[4]
                i = hand_landmarks.landmark[8]
                m = hand_landmarks.landmark[12]
                
                dist_ti = np.sqrt((t.x - i.x)**2 + (t.y - i.y)**2) / hand_scale
                dist_tm = np.sqrt((t.x - m.x)**2 + (t.y - m.y)**2) / hand_scale
                
                if dist_ti < 0.35 or dist_tm < 0.35:
                    is_pinching = True

        head_down = False
        if face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0]
            forehead = face.landmark[10]
            chin = face.landmark[152]
            nose = face.landmark[1]
            
            face_height = np.abs(forehead.y - chin.y)
            if face_height > 0:
                nose_to_chin_ratio = (chin.y - nose.y) / face_height
                if nose_to_chin_ratio < 0.36:
                    head_down = True

        return hand_coords, is_pinching, head_down