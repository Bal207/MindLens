import cv2
import numpy as np
import urllib.request
import os
import tempfile

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


def _get_model_path(filename: str, url: str) -> str:
    cache_dir = os.path.join(tempfile.gettempdir(), "mindlens_models")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        print(f"[PoseEstimator] Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"[PoseEstimator] Saved to {path}")
    return path


class PoseEstimator:
    _HAND_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    _FACE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )

    def __init__(self):
        hand_model_path = _get_model_path("hand_landmarker.task", self._HAND_MODEL_URL)
        face_model_path = _get_model_path("face_landmarker.task", self._FACE_MODEL_URL)

        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

        face_opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_results = self.hand_landmarker.detect(mp_image)
        face_results = self.face_landmarker.detect(mp_image)

        h, w, _ = frame.shape
        hand_coords = []
        is_pinching = False
        chin_y_px = h // 2

        for hand_landmarks in hand_results.hand_landmarks:
            for lm in hand_landmarks:
                hand_coords.append((int(lm.x * w), int(lm.y * h)))

            wrist   = hand_landmarks[0]
            knuckle = hand_landmarks[9]
            hand_scale = np.sqrt(
                (wrist.x - knuckle.x) ** 2 +
                (wrist.y - knuckle.y) ** 2 +
                (wrist.z - knuckle.z) ** 2
            )
            if hand_scale == 0:
                hand_scale = 1

            t = hand_landmarks[4]  
            i = hand_landmarks[8]  
            m = hand_landmarks[12]  

            dist_ti = np.sqrt((t.x-i.x)**2 + (t.y-i.y)**2 + (t.z-i.z)**2) / hand_scale
            dist_tm = np.sqrt((t.x-m.x)**2 + (t.y-m.y)**2 + (t.z-m.z)**2) / hand_scale

            if dist_ti < 0.33 or dist_tm < 0.33:
                is_pinching = True

        head_down = False
        if face_results.face_landmarks:
            face = face_results.face_landmarks[0]

            forehead = face[10]
            chin     = face[152]
            nose     = face[1]

            chin_y_px = int(chin.y * h)

            face_height = abs(forehead.y - chin.y)
            if face_height > 0:
                nose_to_chin_ratio = (chin.y - nose.y) / face_height
                if nose_to_chin_ratio < 0.36:
                    head_down = True

        return hand_coords, is_pinching, head_down, chin_y_px