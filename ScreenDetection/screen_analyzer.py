import pyautogui
import cv2
import numpy as np
import easyocr
import torch
import time
import threading
from transformers import pipeline

class ScreenAnalyzer:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.gpu_flag = True
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_flag = True
        else:
            self.device = "cpu"
            self.gpu_flag = False
            
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=self.device)
        
        self.MACRO_LABELS = ["software engineering or studying", "entertainment or social media", "neutral desktop background"]
        self.current_state = "Neutral"
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run_loop(self):
        print("[Screen] Analyzer background thread started...")
        while self.running:
            try:
                screenshot = pyautogui.screenshot()
                screenshot_array = np.array(screenshot)
                img = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2GRAY)
                height, width = img.shape
                max_width = 800
                if width > max_width:
                    scale = max_width / width
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                raw_results = self.reader.readtext(img, detail=1)
                
                text_list = [res[1] for res in raw_results if res[2] > 0.4]
                full_text = " ".join(text_list)
                words_to_strip = ["candidate_labels", "classifier", "entertainment", "netflix", "youtube", "social media"]
                cleaned_text = full_text
                for word in words_to_strip:
                    cleaned_text = cleaned_text.replace(word, "")
                
                
                if len(cleaned_text.strip()) < 10:
                    self.current_state = "Neutral"
                    time.sleep(10)
                    continue

         
                cleaned_text = cleaned_text[:1500]

                result = self.classifier(
                    cleaned_text, 
                    self.MACRO_LABELS, 
                    hypothesis_template="The content of this screen is related to {}."
                )
                
                top_label = result['labels'][0]
                
                if top_label == "software engineering or studying":
                    self.current_state = "Productive"
                elif top_label == "entertainment or social media":
                    self.current_state = "Distracted"
                else:
                    self.current_state = "Neutral"
                    
                time.sleep(10)
                
            except Exception as e:
                print(f"[Screen] Error encountered: {e}")
                time.sleep(10)

    def get_state(self):
        return self.current_state
