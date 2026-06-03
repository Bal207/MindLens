import threading
import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeHandler import timeHandler


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_FILE = os.path.join(DATA_DIR, "session_history.json")


class AppState:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.session_started_at = None
        self.camera_enabled = True
        self.screen_enabled = True
        self.camera_state = "Idle"
        self.screen_state = "Neutral"
        self.unified_state = "Neutral"
        self.productive_timer = timeHandler()
        self.distracted_timer = timeHandler()
        self.neutral_timer = timeHandler()
        self.total_timer = timeHandler()
        self.custom_productive = []
        self.custom_distracted = []
        self.latest_frame = None
        self.session_history = self._load_history()

    def _load_history(self):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_history(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.session_history[-50:], f, indent=2)

    @staticmethod
    def _timer_dict(timer):
        seconds, minutes, hours = timer.get_time()
        return {"h": hours, "m": minutes, "s": int(seconds)}

    @staticmethod
    def _timer_seconds(timer):
        seconds, minutes, hours = timer.get_time()
        return int(hours * 3600 + minutes * 60 + seconds)

    def reset_timers(self):
        with self.lock:
            self.productive_timer = timeHandler()
            self.distracted_timer = timeHandler()
            self.neutral_timer = timeHandler()
            self.total_timer = timeHandler()
            self.session_started_at = datetime.now(timezone.utc).isoformat()

    def stop_session(self):
        with self.lock:
            self.is_running = False
            total_seconds = self._timer_seconds(self.total_timer)
            if total_seconds <= 0:
                return None

            entry = {
                "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
                "started_at": self.session_started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "unified_state": self.unified_state,
                "timers": {
                    "productive": self._timer_dict(self.productive_timer),
                    "distracted": self._timer_dict(self.distracted_timer),
                    "neutral": self._timer_dict(self.neutral_timer),
                    "total": self._timer_dict(self.total_timer),
                },
                "seconds": {
                    "productive": self._timer_seconds(self.productive_timer),
                    "distracted": self._timer_seconds(self.distracted_timer),
                    "neutral": self._timer_seconds(self.neutral_timer),
                    "total": total_seconds,
                },
            }
            self.session_history.append(entry)
            self.session_history = self.session_history[-50:]
            self._save_history()
            return entry

    def update_frame(self, frame_bytes):
        with self.lock:
            self.latest_frame = frame_bytes

    def get_history(self):
        with self.lock:
            return list(reversed(self.session_history))

    def get_status_dict(self):
        with self.lock:
            return {
                "is_running": self.is_running,
                "session_started_at": self.session_started_at,
                "camera_enabled": self.camera_enabled,
                "screen_enabled": self.screen_enabled,
                "camera_state": self.camera_state,
                "screen_state": self.screen_state,
                "unified_state": self.unified_state,
                "timers": {
                    "productive": self._timer_dict(self.productive_timer),
                    "distracted": self._timer_dict(self.distracted_timer),
                    "neutral": self._timer_dict(self.neutral_timer),
                    "total": self._timer_dict(self.total_timer),
                },
                "history_count": len(self.session_history),
                "custom_productive": self.custom_productive,
                "custom_distracted": self.custom_distracted,
            }
