import threading
import sys
import os
import json
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeHandler import timeHandler


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_FILE = os.path.join(DATA_DIR, "session_history.json")

VALID_OVERRIDES = {"Productive", "Distracted", "Neutral"}


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
        # Manual user override of the detected state.
        self.override_state = None
        self.override_until = 0.0
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
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.session_history[-50:], f, indent=2)
        # Atomic replace so a crash mid-write can't corrupt the history file.
        os.replace(tmp, HISTORY_FILE)

    @staticmethod
    def _timer_dict(timer):
        seconds, minutes, hours = timer.get_time()
        return {"h": hours, "m": minutes, "s": int(seconds)}

    @staticmethod
    def _timer_seconds(timer):
        seconds, minutes, hours = timer.get_time()
        return int(hours * 3600 + minutes * 60 + seconds)

    # ---- Override helpers (assume caller holds self.lock) ----
    def _override_active(self):
        return self.override_state is not None and time.time() < self.override_until

    def effective_state(self, detected_state):
        """The state that should drive timers/UI, honouring an active override."""
        if self._override_active():
            return self.override_state
        return detected_state

    def set_override(self, state, duration_sec=300):
        if state not in VALID_OVERRIDES:
            return False
        with self.lock:
            self.override_state = state
            self.override_until = time.time() + duration_sec
        return True

    def clear_override(self):
        with self.lock:
            self.override_state = None
            self.override_until = 0.0

    def reset_timers(self):
        with self.lock:
            self.productive_timer = timeHandler()
            self.distracted_timer = timeHandler()
            self.neutral_timer = timeHandler()
            self.total_timer = timeHandler()
            self.override_state = None
            self.override_until = 0.0
            self.session_started_at = datetime.now(timezone.utc).isoformat()

    def stop_session(self):
        with self.lock:
            self.is_running = False
            self.override_state = None
            self.override_until = 0.0
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
            override_active = self._override_active()
            override_remaining = max(0, int(self.override_until - time.time())) if override_active else 0
            return {
                "is_running": self.is_running,
                "session_started_at": self.session_started_at,
                "camera_enabled": self.camera_enabled,
                "screen_enabled": self.screen_enabled,
                "camera_state": self.camera_state,
                "screen_state": self.screen_state,
                "unified_state": self.unified_state,
                "override_state": self.override_state if override_active else None,
                "override_remaining": override_remaining,
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