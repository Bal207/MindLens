import time
from collections import deque


class StateAnalyzer:
    # Vote over a fixed *time* window instead of a fixed frame count, so the
    # amount of smoothing stays constant regardless of how fast the AI loop runs.
    HISTORY_WINDOW_SEC = 1.6
    # A confident phone detection only needs to persist this long to win, so
    # genuine phone pickups are caught quickly without single-frame flicker.
    PHONE_CONFIRM_SEC = 0.5

    def __init__(self):
        self.history = deque()  # (timestamp, state)
        self.last_debug = ""

    def check_overlap(self, hand_coords, bbox):
        x1, y1, x2, y2 = bbox
        for hx, hy in hand_coords:
            if x1 <= hx <= x2 and y1 <= hy <= y2:
                return True
        return False

    def box_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _prune(self, now):
        cutoff = now - self.HISTORY_WINDOW_SEC
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def analyze(self, objects, hand_coords, is_pinching, head_down, chin_y):
        raw_state = "Idle"
        phone_box = None
        book_box = None
        laptop_box = None

        phone_conf = 0.0
        book_conf = 0.0
        laptop_conf = 0.0

        for obj in objects:
            if obj["class"] == 67:
                if obj["conf"] > phone_conf:
                    phone_conf = obj["conf"]
                    phone_box = obj["bbox"]
            elif obj["class"] == 73:
                if obj["conf"] > book_conf:
                    book_conf = obj["conf"]
                    book_box = obj["bbox"]
            elif obj["class"] == 63:
                if obj["conf"] > laptop_conf:
                    laptop_conf = obj["conf"]
                    laptop_box = obj["bbox"]

        in_desk_zone = False
        for hx, hy in hand_coords:
            if hy > chin_y:
                in_desk_zone = True
                break

        has_phone = phone_box and self.check_overlap(hand_coords, phone_box)
        has_book = book_box and self.check_overlap(hand_coords, book_box)
        has_laptop = laptop_box and self.check_overlap(hand_coords, laptop_box)

        phone_area = self.box_area(phone_box) if phone_box else 0
        held_phone = has_phone and phone_conf >= 0.42
        visible_phone = phone_box and phone_conf >= 0.48 and phone_area >= 550
        high_conf_phone = phone_box and phone_conf >= 0.62
        conflicting_study_object = (has_laptop or has_book) and phone_conf < 0.72

        if (held_phone or visible_phone or high_conf_phone) and not conflicting_study_object:
            raw_state = "Actively Using Phone"
        elif is_pinching and in_desk_zone:
            raw_state = "Studying / Writing"
        elif has_laptop:
            raw_state = "Studying / Writing"
        elif has_book:
            raw_state = "Reading"
        elif head_down and in_desk_zone:
            raw_state = "Reading"

        now = time.time()
        self.history.append((now, raw_state))
        self._prune(now)

        self.last_debug = f"phone={phone_conf:.2f} area={phone_area} hand={bool(has_phone)}"

        # Phone use is the most actionable signal, so promote it quickly: if a
        # phone has been seen for at least PHONE_CONFIRM_SEC inside the window,
        # report it immediately instead of waiting for a majority vote.
        if raw_state == "Actively Using Phone":
            phone_span = [t for t, s in self.history if s == "Actively Using Phone"]
            if phone_span and (now - phone_span[0]) >= self.PHONE_CONFIRM_SEC:
                return "Actively Using Phone"
            if len(phone_span) >= 2:
                return "Actively Using Phone"

        # Otherwise return the majority state across the time window.
        states = [s for _, s in self.history]
        if not states:
            return raw_state
        return max(set(states), key=states.count)