class StateAnalyzer:
    def __init__(self):
        self.history = []
        self.history_limit = 4
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

        self.history.append(raw_state)
        if len(self.history) > self.history_limit:
            self.history.pop(0)

        self.last_debug = f"phone={phone_conf:.2f} area={phone_area} hand={bool(has_phone)}"

        if raw_state == "Actively Using Phone":
            recent_phone_count = self.history[-2:].count("Actively Using Phone")
            if recent_phone_count >= 1:
                return "Actively Using Phone"

        return max(set(self.history), key=self.history.count)
