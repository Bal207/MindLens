class StateAnalyzer:
    def __init__(self):
        self.history = []
        self.history_limit = 10

    def check_overlap(self, hand_coords, bbox):
        x1, y1, x2, y2 = bbox
        for hx, hy in hand_coords:
            if x1 <= hx <= x2 and y1 <= hy <= y2:
                return True
        return False

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

        is_real_phone = has_phone and phone_conf >= 0.60 and not is_pinching
        if is_real_phone and not (has_laptop and phone_conf < 0.75) and not (has_book and phone_conf < 0.75):
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

        return max(set(self.history), key=self.history.count)