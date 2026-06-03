class StateAnalyzer:
    def __init__(self):
        self.confirmed_state = "Idle"
        self.candidate_state = "Idle"
        self.consecutive_frames = 0
        self.frame_threshold = 2 

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

        for obj in objects:
            if obj["class"] == 67 and obj["conf"] > 0.5:
                phone_box = obj["bbox"]
            elif obj["class"] == 73:
                book_box = obj["bbox"]
            elif obj["class"] == 63:
                laptop_box = obj["bbox"]

        in_desk_zone = False
        for hx, hy in hand_coords:
            if hy > chin_y:
                in_desk_zone = True
                break

        if phone_box and self.check_overlap(hand_coords, phone_box) and not is_pinching:
            raw_state = "Actively Using Phone"
        elif is_pinching and in_desk_zone:
            raw_state = "Studying / Writing"
        elif laptop_box and self.check_overlap(hand_coords, laptop_box):
            raw_state = "Studying / Writing"
        elif book_box and self.check_overlap(hand_coords, book_box):
            raw_state = "Reading"
        elif head_down and in_desk_zone:
            raw_state = "Reading"

        if raw_state == self.candidate_state:
            self.consecutive_frames += 1
        else:
            self.candidate_state = raw_state
            self.consecutive_frames = 1

        if self.consecutive_frames >= self.frame_threshold:
            self.confirmed_state = self.candidate_state

        return self.confirmed_state