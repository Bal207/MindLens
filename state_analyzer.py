class StateAnalyzer:
    def __init__(self):
        self.state_history = []

    def check_overlap(self, hand_coords, bbox):
        x1, y1, x2, y2 = bbox
        for hx, hy in hand_coords:
            if x1 <= hx <= x2 and y1 <= hy <= y2:
                return True
        return False

    def analyze(self, objects, hand_coords, is_pinching, head_down):
        current_state = "Idle"
        phone_box = None

        for obj in objects:
            if obj["class"] == 67:
                phone_box = obj["bbox"]

        in_desk_zone = False
        for hx, hy in hand_coords:
            if hy > 240:
                in_desk_zone = True
                break

        if phone_box and self.check_overlap(hand_coords, phone_box):
            current_state = "Actively Using Phone"
        elif head_down and is_pinching and in_desk_zone:
            current_state = "Studying / Writing"
        elif head_down and in_desk_zone:
            current_state = "Reading"

        self.state_history.append(current_state)
        if len(self.state_history) > 15:
            self.state_history.pop(0)

        return max(set(self.state_history), key=self.state_history.count)