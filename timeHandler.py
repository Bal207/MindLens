class timeHandler:
    """Accumulates elapsed time.

    Internally stores a single float `total_seconds` so there is no chance of
    minute/hour rollover drift. Public API (get_time / increment_time / __str__)
    is unchanged so existing callers keep working.
    """

    def __init__(self, seconds=0, minutes=0, hours=0):
        self.total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    def increment_time(self, dt):
        if dt and dt > 0:
            self.total_seconds += dt

    def reset(self):
        self.total_seconds = 0.0

    def get_time(self):
        total = int(self.total_seconds)
        seconds = total % 60
        minutes = (total // 60) % 60
        hours = total // 3600
        return seconds, minutes, hours

    # Backwards-compatible attribute access
    @property
    def seconds(self):
        return int(self.total_seconds) % 60

    @property
    def minutes(self):
        return (int(self.total_seconds) // 60) % 60

    @property
    def hours(self):
        return int(self.total_seconds) // 3600

    def __str__(self):
        s, m, h = self.get_time()
        return f"{h:02d}:{m:02d}:{s:02d}"