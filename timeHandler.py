import dataclasses

class timeHandler:
    def __init__(self, seconds = 0, minutes = 0, hours = 0):
        self.seconds = seconds
        self.minutes = minutes
        self.hours = hours

    def increment_time(self, dt):
        self.seconds += dt
        if self.seconds >= 60:
            self.seconds = 0
            self.minutes += 1
            if self.minutes >= 60:
                self.minutes = 0
                self.hours += 1
    
    def get_time(self):
        return self.seconds, self.minutes, self.hours

    def __str__(self):
        return f"{self.hours}:{self.minutes}:{self.seconds}"

        