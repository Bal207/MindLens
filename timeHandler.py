import dataclasses

class timeHandler:
    def __init__(self, seconds = 0, minutes = 0, hours = 0):
        self.seconds = seconds
        self.minutes = minutes
        self.hours = hours

    def increment_time(self, dt):
        self.seconds += dt
        if self.seconds >= 60:
            extra_minutes = int(self.seconds // 60)
            self.seconds = self.seconds % 60
            self.minutes += extra_minutes
            if self.minutes >= 60:
                extra_hours = self.minutes // 60
                self.minutes = self.minutes % 60
                self.hours += extra_hours
    
    def get_time(self):
        return self.seconds, self.minutes, self.hours

    def __str__(self):
        return f"{self.hours}:{self.minutes}:{self.seconds}"

        
