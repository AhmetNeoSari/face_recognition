from dataclasses import dataclass,field
import time

@dataclass
class Fps:
    start_timer : float = field(default=time.time(), init=False)
    end_timer : float = field(default=time.time(),init=False)
    frame_id : int = field(default=0, init=False)

    def calculate_fps(self):
        fps = 1.0 / (self.end_timer-self.start_timer)
        return fps
    