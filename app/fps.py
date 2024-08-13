from dataclasses import dataclass,field
import time

@dataclass
class Fps:
    start_timer : float = field(default=time.time(), init=False)
    frame_id : int = field(default=0, init=False)

    def begin_timer(self):
        self.start_timer = time.time()

    def count_frame(self):
        self.frame_id += 1

    def calculate_fps(self):
        fps = 1.0 / ( time.time() - self.start_timer)
        return fps
    