from dataclasses import dataclass, field
import time

@dataclass
class Fps:
    """
    A class to calculate frames per second (FPS) for a video stream.

    Attributes:
        start_timer (float): The timestamp when the timer started.
        frame_id (int): The number of frames processed.
    """

    start_timer: float = field(default=time.time(), init=False)
    frame_id: int = field(default=0, init=False)

    def begin_timer(self):
        """
        Start or restart the timer for FPS calculation.

        This method sets the start_timer attribute to the current time.
        """
        self.start_timer = time.time()

    def count_frame(self):
        """
        Increment the frame count.

        This method increments the frame_id attribute by 1 each time it is called.
        """
        self.frame_id += 1

    def calculate_fps(self):
        """
        Calculate the current FPS based on the time elapsed since the last timer reset.

        Returns:
            float: The calculated frames per second (FPS).
        """
        fps = 1.0 / (time.time() - self.start_timer)
        return fps