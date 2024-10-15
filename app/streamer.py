from time import sleep
from dataclasses import dataclass
from typing import Any
from .logger import Logger
import cv2
import sys

@dataclass
class Streamer:
    width: int
    height: int
    max_retries: int  # Max retries for reinitializing the video source
    source: Any
    logger : Logger = Any


    def initialize(self):
        """
        Initializes the video source. If the source is invalid,
        logs an error and exits the program.
        """
        try:
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                sys.exit(1)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.logger.info(f"Video source initialized: {self.source} with resolution {self.width}x{self.height}")
        
        except Exception as e:
            self.logger.error(f"Error initializing video source: {e}")
            sys.exit(1)


    def reinitialize(self):
        """
        Attempts to reinitialize the video source by releasing the current source and initializing a new one.
        """
        self.release()
        self.logger.info("Reinitializing video source...")
        sleep(0.2)
        self.initialize()


    def read_frame(self):
        """
        Reads frames from the video source.

        Continuously attempts to read frames, and if a failure occurs, retries up to max_retries times.
        If retries are exhausted, reinitializes the video source.
        
        Yields:
            numpy.ndarray: The next frame from the video source.
        """
        retries = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("End of video stream or failed to read frame.")
                retries += 1
                if retries >= self.max_retries:
                    self.logger.error("Max retries reached. Attempting to reinitialize...")
                    self.reinitialize()
                    retries = 0
                continue
            
            retries = 0
            yield frame

    def release(self):
        """
        Releases the video source.

        Stops the video capture and logs the release action.
        """
        if self.cap:
            self.cap.release()
            self.logger.info("Video source released.")


if __name__ == "__main__":
    streamer = Streamer(webcam=True, source=0, width=1280, height=720)
    streamer.initialize()
    
    for frame in streamer.read_frame():
        cv2.imshow("Webcam Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    streamer.release()
    cv2.destroyAllWindows()




