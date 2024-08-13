from time import sleep
from dataclasses import dataclass
from typing import Any
from .logger import Logger
import cv2

@dataclass
class Streamer:
    webcam: bool
    source: Any
    width: int
    height: int
    max_retries: int  # Max retries for reinitializing the video source
    logger : Logger = Any

    def __post_init__(self):
        try:
            self.logger.info('streamer application started')
        except Exception as e:
            self.logger.error("streamer application failed")
    def initialize(self):
        """
        Initializes the video source. If the source is invalid, logs an error and exits the program.
        """
        try:
            if self.webcam:
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
        Attempts to reinitialize the video source.
        """
        self.release()
        self.logger.info("Reinitializing video source...")
        sleep(0.2)
        self.initialize()


    def read_frame(self):
        """
        Reads a frame from the video source. Maintains the loop if the video has not ended.
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




