import cv2
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class LineSelector:
    frame: Optional = None  # The frame where the line will be selected
    line_start: Tuple[int, int] = field(default=None)  # Start point of the line
    line_end: Tuple[int, int] = field(default=None)  # End point of the line
    drawing: bool = field(default=False, init=False)  # Whether the user is drawing or not

    def select_line(self, frame):
        """
        Allows the user to select a line on the frame using the mouse.
        Once the selection is complete, it returns the line_start and line_end values.

        Args:
            frame (np.ndarray): The image on which the line will be drawn.

        Returns:
            tuple: line_start and line_end points of the selected line.
        """
        self.frame = frame.copy()  # Make a copy of the frame to draw the line on
        cv2.namedWindow("Select Line")
        cv2.setMouseCallback("Select Line", self._draw_line)

        # Loop until the user confirms the line selection by pressing 'q'
        while True:
            cv2.imshow("Select Line", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to exit and finalize the line selection
                break

        cv2.destroyWindow("Select Line")
        return self.line_start, self.line_end

    def _draw_line(self, event, x, y, flags, param):
        """
        Handles mouse events to draw the line interactively.
        
        Args:
            event: The type of mouse event (e.g., left button down, move, left button up).
            x, y: The current position of the mouse.
            flags: Any flags passed by OpenCV.
            param: Additional parameters (not used in this case).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.line_start = (x, y)  # Set the starting point of the line

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.frame.copy()  # Copy the frame to redraw the line during mouse movement
                cv2.line(img_copy, self.line_start, (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Line", img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.line_end = (x, y)  # Set the ending point of the line
            cv2.line(self.frame, self.line_start, self.line_end, (0, 255, 0), 2)  # Finalize the line drawing
