import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import Any

@dataclass
class Draw():
    logger: Any
    def __post_init__(self):
        self.text_scale = 2
        self.text_thickness = 2
        self.line_thickness = 3
        self.font = ImageFont.truetype("DejaVuSans.ttf", 30)
        self.text_color = (0, 0, 255)


    def _get_color(self, idx):
        """
        Generate a color based on an index value.

        Args:
            idx (int): Index to generate a unique color.

        Returns:
            tuple: A tuple representing the BGR color.
        """
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color


    def plot(self, frame:np.ndarray, tlwhs:list, obj_ids:list, names:list):
        im = np.ascontiguousarray(np.copy(frame))
        
        # PIL kullanarak görüntüyü açma
        pil_img = Image.fromarray(im)
        draw = ImageDraw.Draw(pil_img)
        
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if obj_id in names:
                id_text = id_text + ": " + names[obj_id]
            color = self._get_color(abs(obj_id))
            
            # draw rectangle
            try:
                draw.rectangle([ (abs(intbox[0]), abs(intbox[1]) ) , ( abs(intbox[2]), abs(intbox[3]) ) ], outline=color, width=self.line_thickness)
                draw.text(
                (intbox[0], intbox[1]-40),
                id_text,
                font=self.font,
                fill=self.text_color
                )
            except Exception as E:
                self.logger.warning("Error when drawing rectangle")

        # Converting PIL image to numpy array
        im = np.array(pil_img)
        
        return im
    

    def draw_text(self, frame: np.ndarray, total_people: int, people_inside: list):
        """
        Draws the total number of people inside and the names of those people on the frame.

        Args:
            frame (np.ndarray): Frame to draw on.
            total_people (int): Total number of people inside.
            people_inside (list): List of names of people inside.

        Returns:
            np.ndarray: Frame with the text drawn on it.
        """
        im = np.ascontiguousarray(np.copy(frame))
        pil_img = Image.fromarray(im)
        draw = ImageDraw.Draw(pil_img)
        
        # Text content
        text = f"Total Inside: {total_people}"
        people_text = f"People Inside: {', '.join(people_inside)}" if people_inside else "People Inside: 0"
        
        # Draw the total people and the names of people inside
        draw.text((10, 10), text, font=self.font, fill=self.text_color)
        draw.text((10, 50), people_text, font=self.font, fill=self.text_color)

        # Convert back to numpy array
        im = np.array(pil_img)
        return im
    