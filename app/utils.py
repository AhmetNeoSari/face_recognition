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
        
        # Font ayarları (bir TTF dosyası belirtmelisiniz)
        
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if obj_id in names:
                id_text = id_text + ": " + names[obj_id]
            color = self._get_color(abs(obj_id))
            
            # Rectangle çizimi
            try:
                draw.rectangle([ (abs(intbox[0]), abs(intbox[1]) ) , ( abs(intbox[2]), abs(intbox[3]) ) ], outline=color, width=self.line_thickness)
            except Exception as E:
                self.logger.warning("Error when drawing rectangle")
            # Text çizimi
            draw.text(
                (intbox[0], intbox[1]-40),
                id_text,
                font=self.font,
                fill=self.text_color
            )
        
        # PIL görüntüsünü numpy array'e dönüştürme
        im = np.array(pil_img)
        
        return im
