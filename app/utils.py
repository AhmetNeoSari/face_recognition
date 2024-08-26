import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

@dataclass
class Draw():
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
            draw.rectangle([intbox[0:2], intbox[2:4]], outline=color, width=self.line_thickness)
            
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

    # def plot(
    #     self, frame:np.ndarray, tlwhs:list, obj_ids:list, names:list, bboxes:np.ndarray, fps
    # ):
    #     im = np.ascontiguousarray(np.copy(frame))
    #     im_h, im_w = im.shape[:2]
    #     print("names:",names)
    #     top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #     # text_scale = max(1, image.shape[1] / 1600.)
    #     # text_thickness = 2
    #     # line_thickness = max(1, int(image.shape[1] / 500.))
    #     text_scale = 2
    #     text_thickness = 2
    #     line_thickness = 3
    #     radius = max(5, int(im_w / 140.0))
    #     for i, tlwh in enumerate(tlwhs):
    #         x1, y1, w, h = tlwh
    #         intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    #         obj_id = int(obj_ids[i])
    #         id_text = "{}".format(int(obj_id))
    #         if (obj_id) in names:
    #             id_text = id_text + ": " + names[obj_id]
    #         color = self._get_color(abs(obj_id))
    #         cv2.rectangle(
    #             im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
    #         )
    #         cv2.putText(
    #             im,
    #             id_text,
    #             (intbox[0], intbox[1]),
    #             cv2.FONT_HERSHEY_PLAIN,
    #             text_scale,
    #             (0, 0, 255),
    #             thickness=text_thickness,
    #         )
    #     return im