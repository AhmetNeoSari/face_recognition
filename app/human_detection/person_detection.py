from ultralytics import YOLO
import torch
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class PersonDetection:
    model_path : str
    confidence : float 
    iou_thresh : float
    input_size : tuple[int,int]
    logger : Any


    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model=self.model_path, verbose=False)
        self.model.fuse()
        self.model.to(self.device)
        self.logger.debug("PersonDetection İnitialized")

    @torch.no_grad()
    def detect(self, frame):
        return self.model.predict(frame, classes=[0], verbose=False)


    def save_one_box(self, xyxy, im, gain=1.02, pad=10, BGR=True):
        """
        Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

        Args:
            xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
            im (numpy.ndarray): The input image.
            gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
            pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
            BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to True.
            crop_mode (str, optional): Specify the cropping mode ('full', 'upper_half', '4by3'). Defaults to 'full'.

        Returns:
            (numpy.ndarray): The cropped image.
        """
        if not isinstance(xyxy, torch.Tensor):  # may be list
            xyxy = torch.stack(xyxy)

        b = self.xyxy2xywh(xyxy.view(-1, 4)) 
        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad

        xyxy = self.xywh2xyxy(b).long()
        xyxy = self.clip_boxes(xyxy, im.shape)
        
        crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
        return crop


    def clip_boxes(self , boxes, shape):
        """
        Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

        Args:
            boxes (torch.Tensor): the bounding boxes to clip
            shape (tuple): the shape of the image

        Returns:
            (torch.Tensor | numpy.ndarray): Clipped boxes
        """
        if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
            boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
            boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
            boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes


    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        xy = x[..., :2]  # centers
        wh = x[..., 2:] / 2  # half width-height
        y[..., :2] = xy - wh  # top left xy
        y[..., 2:] = xy + wh  # bottom right xy
        return y


    def xyxy2xywh(self, x):
        """
        Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
        """
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y


    def crop_persons_and_bboxes(self, frame, results:list):

        cropped_persons = []
        bboxes = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bboxes.append(box.xyxy.tolist()[0])
                bboxes[i].append(box.conf.tolist()[0])
                cropped = self.save_one_box(box.xyxy, frame)
                cropped_persons.append(cropped)

        return cropped_persons, bboxes
    

if __name__ == "__main__":
    

    class Custom_logger:
        def error(self, message: str):
            print(message)
        
        def warning(self, message: str):
            print(message)
        
        def info(self, message: str):
            print(message)
        
        def debug(self, message: str):
            print(message)
        
        def critical(self, message: str):
            print(message)
        
        def trace(self, message: str):
            print(message)

    def draw(frame, bboxes):
        for box in bboxes:
            if len(box) == 5 :
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)


    person_detector_dict = {
        "model_path"  : "yolov9c.pt",
        "confidence"  : 0.5,
        "iou_thresh"  : 0.5,
        "input_size"  : [640,640]
    }

    logger = Custom_logger()

    person_detector = PersonDetection(**person_detector_dict, logger=logger)
    cap = cv2.VideoCapture("a.mp4")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions

    while True:
        _, frame = cap.read()

        # frame = cv2.resize(frame, (480,480))
        results = person_detector.detect(frame=frame)
        persons, bboxes = person_detector.crop_persons_and_bboxes(frame=frame, results=results)
        draw(frame=frame, bboxes=bboxes)

        # print("bboxes: ",bboxes)

        # for i, person in enumerate(persons):
        #     cv2.imshow(f"person{i}",person)

        
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # cv2.imshow("frame", frame)