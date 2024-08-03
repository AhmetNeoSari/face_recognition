import threading
import time

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.face_detector import Face_Detector
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
import onnxruntime

from dataclasses import dataclass
from typing import Optional

@dataclass
class Face_Recognize:

    is_tracker_use : bool
    video_source : int
    recognizer_model_name : str
    recognizer_model_path : str
    feature_path : str 
    mapping_score_thresh : float
    recognition_score_thresh : float


    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Face recognizer
        self.recognizer = iresnet_inference(
            model_name=self.recognizer_model_name, 
            path=self.recognizer_model_path, 
            device=self.device
        )

        # Load precomputed face features and names
        self.images_names, self.images_embs = read_features(feature_path = self.feature_path)

        # Mapping of face IDs to names
        self.id_face_mapping = {}


    def recognize(self, frame ,bboxes, landmarks, data_mapping:dict = None ):
        """Face recognition in a separate thread."""
        if self.is_tracker_use:
            if data_mapping == None:
                raise AttributeError 
            
            caption = "UN_KNOWN"
            raw_image = frame
            detection_landmarks = landmarks
            detection_bboxes = bboxes
            tracking_ids = data_mapping["tracking_ids"]
            tracking_bboxes = data_mapping["tracking_bboxes"]

            for i in range(len(tracking_bboxes)):
                for j in range(len(detection_bboxes)):
                    mapping_score = self.mapping_bbox(box1=tracking_bboxes[i],
                                                       box2=detection_bboxes[j])
                    if mapping_score > self.mapping_score_thresh:
                        face_alignment = norm_crop(img=raw_image,
                                                   landmark=detection_landmarks[j])

                        score, name = self.recognition(face_image=face_alignment)
                        if name is not None:
                            if score < self.recognition_score_thresh:
                                caption = "UN_KNOWN"
                            else:
                                caption = f"{name}:{score:.2f}"

                        self.id_face_mapping[tracking_ids[i]] = caption

                        detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                        detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                        break

            if tracking_bboxes == []:
                caption = "UN_KNOWN"

        else:
            self.id_face_mapping = {}
            for bbox, landmark in zip(bboxes, landmarks):
                x1, y1, x2, y2, _ = bbox.astype(int)
                
                # Face alignment
                face_alignment = norm_crop(img=frame, landmark=landmark)

                # Face recognition
                score, name = self.recognition(face_image=face_alignment)

                # Draw bounding box and name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if score < self.recognition_score_thresh:
                    caption = "UNKNOWN"
                else:
                    caption = f"{name}: {score:.2f}"
                cv2.putText(frame, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return self.id_face_mapping, caption
    

    @torch.no_grad()
    def get_feature(self, face_image):
        """
        Extract features from a face image.

        Args:
            face_image: The input face image.

        Returns:
            numpy.ndarray: The extracted features.
        """
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Convert to RGB
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess image (BGR)
        face_image = face_preprocess(face_image).unsqueeze(0).to(self.device)

        # Inference to get feature
        emb_img_face = self.recognizer(face_image).cpu().numpy()

        # Convert to array
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)

        return images_emb


    def recognition(self, face_image):
        """
        Recognize a face image.

        Args:
            face_image: The input face image.

        Returns:
            tuple: A tuple containing the recognition score and name.
        """
        # Get feature from face
        query_emb = self.get_feature(face_image)

        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min]
        score = score[0]

        return score, name


    def mapping_bbox(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
            box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

        Returns:
            float: The IoU score.
        """
        # Calculate the intersection area
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
            0, y_max_inter - y_min_inter + 1
        )

        # Calculate the area of each bounding box
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate the union area
        union_area = area_box1 + area_box2 - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou


face_recognizer_dict = {
    "is_tracker_use" : False,
    "video_source" : 0,
    "recognizer_model_name" : "r100",
    "recognizer_model_path" : "/home/ahmet/workplace/face_recognition/face_recognition/arcface/weights/arcface_r100.pth",
    "feature_path" :  "/home/ahmet/workplace/face_recognition/datasets/face_features/feature",
    "mapping_score_thresh" : 0.9,
    "recognition_score_thresh" : 0.25
}


face_detector_dict = {
    "model_file"  : "/home/ahmet/workplace/face_recognition/face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
    "taskname"    : "detection",
    "batched"     : False,
    "nms_thresh"  : 0.4,
    "center_cache": {},
    "session"     : None,
    "detect_thresh" : 0.5,
    "detect_input_size" : (128,128),
    "max_num"     : 0,
    "metric"      : "default",
    "scalefactor" : 1.0 / 128.0,
}

if __name__ == "__main__":
    """Main function to start face tracking and recognition threads."""

    cap = cv2.VideoCapture(0)
    face_recognizer = Face_Recognize(**face_recognizer_dict)
    face_detector = Face_Detector(**face_detector_dict)


    frame_id = 0
    id_face_mapping = {}
    while True:
        _, frame = cap.read()
        frame_id += 1

        outputs, img_info, bboxes, landmarks = face_detector.detect_tracking(frame)

        id_face_mapping ,caption = face_recognizer.recognize(frame ,bboxes, landmarks)
        
        print(caption)