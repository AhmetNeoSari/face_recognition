import cv2
import numpy as np
import torch
from torchvision import transforms
import argparse
from dataclasses import dataclass
from typing import Any, Callable
import time
if not __name__ == "__main__":
    from .recognizer_utils import norm_crop, compare_encodings, read_features, iresnet_inference


@dataclass
class Face_Recognize:

    recognizer_model_name : str
    recognizer_model_path : str
    feature_path : str 
    mapping_score_thresh : float
    recognition_score_thresh : float
    frame_for_recognize : int
    face_location_tolerance : float
    logger : Any

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Face recognizer
        self.recognizer = iresnet_inference(
            model_name=self.recognizer_model_name, 
            path=self.recognizer_model_path, 
            device=self.device
        )

        self.images_names, self.images_embs = read_features(feature_path = self.feature_path)

        # Mapping of face IDs to names
        self.id_face_mapping = {}
        self.frame_counters  = {}

    def recognize(self, frame ,face_bbox, person_bbox ,landmarks, data_mapping:dict):
        """
        Perform face recognition on the provided frame.

        Args:
            frame (numpy.ndarray): The input video frame.
            bboxes (numpy.ndarray): Detected bounding boxes for faces.
            landmarks (numpy.ndarray): Detected landmarks for faces.
            data_mapping (dict, optional): Mapping data for tracking. Required if `is_tracker_use` is True.

        Returns:
            tuple: A tuple containing:
                - id_face_mapping (dict): Mapping of face IDs to recognition results.
                - caption (str): Recognition result for the current frame.
        """
        detection_landmarks = landmarks
        face_detection_bboxes = face_bbox
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            tracking_id = tracking_ids[i]

            if tracking_id not in self.frame_counters:
                self.frame_counters[tracking_id] = 0

            if tracking_id in self.id_face_mapping:
                if self.id_face_mapping[tracking_id] != "UN_KNOWN":
                    self.frame_counters[tracking_id] += 1
                    if self.frame_counters[tracking_id] < self.frame_for_recognize or self.frame_for_recognize == 0:
                        continue
                    self.frame_counters[tracking_id] = 0
                else:
                    self.frame_counters[tracking_id] = 0
                
            self.id_face_mapping[tracking_id] = "UN_KNOWN"
            for j in range(len(face_detection_bboxes)): # complexity O(n) = 1
                mapping_score = self.mapping_bbox(box1=tracking_bboxes[i],
                                                box2=person_bbox)
                if mapping_score > self.mapping_score_thresh and self.is_face_reasonably_positioned(person_bbox=person_bbox, face_bbox=face_bbox[j]):                    
                # if mapping_score > self.mapping_score_thresh:                    
                    face_alignment = norm_crop(img=frame,
                                            landmark=detection_landmarks[j])

                    score, name = self.recognition(face_image=face_alignment)
                    if name is not None:
                        if score < self.recognition_score_thresh:
                            caption = "UN_KNOWN"
                        else:
                            caption = f"{name}:{score:.2f}"

                    new_id = tracking_id
                    new_value = caption

                    person_name = new_value.split(':')[0]
                    old_key = None
                    for key,value in self.id_face_mapping.items():
                        if person_name in value:
                            old_key = key
                            break
                    
                    if old_key is not None:
                        del self.id_face_mapping[old_key]
                        self.id_face_mapping[new_id] = new_value
                    else:
                        self.id_face_mapping[new_id] = new_value

                    face_detection_bboxes = np.delete(face_detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                    break

    def is_face_reasonably_positioned(self, person_bbox, face_bbox):
        """
        Checks whether the face is in a logical place in the general bounding box of the person.
        
        Args:
            person_bbox (tuple): General bounding box of the person (x1, y1, x2, y2).
            face_bbox (tuple): Face bounding box (x1, y1, x2, y2).
        
        Returns:
            bool: True if the face is in a reasonable position within the person's general bounding box, False otherwise.

        """
        face_center_x = (face_bbox[0] + face_bbox[2]) / 2
        face_center_y = (face_bbox[1] + face_bbox[3]) / 2

        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]

        face_height = face_bbox[3] - face_bbox[1]

        if face_height > (person_height / 2) : #webcam
            return True
        if person_width * 0.1 < face_center_x < person_width * 0.9 and face_center_y < person_height / 2:
            return True
        return False
        

    @torch.no_grad()
    def get_feature(self, face_image):
        """
        Extract features from a face image.

        Args:
            face_image (numpy.ndarray): The input face image.

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
            face_image (numpy.ndarray): The input face image.

        Returns:
            tuple: A tuple containing:
                - score (float): The recognition score.
                - name (str): The recognized name.
        """
        try:
            # Get feature from face
            query_emb = self.get_feature(face_image)

            score, id_min = compare_encodings(query_emb, self.images_embs)
            name = self.images_names[id_min]
            score = score[0]
            
            if score < self.recognition_score_thresh:
                return score, "UN_KNOWN"

            return score, name
        except ValueError:
            # If there are no embeddings in the database, return unknown
            return 0, "UN_KNOWN"


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



if __name__ == "__main__":     
    import os
    import sys
    from recognizer_utils import norm_crop, compare_encodings, read_features, iresnet_inference

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    sys.path.append(root_dir)
    sys.path.append(current_dir)

    from face_detection.scrfd.face_detector import Face_Detector
    
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


    face_recognizer_dict = {
        "recognizer_model_name" : "r100",
        "recognizer_model_path" : "weights/arcface_r100.pth",
        "feature_path" :  "datasets/face_features/feature",
        "mapping_score_thresh" : 0.9,
        "recognition_score_thresh" : 0.25
    }


    face_detector_dict = {
        "model_file"  : "../../face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
        "taskname"    : "detection",
        "batched"     : False,
        "nms_thresh"  : 0.4,
        "session"     : "",
        "detect_thresh" : 0.5,
        "detect_input_size" : (128,128),
        "max_num"     : 0,
        "metric"      : "default",
        "scalefactor" : 1.0 / 128.0,
    }


    def video_source_type(value):
        try:
            return int(value)
        except ValueError:
            return str(value)
        
    logger = Custom_logger()

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--video-source', type=video_source_type, default=0, 
                        help='Video source (0: webcam, use string for file path)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)

    face_recognizer = Face_Recognize(**face_recognizer_dict,logger=logger)
    face_detector = Face_Detector(**face_detector_dict,logger=logger)
    data_mapping = None
    id_face_mapping = {}
    while True:
        star_timer = time.time()
        _, frame = cap.read()
        outputs, bboxes, landmarks = face_detector.detect(frame)
        face_recognizer.recognize(frame ,bboxes, landmarks, data_mapping, False)
        end_time = time.time()

        fps = 1 / (end_time-star_timer)
        print("fps: ",fps)
