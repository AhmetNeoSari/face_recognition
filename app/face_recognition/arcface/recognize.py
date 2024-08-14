import cv2
import numpy as np
import torch
from torchvision import transforms
import argparse
from dataclasses import dataclass
from typing import Any

if __name__ == "__main__":
    from recognizer_utils import norm_crop, compare_encodings, read_features, iresnet_inference
else:
    from .recognizer_utils import norm_crop, compare_encodings, read_features, iresnet_inference


@dataclass
class Face_Recognize:

    recognizer_model_name : str
    recognizer_model_path : str
    feature_path : str 
    mapping_score_thresh : float
    recognition_score_thresh : float
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

    def recognize(self, frame ,bboxes, landmarks, data_mapping:dict, is_tracker_available:bool):
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
        self.id_face_mapping = {}
        if is_tracker_available == False:
            # captions = {"id",name}  # Dictionary to store captions for each bounding box
            counter = -1
            for bbox, landmark in zip(bboxes, landmarks):
                # Face alignment
                counter += 1
                face_alignment = norm_crop(img=frame, landmark=landmark)

                score, name = self.recognition(face_image=face_alignment)

                if name is not None:
                    if score < self.recognition_score_thresh:
                        caption = "UN_KNOWN"
                    else:
                        caption = f"{name}:{score:.2f}"
                else:
                    caption = "UN_KNOWN"

                # Use bbox as key to ensure unique identification
                self.id_face_mapping[counter] = caption
            return
        
        if data_mapping["tracking_bboxes"] == None:
            self.logger.error("data_mapping must be provided when is_tracker_use is True")
            raise ValueError("data_mapping must be provided when is_tracker_use is True")
        
        detection_landmarks = landmarks
        detection_bboxes = bboxes
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = self.mapping_bbox(box1=tracking_bboxes[i],
                                                box2=detection_bboxes[j])
                if mapping_score > self.mapping_score_thresh:
                    face_alignment = norm_crop(img=frame,
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



if __name__ == "__main__":
    """Main function to start recognition threads."""      

    import os
    import sys

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
        "center_cache": {},
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
        _, frame = cap.read()
        outputs, bboxes, landmarks = face_detector.detect_tracking(frame)
        face_recognizer.recognize(frame ,bboxes, landmarks, data_mapping, False)
