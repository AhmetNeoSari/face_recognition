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
    video_source : Optional[int|str]
    model_file  : str
    taskname    : str 
    batched     : bool
    nms_thresh  : int
    center_cache: dict
    session     : onnxruntime.InferenceSession
    detect_thresh : float
    detect_input_size : tuple[int,int]
    max_num     : int
    metric      : str
    scalefactor : float

    recognizer_model_name : str
    recognizer_model_path : str
    feature_path : str #"./datasets/face_features/feature"
    mapping_score_thresh : float
    recognition_score_thresh : float

    min_box_area: int
    aspect_ratio_thresh: float
    ckpt: str
    fps     : int
    match_thresh: Optional[int]    = None
    track_buffer: Optional[int]    = None
    track_thresh: Optional[float]  = None
    fp16: Optional[bool]           = None
    tracker_frame_rate : Optional[int] = None

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Device configuration
        if self.is_tracker_use == True:
            if (self.device == None or self.match_thresh == None or self.track_buffer == None or \
                self.track_thresh == None or self.fp16 == None or self.tracker_frame_rate == None) :
                
                raise ValueError("If a tracker is to be used, the necessary parameters must be specified.")

        self.detector = Face_Detector(  self.model_file,
                                        self.taskname,
                                        self.batched,
                                        self.nms_thresh,
                                        self.center_cache,
                                        self.session,
                                        self.detect_thresh,
                                        self.detect_input_size,
                                        self.max_num,
                                        self.metric,
                                        self.scalefactor)

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

        # Data mapping for tracking information
        self.data_mapping = {
            "raw_image": [],
            "tracking_ids": [],
            "detection_bboxes": [],
            "detection_landmarks": [],
            "tracking_bboxes": [],
        }


    def process_tracking(self, frame, tracker, frame_id):
        """
        Process tracking for a frame.

        Args:
            frame: The input frame.
            tracker: The object tracker.
            args (dict): Tracking configuration parameters.
            frame_id (int): The frame ID.

        Returns:
            numpy.ndarray: The processed tracking image.
        """
        # Face detection and tracking
        outputs, img_info, bboxes, landmarks = self.detector.detect_tracking(image=frame)

        tracking_tlwhs = []
        tracking_ids = []
        tracking_scores = []
        tracking_bboxes = []

        if outputs is not None:
            online_targets = tracker.update(
                outputs, [img_info["height"], img_info["width"]], (128, 128)
            )

            for i in range(len(online_targets)):
                t = online_targets[i]
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    x1, y1, w, h = tlwh
                    tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)
                    tracking_scores.append(t.score)

            tracking_image = self.plot_tracking(
                img_info["raw_img"],
                tracking_tlwhs,
                tracking_ids,
                names=self.id_face_mapping,
                frame_id=frame_id + 1,
            )
        else:
            tracking_image = img_info["raw_img"]

        self.data_mapping["raw_image"] = img_info["raw_img"]
        self.data_mapping["detection_bboxes"] = bboxes
        self.data_mapping["detection_landmarks"] = landmarks
        self.data_mapping["tracking_ids"] = tracking_ids
        self.data_mapping["tracking_bboxes"] = tracking_bboxes

        return tracking_image


    def plot_tracking(
        self, image, tlwhs, obj_ids, scores=None, frame_id=0, ids2=None, names=[]
    ):
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

        # text_scale = max(1, image.shape[1] / 1600.)
        # text_thickness = 2
        # line_thickness = max(1, int(image.shape[1] / 500.))
        text_scale = 2
        text_thickness = 2
        line_thickness = 3

        radius = max(5, int(im_w / 140.0))
        cv2.putText(
            im,
            "frame: %d fps: %.2f num: %d" % (frame_id, self.fps, len(tlwhs)),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            thickness=2,
        )

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if (obj_id) in names:
                id_text = id_text + ": " + names[obj_id]
            if ids2 is not None:
                id_text = id_text + ", {}".format(int(ids2[i]))
            color = self.get_color(abs(obj_id))
            cv2.rectangle(
                im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
            )
            cv2.putText(
                im,
                id_text,
                (intbox[0], intbox[1]),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=text_thickness,
            )
        return im
    


    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        return color


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


    def tracking(self):
        """
        Face tracking in a separate thread.

        Args:
            args (dict): Tracking configuration parameters.
        """
        # Initialize variables for measuring frame rate
        start_time = time.time_ns()
        frame_count = 0
        fps = -1

        # Initialize a tracker and a timer
        tracker = BYTETracker(  self.device,
                                self.match_thresh,
                                self.track_buffer,
                                self.track_thresh,
                                self.fp16,
                                self.tracker_frame_rate)
        frame_id = 0

        cap = cv2.VideoCapture(self.video_source)

        while True:
            print("zzz")
            _, img = cap.read()

            tracking_image = self.process_tracking(img, tracker, frame_id)

            # Calculate and display the frame rate
            frame_count += 1
            if frame_count >= 30:
                self.fps = 1e9 * frame_count / (time.time_ns() - start_time)
                frame_count = 0
                start_time = time.time_ns()

            cv2.imshow("Face Recognition", tracking_image)

            # Check for user exit input
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break


    def recognize(self):
        """Face recognition in a separate thread."""
        while True:
            raw_image = self.data_mapping["raw_image"]
            detection_landmarks = self.data_mapping["detection_landmarks"]
            detection_bboxes = self.data_mapping["detection_bboxes"]
            tracking_ids = self.data_mapping["tracking_ids"]
            tracking_bboxes = self.data_mapping["tracking_bboxes"]

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
                # print("Waiting for a person...")
                pass

if __name__ == "__main__":
    
    
    """Main function to start face tracking and recognition threads."""

    my_dict = {
        "is_tracker_use" : True,
        "video_source" : 0,
        "model_file" : "/home/ahmet/workplace/face_recognition/face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
        "taskname" : "detection",
        "batched" : False,
        "nms_thresh" : 0.4,
        "center_cache" : {},
        "session" : None,
        "detect_thresh" : 0.5,
        "detect_input_size" :(128,128),
        "max_num" : 0,
        "metric" : "default",
        "scalefactor" : 1.0 / 128.0,
        "recognizer_model_name" : "r100",
        "recognizer_model_path" : "/home/ahmet/workplace/face_recognition/face_recognition/arcface/weights/arcface_r100.pth",
        "feature_path" : "/home/ahmet/workplace/face_recognition/datasets/face_features/feature",
        "mapping_score_thresh" : 0.9,
        "recognition_score_thresh" : 0.25,
        "min_box_area": 10,
        "aspect_ratio_thresh": 1.6,
        "ckpt": "bytetrack_s_mot17.pth.tar",
        "fps"     : 30,
        "match_thresh": 0.8,
        "track_buffer": 30,
        "track_thresh": 0.5,
        "fp16": True,
        "tracker_frame_rate" : 30
    }

    obje = Face_Recognize(**my_dict)

    # Start tracking thread
    thread_track = threading.Thread(target=obje.tracking)
    thread_track.start()

    # Start recognition thread
    thread_recognize = threading.Thread(target=obje.recognize)
    thread_recognize.start()

