import os
import time
import sys

import cv2
import numpy as np
import onnxruntime
import torch
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from face_detection.scrfd.face_detector import Face_Detector
from face_tracking.tracker.byte_tracker import BYTETracker
from recognize import Face_Recognize


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


face_recognizer_dict = {
    "is_tracker_use" : True,
    "video_source" : 0,
    "recognizer_model_name" : "r100",
    "recognizer_model_path" : "/home/ahmet/workplace/face_recognition/face_recognition/arcface/weights/arcface_r100.pth",
    "feature_path" :  "/home/ahmet/workplace/face_recognition/datasets/face_features/feature",
    "mapping_score_thresh" : 0.9,
    "recognition_score_thresh" : 0.25
}


face_tracker_dict = {
    "match_thresh": 0.8,
    "track_buffer": 30,
    "track_thresh": 0.5,
    "fp16": True,
    "frame_rate" : 30,
    "min_box_area": 10,
    "aspect_ratio_thresh": 1.6,
    "ckpt": "bytetrack_s_mot17.pth.tar",
    "fps" : 30,
    "track_img_size" : (128,128)
}


def main():

    """Main function to start face tracking and recognition threads."""

    cap = cv2.VideoCapture(0)
    face_recognizer = Face_Recognize(**face_recognizer_dict)
    face_detector = Face_Detector(**face_detector_dict)
    face_tracker = BYTETracker(**face_tracker_dict)
    frame_id = 0
    id_face_mapping = {}
    while True:
        _, frame = cap.read()
        frame_id += 1

        outputs, img_info, bboxes, landmarks = face_detector.detect_tracking(frame)
        tracking_image, data_mapping = face_tracker.track(outputs, img_info, bboxes, landmarks, id_face_mapping,frame_id)
        id_face_mapping ,caption = face_recognizer.recognize(img_info["raw_img"] ,bboxes, landmarks, data_mapping)

        cv2.imshow("frame", tracking_image)
        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        

if __name__ == "__main__":
    main()
