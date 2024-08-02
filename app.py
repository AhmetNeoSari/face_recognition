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
from face_tracking.tracker.kalman_filter import KalmanFilter

from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.basetrack import BaseTrack, TrackState
from face_tracking.tracker.stracker import STrack
from recognize import Face_Recognize

face_detector_dict = {
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
"scalefactor" : 1.0 / 128.0
}

app_dict = {
    "tracker_usage" : True,
    "recognizer_usage" : True
}

tracker_args = {
    "device"    : "cpu",
    "fps"       : 30,
    "match_thresh": 0.8,
    "min_box_area": 10,
    "save_result" : True,
    "track_buffer": 30,
    "track_thresh": 0.5,
    "aspect_ratio_thresh": 1.6,
    "ckpt": "bytetrack_s_mot17.pth.tar",
    "fp16": True
}


all_args ={
    "device"    : "cpu",
    "fps"       : 30,
    "match_thresh": 0.8,
    "min_box_area": 10,
    "save_result" : True,
    "track_buffer": 30,
    "track_thresh": 0.5,
    "aspect_ratio_thresh": 1.6,
    "ckpt": "bytetrack_s_mot17.pth.tar",
    "fp16": True,
    "model_file" : "/home/ahmet/Documents/face-recognition/face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
    "taskname" : "detection",
    "batched" : False,
    "nms_thresh" : 0.4,
    "center_cache" : {},
    "session" : None,
    "detect_thresh" : 0.5,
    "detect_input_size" :(128,128),
    "max_num" : 0,
    "metric" : "default",
    "scalefactor" : 1.0 / 128.0
}

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

def main():

    """Main function to start face tracking and recognition threads."""

    recognizer = Face_Recognize(**my_dict)

    if my_dict["is_tracker_use"] == True:

        # Start tracking thread
        thread_track = threading.Thread(target=recognizer.tracking)
        thread_track.start()

        # Start recognition thread
        thread_recognize = threading.Thread(target=recognizer.recognize)
        thread_recognize.start()

    else:
        cap = cv2.VideoCapture(my_dict["video_source"])

        detector = Face_Detector(my_dict["model_file"],
                                 my_dict["taskname"] ,
                                 my_dict["batched"],
                                 my_dict["nms_thresh"],
                                 my_dict["center_cache"],
                                 my_dict["session"],
                                 my_dict["detect_thresh"],
                                 my_dict["detect_input_size"],
                                 my_dict["max_num"],
                                 my_dict["metric"],
                                 my_dict["scalefactor"])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bboxes, landmarks = detector.detect(image=frame)


if __name__ == "__main__":
    main()
