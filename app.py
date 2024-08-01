import os
import time
import sys

import cv2
import numpy as np
import onnxruntime
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from face_detection.scrfd.face_detector import Face_Detector
from face_tracking.tracker.kalman_filter import KalmanFilter

from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.basetrack import BaseTrack, TrackState
from face_tracking.tracker.stracker import STrack


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



def main():

    video_source = 0 #camera or video path
    cap = cv2.VideoCapture(video_source)

    detector = Face_Detector(**face_detector_dict)

    while True:

        # Capture a frame from the camera
        _, frame = cap.read()

        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)



        if app_dict["tracker_usage"] == 1:
            
        else:
            pass
    



if __name__ == "__main__":
    main()




