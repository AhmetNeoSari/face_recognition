import time
import cv2
import argparse
import sys
import os

# print("\n")
# print(os.path.dirname(os.path.abspath(__file__)))
# print("\n")

from app.face_detection.scrfd.face_detector import Face_Detector
from app.face_tracking.byte_tracker import BYTETracker
from app.face_recognition.arcface.recognize import Face_Recognize
from app.config import Config


def video_source_type(value):
    try:
        return int(value)
    except ValueError:
        return str(value)

def main(args):
    cap = cv2.VideoCapture(args.video_source)
    config = Config("configs")
    attributes = config.load(args.env)

    face_recognizer = Face_Recognize(**attributes["recognition"])
    face_detector = Face_Detector(**attributes["detection"])
    face_tracker = BYTETracker(**attributes["tracker"])

    frame_id = 0
    start_time = time.time_ns()
    frame_count = 0
    fps = -1
    id_face_mapping = {}
    while True:
        _, frame = cap.read()
        frame_id += 1

        outputs, img_info, bboxes, landmarks = face_detector.detect_tracking(frame)
        tracking_image, data_mapping = face_tracker.track(outputs, img_info, fps, id_face_mapping)
        id_face_mapping,caption = face_recognizer.recognize(img_info["raw_img"] ,bboxes, landmarks, data_mapping)

        end_timer = time.time()
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        if args.show:
            cv2.imshow("frame", tracking_image)
            # Press 'Q' on the keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--video-source', type=video_source_type, 
                    help='Video source (0: webcam, use string for file path)')
    parser.add_argument('--show', action='store_true', help='Enable showing the result of tracker')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Disable showing the result of tracker')
    parser.add_argument('--env', type=str, required=True, help='Specify the environment (local or prod or etc.)')

    parser.set_defaults(video_source=0 ,show=True)
    args = parser.parse_args()

    main(args)
