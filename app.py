import cv2
import argparse
import torch

from app.face_detection.scrfd.face_detector import Face_Detector
from app.face_tracking.byte_tracker import BYTETracker
from app.face_recognition.arcface.recognize import Face_Recognize
from app.config import Config
from app.fps import Fps
from app.utils import Draw
from app.logger import Logger
from app.streamer import Streamer
from app.human_detection.person_detection import PersonDetection
from app.person_counting.person_counter import ObjectCounter

def main(args):
    config = Config("configs")
    configs = config.load()

    logger = Logger(**configs["logger"])
    logger.debug('Application started')

    streamer         =    Streamer(           **configs["streamer"],           logger=logger)
    face_recognizer  =    Face_Recognize(     **configs["recognition"],        logger=logger)
    face_detector    =    Face_Detector(      **configs["detection"],          logger=logger)
    face_tracker     =    BYTETracker(        **configs["tracker"],            logger=logger)
    person_detector  =    PersonDetection(    **configs["person_detection"],   logger=logger)
    object_counter   =    ObjectCounter(      **configs["person_counter"],     logger=logger)
    plot_object      =    Draw(                                                logger=logger)
    fps_object       =    Fps()

    streamer.initialize()
    if args.show:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while True:
        frame = next(streamer.read_frame())
        if args.fps:
            fps_object.begin_timer()
            fps_object.count_frame()

        results = person_detector.detect(frame=frame)
        persons, bboxes_ = person_detector.crop_persons_and_bboxes(frame=frame, results=results)
        face_tracker.track(torch.tensor(bboxes_), frame.shape[0], frame.shape[1])
        for cropped_frame, person_bbox in zip(persons, bboxes_):
            outputs, face_bboxes, landmarks = face_detector.detect(cropped_frame)
            face_recognizer.recognize(cropped_frame, face_bboxes, person_bbox, landmarks, face_tracker.data_mapping)

        object_counter.count(frame, results=results[0], tracker_results=face_tracker.data_mapping, names=face_recognizer.id_face_mapping)
       
        if args.fps:
            fps = fps_object.calculate_fps()
            logger.info(f"fps:{int(fps)}")

        if not args.show:
            continue

        frame = plot_object.plot( frame   = frame,
                                  tlwhs   = face_tracker.data_mapping["tracking_tlwhs"],
                                  obj_ids = face_tracker.data_mapping["tracking_ids"],
                                  names   = face_recognizer.id_face_mapping)
        
        cv2.imshow("frame", frame)
        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--show', action='store_true', 
                        help='Enable showing the result of tracker (default is False)')
    parser.add_argument('--fps', action='store_true', 
                        help='Print fps to terminal (default is False)')
    args = parser.parse_args()

    main(args)
