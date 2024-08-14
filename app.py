import cv2
import argparse

from app.face_detection.scrfd.face_detector import Face_Detector
from app.face_tracking.byte_tracker import BYTETracker
from app.face_recognition.arcface.recognize import Face_Recognize
from app.config import Config
from app.fps import Fps
from app.utils import plot
from app.logger import Logger
from app.streamer import Streamer

def main(args):
    config = Config("configs")
    configs = config.load(args.env)

    logger = Logger(**configs["logger"])
    
    logger.info('Application started')
    logger.info(f'Configuration loaded for environment: {args.env}')

    streamer         =    Streamer(           **configs["streamer"],           logger=logger)
    face_recognizer  =    Face_Recognize(     **configs["recognition"],        logger=logger)
    face_detector    =    Face_Detector(      **configs["detection"],          logger=logger)
    face_tracker     =    BYTETracker(        **configs["tracker"],            logger=logger)
    fps_object       =    Fps()

    streamer.initialize()
    fps = None
    data_mapping = {
        "tracking_ids": None,
        "tracking_bboxes": None,
        "tracking_tlwhs" : None,
    }
    while True:
        frame = next(streamer.read_frame())

        if args.fps:
            fps_object.begin_timer()
            fps_object.count_frame()

        outputs, bboxes, landmarks = face_detector.detect_tracking(frame)
        data_mapping = face_tracker.track(outputs, frame.shape[0], frame.shape[1])
        face_recognizer.recognize(frame ,bboxes, landmarks, data_mapping, face_tracker.is_tracker_available)
        
        if args.fps:
            fps = fps_object.calculate_fps()
            logger.info(f"fps:{int(fps)}")

        if not args.show:
        # If args.show is False, skip the block of code that shows the frame
            continue

        frame = plot(frame=frame,
                    tlwhs=data_mapping["tracking_tlwhs"],
                    obj_ids=data_mapping["tracking_ids"],
                    names=face_recognizer.id_face_mapping,
                    bboxes=bboxes,
                    )

        cv2.imshow("frame", frame)
        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--show', action='store_true', 
                        help='Enable showing the result of tracker (default is False)')
    parser.add_argument('--env', type=str, required=True, 
                        help='Specify the environment (e.g., local, prod)')
    parser.add_argument('--fps', action='store_true', 
                        help='Print fps to terminal (default is False)')
    args = parser.parse_args()

    main(args)
