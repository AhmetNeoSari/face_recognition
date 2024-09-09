import cv2
import argparse
import torch
from multiprocessing import Process, Queue, set_start_method
from time import sleep

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

def person_detect_tracking(frame_queue: Queue, detection_queue: Queue, logger_config:dict, person_detector_config:dict, person_tracker_config:dict):
    """
    This function performs person detection and tracking.

    Args:
        frame_queue (Queue): Queue from which frames are retrieved.
        detection_queue (Queue): Queue where detection results are sent.
        logger_config (dict): Configuration for logging.
        person_detector_config (dict): Configuration for person detection model.
        person_tracker_config (dict): Configuration for person tracking model.
    """
    logger = Logger(**logger_config)
    person_detector  =    PersonDetection(    **person_detector_config,   logger=logger)  
    person_tracker   =    BYTETracker(        **person_tracker_config,    logger=logger)

    while True:
        try:
            frame = frame_queue.get(timeout=1)
            # Person Detection
            results = person_detector.detect(frame=frame)
            persons, bboxes_ = person_detector.crop_persons_and_bboxes(frame=frame, results=results)
            
            person_tracker.track(torch.tensor(bboxes_), frame.shape[0], frame.shape[1])
            
            detection_queue.put({
                "frame"       : frame,
                "results"     : results,
                "persons"     : persons,
                "bboxes_"     : bboxes_,
                "data_mapping": person_tracker.data_mapping
            })
        except Exception as e:
            # Kuyruk boşsa timeout hatasını yoksaymak için
            logger.error(f"Error in detection_process: {e}")
            continue


def face_detect_recognize_count(detection_queue: Queue, show:bool,
                                logger_config:dict, face_detector_config:dict, face_recognizer_config:dict, person_counter_config:dict):
    
    """
    This function performs face detection, recognition, and people counting.

    Args:
        detection_queue (Queue): Queue from which detection results are retrieved.
        show (bool): If true, displays the processed frame.
        logger_config (dict): Configuration for logging.
        face_detector_config (dict): Configuration for face detection model.
        face_recognizer_config (dict): Configuration for face recognition model.
        person_counter_config (dict): Configuration for person counting system.
    """
    logger           =    Logger(             **logger_config)
    face_recognizer  =    Face_Recognize(     **face_recognizer_config,         logger=logger)
    face_detector    =    Face_Detector(      **face_detector_config,           logger=logger)
    person_counter   =    ObjectCounter(      **person_counter_config,          logger=logger)
    plot_object      =    Draw(                                                 logger=logger)

    while True:
        try:
            if show:
                cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            data = detection_queue.get(timeout=1)

            frame   = data['frame']
            results = data['results']
            persons = data["persons"]
            bboxes_ = data["bboxes_"]
            data_mapping = data['data_mapping']

            # Perform face detection and recognition for each detected person
            for cropped_frame, person_bbox in zip(persons, bboxes_):
                face_bboxes, landmarks = face_detector.detect(cropped_frame)
                face_recognizer.recognize(cropped_frame, face_bboxes, person_bbox, landmarks, data_mapping)

            # People counting
            person_counter.count(frame, results=results[0], tracker_results=data_mapping, names=face_recognizer.id_face_mapping)

            # If show flag is true, plot the results and display the frame
            if show:
                frame = plot_object.plot( frame   = frame,
                                        tlwhs   = data_mapping["tracking_tlwhs"],
                                        obj_ids = data_mapping["tracking_ids"],
                                        names   = face_recognizer.id_face_mapping)


                cv2.imshow("frame", frame)
                # Press 'Q' on the keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            logger.error(f"error in recognition_process {e}")
            sleep(0.01)
            continue


def main(args):
    """
    Main function that sets up the multiprocessing system and starts the detection and recognition processes.

    Args:
        args (argparse.Namespace): Command-line arguments for display and FPS settings.
    """
    # Set the multiprocessing start method to 'spawn' for CUDA
    set_start_method('spawn', force=True)

    config = Config("configs")
    configs = config.load()

    logger_config           = configs["logger"]
    person_detector_config  = configs["person_detection"]
    person_tracker_config   = configs["tracker"]
    face_detector_config    = configs["detection"]
    face_recognizer_config  = configs["recognition"]
    person_counter_config   = configs["person_counter"]

    logger = Logger(**configs["logger"])
    logger.debug('Application started')

    streamer         =    Streamer(**configs["streamer"], logger=logger)
    fps_object       =    Fps()

    frame_queue = Queue(maxsize=1)
    detection_queue = Queue(maxsize=1)

    detection_process = Process(target=person_detect_tracking, args=(frame_queue, 
                                                                    detection_queue,
                                                                    logger_config,
                                                                    person_detector_config,
                                                                    person_tracker_config), daemon=True)

    recognition_process = Process(target=face_detect_recognize_count, args=(detection_queue,
                                                                            args.show,
                                                                            logger_config,
                                                                            face_detector_config,
                                                                            face_recognizer_config,
                                                                            person_counter_config), daemon=True)

    detection_process.start()
    recognition_process.start()
    
    streamer.initialize()
    try:
        while True:
            if args.fps:
                fps_object.begin_timer()
                fps_object.count_frame()

            frame = next(streamer.read_frame())
            frame_queue.put(frame)

            if args.fps:
                fps = fps_object.calculate_fps()
                logger.info(f"fps:{int(fps)}")

    except KeyboardInterrupt:
        logger.info("Terminating the program...")
    finally:

        if args.show:
            cv2.destroyAllWindows()

        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--show', action='store_true', 
                        help='Enable showing the result of tracker (default is False)')
    parser.add_argument('--fps', action='store_true', 
                        help='Print fps to terminal (default is False)')
    args = parser.parse_args()

    main(args)
