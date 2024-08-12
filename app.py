import time
import cv2
import argparse
import sys

from app.face_detection.scrfd.face_detector import Face_Detector
from app.face_tracking.byte_tracker import BYTETracker
from app.face_recognition.arcface.recognize import Face_Recognize
from app.config import Config
from app.fps import Fps
from app.utils import plot
from app.logger import Logger
from app.streamer import Streamer

def main(args):

    loguru = Logger('app_logger', 'logs/app.log', 'DEBUG').get_logger()
    loguru.info('Application started')
    data_mapping = {
        "tracking_ids": None,
        "tracking_bboxes": None,
        "tracking_tlwhs" : None,
    }

    #TODO try exceptler burada olmayacak
    streamer = Streamer(True, 0, loguru, 640, 480)
    streamer.initialize()
        
    config = Config("configs")
    attributes = config.load(args.env)
    loguru.info(f'Configuration loaded for environment: {args.env}')

    face_recognizer = Face_Recognize(**attributes["recognition"])
    face_detector = Face_Detector(**attributes["detection"])
    face_tracker = BYTETracker(**attributes["tracker"])
    fps_object = Fps()
    
    while True:
        try:
            frame = next(streamer.read_frame())
            print(frame.shape)
        except StopIteration:
            # Handle the case where the generator is exhausted or ends
            print("No more frames from streamer.")
            break
        except Exception as e:
            # Handle other potential errors (e.g., frame retrieval issues)
            print(f"Error retrieving frame: {e}")
            continue


        #TODO frame gelmezse veya bir hata alırsa kamera kapanıp tekrar açılabilir bir yapıda olacak
        #Alttaki iki işlem method içerisinde yapılacak
        fps_object.frame_id += 1
        fps_object.start_timer = time.time()

        outputs, bboxes, landmarks = face_detector.detect_tracking(frame)
        data_mapping = face_tracker.track(outputs, frame.shape[0], frame.shape[1])
        face_recognizer.recognize(frame ,bboxes, landmarks, data_mapping, face_tracker.is_tracker_available)
        #TODO end time işlemi class içerisinde
        fps_object.end_timer = time.time()
        fps = fps_object.calculate_fps()

        if args.fps:
            #TODO Logger olacak
            print("fps: ",fps)

        if args.show:
            #TODO Guarded clauses
            """if bot args.show:
                    continue"""
            
            frame = plot(frame=frame,
                                tlwhs=data_mapping["tracking_tlwhs"],
                                obj_ids=data_mapping["tracking_ids"],
                                frame_id=fps_object.frame_id,
                                fps=fps,
                                names=face_recognizer.id_face_mapping,
                                bboxes=bboxes,
                                landmarks=landmarks
                                )

            cv2.imshow("frame", frame)
            # Press 'Q' on the keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        
if __name__ == "__main__":
    def video_source_type(value):
        try:
            return int(value)
        except ValueError:
            return str(value)
    #TODO video source olmayacak
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--video-source', type=video_source_type, default=0, 
                        help='Video source (0: webcam, use integer for file path)')
    parser.add_argument('--show', action='store_true', 
                        help='Enable showing the result of tracker (default is False)')
    parser.add_argument('--env', type=str, required=True, 
                        help='Specify the environment (e.g., local, prod)')
    parser.add_argument('--fps', action='store_true', 
                        help='Print fps to terminal (default is False)')
    args = parser.parse_args()

    main(args)
