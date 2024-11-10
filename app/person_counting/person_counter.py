import threading
import numpy as np
import supervision as sv
from supervision import Detections
from supervision.geometry.core import Point
from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import Counter
from multiprocessing import Queue

@dataclass
class ObjectCounter:
    line_start: tuple
    line_end: tuple
    is_activate : bool
    logger: Any
    log_queue : Queue
    inside_office: Counter  = field(default_factory=Counter)
    entered: List[str]      = field(default_factory=list)
    exited: List[str]       = field(default_factory=list)
    unrecognized_entries: Dict[int, threading.Timer] = field(default_factory=dict)

    def __post_init__(self):
        self.tracker_id_to_name = {}

        self.line_start = Point(self.line_start[0], self.line_start[1])
        self.line_end = Point(self.line_end[0], self.line_end[1])

        self.box_annotator          = sv.BoxAnnotator()
        self.line_zone              = sv.LineZone(start=self.line_start, end=self.line_end, triggering_anchors=[sv.Position.BOTTOM_CENTER])
        self.line_zone_annotator    = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
        self.label_annotator        = sv.LabelAnnotator(text_thickness=4, text_scale=2)
        self.trace_annotator        = sv.TraceAnnotator(thickness=4)
        self.logger.debug("ObjectCounter class initialized")

    def count(self, frame: np.ndarray, results, tracker_results, names: dict):
        if self.is_activate == False:
            return
        detections = Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int),
        )
        if len(detections) == 0 or len(tracker_results["tracking_bboxes"]) == 0:
            return

        if len(tracker_results["tracking_bboxes"]) > 0:
            detections.tracker_id = np.array(tracker_results["tracking_ids"], dtype=int)
            detections.xyxy = np.array(tracker_results["tracking_bboxes"], dtype=int)
            detections.confidence = np.array(tracker_results["tracking_scores"], dtype=np.float32)

        annotated_frame = self.trace_annotator.annotate(
            scene=frame,
            detections=detections)
        crossed_in, crossed_out = self.line_zone.trigger(detections=detections)

        self.determine_entries_exits(names, tracker_results["tracking_ids"], crossed_in, crossed_out)
        self.line_zone_annotator.annotate(frame=annotated_frame, line_counter=self.line_zone)

    def determine_entries_exits(self, names: Dict[int, str], obj_ids, crossed_in: np.ndarray, crossed_out: np.ndarray):
        for i, (cross_in, cross_out) in enumerate(zip(crossed_in, crossed_out)):
            tracker_id = obj_ids[i]
            if tracker_id not in names:
                names[tracker_id] = "UN_KNOWN"

            current_name = names[tracker_id]
            if tracker_id in self.tracker_id_to_name:
                previous_name = self.tracker_id_to_name[tracker_id]
                if (previous_name != current_name) and current_name != "UN_KNOWN":
                    self._update_name(tracker_id, previous_name, current_name)
                    if tracker_id in self.unrecognized_entries:
                        self.unrecognized_entries[tracker_id].cancel() # Cancel old timer
                        del self.unrecognized_entries[tracker_id]
                        self.logger.info(f"{current_name} went in.")
                        tmp = f"{current_name} went in."
                        self.log_queue.put(tmp)
                        continue  # No need to wait 5 seconds as the person is recognized
                else:
                    current_name = previous_name
            else:
                self.tracker_id_to_name[tracker_id] = current_name

            if cross_in:
                if current_name not in self.inside_office or self.inside_office[current_name] == 0:
                    self.entered.append(current_name)
                self.inside_office[current_name] += 1

                if current_name == "UN_KNOWN":
                    if tracker_id in self.unrecognized_entries:
                        self.unrecognized_entries[tracker_id].cancel()
                    
                    timer = threading.Timer(3.0, self._log_unrecognized, [tracker_id, current_name])
                    self.unrecognized_entries[tracker_id] = timer
                    timer.start()
                else:
                    self.logger.info(f"{current_name} went in.")
                    tmp = f"{current_name} went in."
                    self.log_queue.put(tmp)

            elif cross_out:
                if self.inside_office.get(current_name, 0) > 0:
                    self.exited.append(current_name)
                    self.inside_office[current_name] -= 1
                    self.logger.info(f"{current_name} went out.")
                    tmp = f"{current_name} went out."
                    self.log_queue.put(tmp)

    def _log_unrecognized(self, tracker_id, current_name):
        if tracker_id in self.tracker_id_to_name and self.tracker_id_to_name[tracker_id] == "UN_KNOWN":
            self.logger.info(f"{self.tracker_id_to_name[tracker_id]} went in.")
            tmp = f"{self.tracker_id_to_name[tracker_id]} went in."
            self.log_queue.put(tmp)
            del self.unrecognized_entries[tracker_id]

    def _update_name(self, tracker_id, previous_name, current_name):
        # If the person entering is registered as ‘UN_KNOWN’ and then recognised as ‘current_name’, the name is updated.
        if previous_name in self.inside_office:
            self.inside_office[previous_name] -= 1
            self.inside_office[current_name] = self.inside_office.get(current_name, 0) + 1
        self.tracker_id_to_name[tracker_id] = current_name

    def get_current_population(self):
        return sum(count for count in self.inside_office.values() if count > 0)

    def get_entered_people(self):
        return self.entered

    def get_exited_people(self):
        return self.exited

    def get_who_in_the_office(self):
        return {k: v for k, v in self.inside_office.items() if v > 0}
