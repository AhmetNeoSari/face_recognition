import torch
from typing import Any
import numpy as np
from dataclasses import dataclass

from . import tracker_utils
from .tracker_utils import TrackState, KalmanFilter, STrack

@dataclass
class BYTETracker(object):
    """
    BYTETracker class for tracking objects in video frames.

    Args:
        is_tracker_available (bool)
        match_thresh (float): The threshold for matching tracks.
        track_buffer (int): Buffer size for tracking objects.
        track_thresh (float): Threshold for object tracking.
        fp16 (bool): Whether to use FP16 for computations.
        frame_rate (int): Frame rate of the video.
        min_box_area (int): Minimum area of bounding boxes to be considered.
        aspect_ratio_thresh (float): Threshold for aspect ratio of bounding boxes.
        ckpt (str): Path to the model checkpoint file.
        track_img_size (int): Size of the tracking image.
    """
    is_tracker_available: bool
    match_thresh: float
    track_buffer: int
    track_thresh: float
    fp16: bool
    frame_rate : int
    min_box_area: int
    aspect_ratio_thresh: float
    ckpt: str
    track_img_size : int
    logger : Any

    def __post_init__(self):
        """
        Initialize the BYTETracker object with necessary settings and parameters.
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tracked_stracks = []  # type: list[STrack]
            self.lost_stracks = []  # type: list[STrack]
            self.removed_stracks = []  # type: list[STrack]

            self.frame_id = 0
            self.det_thresh = self.track_thresh + 0.1
            self.buffer_size = int(self.frame_rate / 30.0 * self.track_buffer)
            self.max_time_lost = self.buffer_size
            self.kalman_filter = KalmanFilter()

            self.data_mapping = {
                "tracking_ids": [],
                "tracking_bboxes": [],
                "tracking_tlwhs" : [],
            }  
            
            self.logger.info("BYTETracker initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error during BYTETracker initialization: {e}")


    def track(self, outputs:torch.Tensor, img_height:int, img_width:int):
        """
        Track objects in a frame and update tracking information.

        Args:
            outputs (torch.Tensor): Detection outputs from the model.
            fps (float): Frames per second of the video.
            id_face_mapping (dict): Mapping of face IDs to names.

        Returns:
            tuple: A tuple containing the tracked image and data mapping.
                - data_mapping (dict): Updated data mapping with tracking IDs and bounding boxes.
        """
        if self.is_tracker_available == False:
            return

        try:
            tracking_tlwhs = []
            tracking_ids = []
            tracking_scores = []
            tracking_bboxes = []

            if outputs is not None:
                online_targets = self.update(
                    outputs, img_height, img_width, self.track_img_size
                )
                for i in range(len(online_targets)):
                    t = online_targets[i]
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                        x1, y1, w, h = tlwh
                        tracking_tlwhs.append(tlwh)
                        tracking_ids.append(tid)
                        tracking_scores.append(t.score)
                        tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                
            self.data_mapping["tracking_tlwhs"] = tracking_tlwhs
            self.data_mapping["tracking_ids"] = tracking_ids
            self.data_mapping["tracking_bboxes"] = tracking_bboxes

        except Exception as e:
            self.logger.error(f"Error during tracking: {e}")
            return


    def update(self, output_results:torch.Tensor, img_height:int, img_width:int, img_size:tuple):
        """
        Update the tracking information based on the detection results.

        Args:
            output_results (torch.Tensor): Detection results from the model.
            img_info (list): Information about the image dimensions.
            img_size (tuple): Size of the tracking image.

        Returns:
            list: Updated list of tracked objects.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        try:
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                output_results = output_results.cpu().numpy()
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2

        except Exception as E:
            self.logger.error(f"error when creating scores and bboxes {E}")

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second.to(torch.bool)]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second.to(torch.bool)]

        if len(dets) > 0:
            """Detections"""
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = tracker_utils.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = tracker_utils.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = tracker_utils.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = tracker_utils.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = tracker_utils.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = tracker_utils.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


    def joint_stracks(self, tlista, tlistb):
        """
        Combine two lists of tracks, avoiding duplicates.

        Args:
            tlista (list): The first list of tracks.
            tlistb (list): The second list of tracks.

        Returns:
            list: Combined list of tracks.
        """
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res


    def sub_stracks(self, tlista, tlistb):
        """
        Subtract one list of tracks from another.

        Args:
            tlista (list): The list of tracks to subtract from.
            tlistb (list): The list of tracks to be subtracted.

        Returns:
            list: The resulting list of tracks after subtraction.
        """
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())


    def remove_duplicate_stracks(self, stracksa, stracksb):
        """
        Remove duplicate tracks from two lists.

        Args:
            stracksa (list): The first list of tracks.
            stracksb (list): The second list of tracks.

        Returns:
            tuple: Two lists of tracks after removing duplicates.
                - list: Tracks from the first list without duplicates.
                - list: Tracks from the second list without duplicates.
        """
        pdist = tracker_utils.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = [t for i, t in enumerate(stracksb) if not i in dupb]
        return resa, resb
