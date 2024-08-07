import os.path 
import time
import sys

import cv2
import numpy as np
import onnxruntime
import torch
import argparse
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
app_dir = os.path.dirname(parent_dir)
sys.path.append(app_dir)
sys.path.append(current_dir)

from config import Config

@dataclass
class Face_Detector:
    """
    This class detects human face using Class SCRFD model

    The Face_Detector class detects human faces using an SCRFD model provided in ONNX format.
    This class initializes the model, performs various preprocessing steps,
    and returns detection results. The model might have different capabilities
    (e.g., detecting key points). The class includes methods for processing and sorting detection results.

    Args:
        model_file   (str)  : Path to the ONNX format model file.
        taskname     (str)  : Name of the task (default is "detection").
        batched      (bool) : Indicates whether the model supports batched inputs.
        nms_thresh   (int)  : Threshold value for Non-Maximum Suppression (NMS).
        center_cache (dict) : Cached values of anchor centers.
        session      (onnxruntime.InferenceSession) : ONNX Runtime session.
        detect_thresh (float) : Threshold value for detection.
        detect_input_size (tuple[int,int]) : Input size expected by the model.
        max_num (int) : Maximum number of objects to detect.
        metric  (str) : Metric for object ranking ("default" or "max").
        scalefactor (float) : "The image is resized to match the model's input dimensions."
    """
    model_file  : str
    taskname    : str 
    batched     : bool
    nms_thresh  : int
    center_cache: dict
    session     : onnxruntime.InferenceSession
    detect_thresh : float
    detect_input_size : tuple[int,int]
    max_num     : int
    metric      : str
    scalefactor : float

    def __post_init__(self):
        """
        Initializes the model session and variables.

        The __post_init__ method is called automatically after the class is initialized.
        It creates an ONNX Runtime session if a model file is provided 
        and initializes various variables needed for the model.
        """

        if self.session == "":
            assert self.model_file is not None
            assert os.path.exists(self.model_file)
            assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()

            providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
            self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)

        self._init_vars()


    def _init_vars(self):
        """
        Initializes configuration variables for the model.

        The _init_vars method analyzes the model's input and output shapes to initialize various configuration variables.
        It determines model features based on the number of outputs (e.g., whether key points are detected).
        """

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if not isinstance(input_shape[2], str):
            self.detect_input_size = tuple(input_shape[2:4][::-1])            
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True


    def distance2kps(self, points: np.ndarray, distance: np.ndarray, max_shape=None):
        """
        Decode distance prediction to bounding box.
        Args:
            points (numpy.ndarray): Shape (n, 2), [x, y].
            distance (numpy.ndarray): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor (numpy.ndarray): Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)


    def distance2bbox(self, points: np.ndarray, distance: np.ndarray, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (numpy.ndarray): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (numpy.ndarray): Shape of the image.

        Returns:
            (numpy.ndarray): Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


    def forward(self, img:np.ndarray):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, self.scalefactor, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.detect_thresh)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = self.distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list


    def preprocess_image(self):
        """
        This function adapts the processed image to the input dimensions of the model.

        Return:
            det_img     (numpy.ndarray): Rendered image, suitable for the input dimensions of the model.
            det_scale   (float): Scale factor for resizing the image.
        """
        assert self.detect_input_size is not None

        im_ratio = float(self.image.shape[0]) / self.image.shape[1]
        model_ratio = float(self.detect_input_size[1]) / self.detect_input_size[0]
        if im_ratio > model_ratio:
            new_height = self.detect_input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.detect_input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / self.image.shape[0]
        resized_img = cv2.resize(self.image, (new_width, new_height))
        det_img = np.zeros((self.detect_input_size[1], self.detect_input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        return det_img, det_scale


    def nms(self, dets):
        """
        Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
        
        Args:
            dets (numpy.ndarray): Array of shape (N, 5) where N is the number of detections.
                Each row contains [x1, y1, x2, y2, score]:
                - (x1, y1): Coordinates of the top-left corner of the bounding box
                - (x2, y2): Coordinates of the bottom-right corner of the bounding box
                - score: Confidence score for the bounding box
        
        Returns:
            keep (list): Indices of bounding boxes to keep after applying NMS.
        """
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


    def detect(self,image):
        """
        Detects faces in an image and returns the results.

        The detect method resizes the image to match the model's input size,
        performs inference with the model, and processes the detection results.
        It applies Non-Maximum Suppression (NMS) to filter the results and returns the top detections.
        If key points are detected, they are also returned.

        Args:
            image (np.ndarray) : Image on which detection will be performed.
        Returns:
            bboxes    (np.ndarray): Bounding boxes of the detected faces.
            landmarks (np.ndarray): Key points of the detected faces (if applicable).
        """
        self.image = image
        assert self.detect_input_size is not None

        det_img, det_scale = self.preprocess_image()

        scores_list, bboxes_list, kpss_list = self.forward(det_img)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if self.max_num > 0 and det.shape[0] > self.max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = self.image.shape[0] // 2, self.image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if self.metric == "max":
                values = area
            else:
                values = (
                    area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:self.max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        bboxes = np.int32(det)
        landmarks = np.int32(kpss)

        return bboxes, landmarks


    def detect_tracking(self, image):
        self.image = image
        assert self.detect_input_size is not None
        height, width = image.shape[:2]
        img_info = {"id": 0}
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image


        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(self.detect_input_size[1]) / self.detect_input_size[0]
        if im_ratio > model_ratio:
            new_height = self.detect_input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.detect_input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((self.detect_input_size[1], self.detect_input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        if self.use_kps:
            kpss = np.vstack(kpss_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if self.max_num > 0 and det.shape[0] > self.max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if self.metric == "max":
                values = area
            else:
                values = (
                    area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:self.max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        bboxes = np.int32(det / det_scale)
        landmarks = np.int32(kpss / det_scale)

        return torch.tensor(det), img_info, bboxes, landmarks

    def draw_bboxes_landmarks(self, bboxes, landmarks):

        h, w, c = self.image.shape

        tl = 1 or round(0.002 * (h + w) / 2) + 1  # Line and font thickness
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # Draw bounding boxes and landmarks on the frame
        for i in range(len(bboxes)):
            # Get location of the face
            x1, y1, x2, y2, score = bboxes[i]
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 146, 230), 2)

            # Draw facial landmarks
            for id, key_point in enumerate(landmarks[i]):
                cv2.circle(self.image, tuple(key_point), tl + 1, clors[id], -1)


def video_source_type(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--video-source', type=video_source_type, 
                        help='Video source (0: webcam, use string for file path)')
    parser.add_argument('--show', action='store_true', help='Enable showing the result of tracker')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Disable showing the result of tracker')
    parser.set_defaults(video_source=0 ,show=True)
    args = parser.parse_args()

    config = Config()
    attributes = config.load()

    cap = cv2.VideoCapture(args.video_source)
    detector = Face_Detector(**attributes["detection"])

    while True:
        _, frame = cap.read()
        bboxes, landmarks = detector.detect(image=frame)
        if args.show:
            detector.draw_bboxes_landmarks(bboxes, landmarks)
            cv2.imshow("Face Detection", detector.image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
