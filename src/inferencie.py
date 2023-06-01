# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch

from .core import Detection


class YoloDetector:
    def __init__(self, path_model: str) -> None:
        """
        Initializes an instance of YoloDetector class.
        """
        self.log = getLogger(__name__)
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.log.info(f"Using Device: {self.device}")
        self.load_model(path_model)

    def load_model(self, path_model: str) -> None:
        """
        Loads the YOLO model for object detection.
        """
        self.model = YOLO(path_model)
        self.model.to("cuda")

    def detect_object(self, frame: np.ndarray) -> Detection:
        """
        Detects objects in a given frame using the YOLO model.

        Args:
            frame (np.ndarray): The input frame for object detection.

        Returns:
            Detection.
        """
        results = self.model(
            frame,
            imgsz=320,
            device=self.device,
            conf=0.2,
            classes=[0],
            line_width=1,
            show=False,
            agnostic_nms=True,
        )[0]
        detections = sv.Detections.from_yolov8(results)

        return Detection(box=detections.xyxy, score=detections.confidence)
