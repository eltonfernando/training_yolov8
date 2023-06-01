# -*- coding: utf-8 -*-
import cv2
from numpy import ndarray


class Detection:
    def __init__(self, box: ndarray = None, score: ndarray = None):
        self.status = False if len(score) == 0 else True
        self.box = box
        self.score = score

    def get_center_box(self):
        center_box = []
        if self.status:
            for x_min, y_min, x_max, y_max in self.box:
                center_box.append([(x_max - x_min) / 2, (y_max - y_min) / 2])
        return center_box

    def draw(self, image: ndarray):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)
        thickness = 2

        for box, score in zip(self.box, self.score):
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(
                image,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                color,
                thickness,
            )
            cv2.putText(
                image,
                " " + str(score),
                (int(x_min), int(y_min) - 5),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def __repr__(self):
        return f"Detection(box={self.box}, score={self.score},status {self.status})"
