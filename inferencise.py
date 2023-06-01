# -*- coding: utf-8 -*-
import os
import cv2
from glob import glob
from src import YoloDetector, Detection

path_model = os.path.join(
    "runs", "yolov8m_customv1_pessoa_90_all", "weights", "best.pt"
)
yolo = YoloDetector(path_model)

for path_img in glob("/media/elton/D/dataset_person/90_nova_alta_magic/*.jpg"):
    img = cv2.imread(path_img)

    result: Detection = yolo.detect_object(img)
    result.draw(img)
    cv2.imshow("result", img)
    k = cv2.waitKey(0)
    if k == ord("q"):
        break
