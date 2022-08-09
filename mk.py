from PIL import Image
import cv2
import numpy as np
from tensorflow import keras
from classify import get_prep_img, predict_safety
from detector import DetectorAPI


class Prediction(object):
    def __init__(self):
        self.odapi = DetectorAPI(path_to_ckpt='./faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
        self.threshold = 0.7

    def impose_prediction(self, img):
        frame = np.array(img)
        # Convert RGB to BGR
        frame = frame[:, :, ::-1].copy()
        frame = cv2.resize(frame, (640, 480))

        boxes, scores, classes, num = self.odapi.processFrame(frame)

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return cv2image

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2image
