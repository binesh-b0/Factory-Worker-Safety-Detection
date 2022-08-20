import cv2
import numpy as np
import tensorflow as tf
from detector import DetectorAPI
from classify import predict_safety, get_prep_img
import time


class Prediction(object):
    def __init__(self):
        self.odapi = DetectorAPI(path_to_ckpt='./faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
        self.threshold = 0.7
        self.model = tf.keras.models.load_model('classify_model')

    def impose_prediction(self, img):
        print("PREDICTING")
        frame = np.array(img)
        # Convert RGB to BGR
        frame = frame[:, :, ::-1].copy()
        frame = cv2.resize(frame, (640, 480))
        if time.time() % 3==0:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes, num = self.odapi.processFrame(frame)

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                prep_img = get_prep_img(frame)
                classified_label = predict_safety(prep_img, self.model)
                if classified_label == 'SAFE':
                    box_color = (0, 255, 0)
                else:
                    box_color = (255, 0, 0)
                cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(box[0] + box[2]), round(box[1] + box[3])),
                              box_color, 2)
                cv2.putText(frame, classified_label, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            box_color, 2)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return cv2image

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2image
