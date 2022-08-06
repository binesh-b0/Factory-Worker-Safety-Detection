from PIL import Image
import cv2
import numpy as np
from tensorflow import keras
from classify import get_prep_img, predict_safety


class Prediction(object):
    def __init__(self):
        self.model = keras.models.load_model('classify_model')
        self.classes = None
        with open('coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        pass

    def impose_prediction(self, img):

        Width = 640
        Height = 480

        frame = np.array(img)
        # Convert RGB to BGR
        frame= frame[:, :, ::-1].copy()
        frame = cv2.resize(frame, (640, 480))

        self.net.setInput(cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False))

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

        for i in indices:
            i = i
            box = boxes[i]
            if class_ids[i] == 0:
                lb = str(self.classes[class_id])
                prepped_img = get_prep_img(frame)
                # predicted_val = predict_safety(prepped_img, self.model)
                predicted_val ='SAFE'

                if predicted_val == 'SAFE':
                    box_color = (0, 255, 0)
                else:
                    box_color = (255, 0, 0)
                cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(box[0] + box[2]), round(box[1] + box[3])),
                              box_color, 2)
                cv2.putText(frame, predicted_val, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            box_color, 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return cv2image