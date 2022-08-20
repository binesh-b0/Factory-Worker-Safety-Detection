import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from classify import get_prep_img, predict_safety
from detector import DetectorAPI
from classify import predict_safety, get_prep_img
import time
from PIL import Image

app = Flask(__name__)

odapi = DetectorAPI(path_to_ckpt='./faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
threshold = 0.7
model = tf.keras.models.load_model('classify_model')
detection = True

# open webcam video stream
cap = cv2.VideoCapture(0)


def impose_prediction(img):
    frame = np.array(img)
    # Convert RGB to BGR
    frame = frame[:, :, ::-1].copy()
    frame = cv2.resize(frame, (640, 480))
    if time.time() % 3 == 0:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores, classes, num = odapi.processFrame(frame)

    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]

            prep_img = get_prep_img(frame[box[0]:box[2],box[1]:box[3]])
            classified_label = predict_safety(prep_img, model)
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


def gen_frames():
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()

        if not success:
            break
        else:
            # resizing for faster detection
            frame = cv2.resize(frame, (640, 480))
            if detection:
                boxes, scores, classes, num = odapi.processFrame(frame)
                frame = impose_prediction(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index/', methods=['GET', 'POST'])
def index():
    global detection
    if request.method == 'POST':
        try:
            detection = not detection
            # if detection ==  False:
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     cv2.waitKey(1)
            return redirect(url_for('index'))
        except:
            flash("Invalid type for variable")
        return redirect(url_for('index'))
    return render_template('index.html',detection=detection)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
