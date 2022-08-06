import numpy as np
from sys import stdout
import logging
from mk import Prediction
import cv2
import os
from tensorflow import keras
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from cam import Camera
from utils import base64_to_pil_image, pil_image_to_base64
from PIL import Image
from classify import get_prep_img, predict_safety

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
app.config['PORT'] = int(os.environ.get("PORT", 8080))
socketio = SocketIO(app,cors_allowed_origins=["https://factorysafety.azurewebsites.net","http://127.0.0.1:8080","http://10.0.0.182:8080","*"])
camera = Camera(Prediction())

detection = False

@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    # image_data = input # Do your magical Image processing here!!
    # #image_data = image_data.decode("utf-8")
    # image_data = "data:image/jpeg;base64," + image_data
    # print("OUTPUT " + image_data)
    # emit('out-image-event', {'image_data': image_data}, namespace='/test')
    # camera.enqueue_input(base64_to_pil_image(input))

# @socketio.on('connect', namespace='/test')
# def test_connect():
#     app.logger.info("client connected")

@app.route('/', methods=['GET', 'POST'])
@app.route('/index/', methods=['GET', 'POST'])
def index():
    global detection
    if request.method == "POST":
        try:
            detection = not detection
            return redirect(url_for('index'))
        except:
            flash("Invalid type for variable")
        return redirect(url_for('index'))
    return render_template('index.html',detection=detection)



def gen():
    """Video streaming generator function."""
    app.logger.info("starting to generate frames!")
    print("generating frames")
    while True:
        frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=8080)