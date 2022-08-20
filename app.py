from sys import stdout
import logging
from mk import Prediction
import cv2
import os
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from cam import Camera

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
app.config['PORT'] = int(os.environ.get("PORT", 8082))
socketio = SocketIO(app,cors_allowed_origins=[ "https://flask-fire-zsob7rfega-ue.a.run.app" "http://127.0.0.1:8082","https://flask-fire-zsob7rfega-ue.a.run.app/index", "*"])
camera = Camera(Prediction())

detection = False


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    if detection==True:
        camera.enqueue_input(input)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index/', methods=['GET', 'POST'])
def index():
    global detection
    if request.method == "POST":
        try:
            detection = not detection
            camera.cam_flush()
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
        frame = camera.get_frame()
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
    socketio.run(app, host='0.0.0.0', port=8082)
