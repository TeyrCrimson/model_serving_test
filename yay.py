import os
import time
import numpy as np

from flask import Flask, request, jsonify

import json
import cv2
from PIL import Image

def create_app():
    app = Flask(__name__)
    # app.config['folder'] = folder
    # app.config['p'] = p
    return app

def process_video(video_file):
    print(os.listdir('.'), video_file)
    cap = cv2.VideoCapture(video_file)
    print('cap', cap.isOpened())
    frame_no = 0

    while(cap.isOpened()):
        print('Reading...', cv2.waitKey(1))
        print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = cap.read()
        print(frame)
        if ret:
            frame_no += 1
            cv2.waitKey(1)
        else:
            print('Breaking...')
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame_no

app = create_app()

@app.route('/test')
def test():
    return "Hello World!"

@app.route("/", methods=['POST'])
def index():
    print('START')
    print('You sent: ', request.get_data())
    no_frames = process_video(request.get_data())
    print('FINISHED!', no_frames)
    return jsonify(no_frames)

app.run(debug=True, host='0.0.0.0', port=5000)