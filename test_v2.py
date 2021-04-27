import argparse
import json
from pp_ocr import file_utils, imgproc
from pp_ocr.text_engine import TextEngine
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def create_app():
    app = Flask(__name__)
    # app.config['folder'] = folder
    # app.config['p'] = p
    return app

def start_engine():
    # start engine
    text_engine = TextEngine(cuda=True)
    return text_engine

def proc(batch_of_images, text_engine):
    cropped_images_of_text = text_engine.detect_and_recognize_text(batch_of_images, padding=0.0, show_time=False,
                                                                    show_images=False,
                                                                    text_confidence_threshold=0.7)

    for item in cropped_images_of_text:
        to_print = list(item)
        points = to_print[2][1]
        xs, ys = [item[0] for item in points], [item[1] for item in points]
        l, t, r, b = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        print('results:',to_print[2][0].shape, [l,t,r,b], to_print[2][2])

""" Read images from folder """
# parser = argparse.ArgumentParser()
# parser.add_argument('--test_folder', default='/media/dh/New Volume/Datasets/Projects/pp/ocr',
#                     type=str, help='folder path to input images')
# parser.add_argument("-p", "--padding", type=float, default=0.1,
#                     help="amount of padding to add to each border of ROI by scaling factor")
# args = parser.parse_args()
# folder = args.test_folder
# p = args.padding
text_engine = start_engine()
# app = create_app(folder, p)
app = create_app()

@app.route('/test')
def test():
    return "Hello World!"

@app.route("/", methods=['POST'])
def index1(text_engine=text_engine):
    print('start')
    # if request.method == 'GET':
    #     return request.get_json()
    img_batch = []
    decode = base64.urlsafe_b64decode(json.loads(request.get_data())['instances'][0])
    buffer = BytesIO(decode)
    img = Image.open(buffer)
    req_data = np.array(img).astype(np.uint8)
    print(req_data)
    if len(req_data.shape) > 3:
        for img in req_data:
            img_batch.append(np.transpose(img, (1,0,2)))
            # print(img.shape)
    else:
        img_batch.append(np.transpose(req_data, (1,0,2)))
        # print(img_batch[0].shape)
    cropped_images_of_text = text_engine.detect_and_recognize_text(img_batch, padding=0.0, show_time=False,
                                                                    show_images=False,
                                                                    text_confidence_threshold=0.7)
    results = {
        'predictions': []
    }
    try:
        for obj in cropped_images_of_text: # cropped_images_of_text is one zip file
            for image, points, text in obj:
                xs, ys = [item[0] for item in points], [item[1] for item in points]
                l, t, r, b = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                # print('results:',to_print[2][0].shape, [l,t,r,b], to_print[2][2])
                temp = {
                    'points': [l,t,r,b],
                    'attributes-value': text
                }
                results['predictions'].append(temp)
    except:
        print('No text')
    print('end')
    # print('Finished!', len(results))
    return jsonify(results)

@app.route("/recognize", methods=['POST'])
def index2(text_engine=text_engine):
    print('start')
    # if request.method == 'GET':
    #     return request.get_json()
    img_batch = []
    decode = base64.urlsafe_b64decode(json.loads(request.get_data())['instances'][0])
    buffer = BytesIO(decode)
    img = Image.open(buffer)
    req_data = np.array(img).astype(np.uint8)
    print(req_data)
    if len(req_data.shape) > 3:
        for img in req_data:
            # print(np.transpose(img, (1,0,2)).shape)
            # img_batch.append(np.transpose(img, (1,0,2)))
            img_batch.append(np.array(img))
            # print(img.shape)
    else:
        # print(np.transpose(img, (1,0,2)).shape)
        # img_batch.append(np.transpose(req_data, (1,0,2)))
        img_batch.append(np.array(img))
        # print(img_batch[0].shape)

    print(np.array(img_batch).shape)
    text = text_engine.recognize_text(text_engine.images_to_tensors(img_batch))
    # cropped_images_of_text = text_engine.detect_and_recognize_text(img_batch, padding=0.0, show_time=False,
    #                                                                 show_images=False,
    #                                                                 text_confidence_threshold=0.7)
    results = {
        'predictions': []
    }
    try:
        temp = {
            'attributes-value': text
        }
        results['predictions'].append(temp)
    except:
        print('No text')
    print('end')
    # print('Finished!', len(results))
    return jsonify(results)

app.run(debug=True, host='0.0.0.0', port=5000)
# image_list, _, _ = file_utils.get_files(folder)


# if __name__ == '__main__':
    
    # # load data from directory
    # batch_of_images = []
    # for k, image_path in enumerate(image_list):
    #     # print("Test image {:d}/{:d}: {:s}".format(k + 1,
    #     #                                           len(image_list), image_path))
    #     # (batch_size, width, height, channel)
    #     batch_of_images.append(imgproc.loadImage(image_path))
        
    
    # proc(batch_of_images, text_engine)
    
    

    # print(list(cropped_images_of_text))
    # print(len(list(cropped_images_of_text)))


