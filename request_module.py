from __future__ import print_function

# import fire
import requests
import json
import tensorflow as tf
import os
import numpy as np
import cv2
import time
import base64
from io import BytesIO
from PIL import Image
# from config import * # No boi you stand alone now
# from main import Classifier # Just testing

'''
Worked using 
docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/data/hiPPsaurus_old/classification/models/B6/saved_model,target=/models/model/1 -e MODEL_NAME=model -t tensorflow/serving:latest-gpu

For some reason running the docker still requires a proper conda environment
Probably because of cuda?
'''
def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx:min(ndx + bs, l)]

class Serving_clf(object):
    # change url from 0.0.0.0 to popeyes_tf-serving_1 after testing
    def __init__(self, url="http://0.0.0.0:8501/v1/models/testnet:predict", stream=None, target_size=(240,320)):
        self.url = url
        self.stream = stream
        self.TARGET_SIZE = target_size

    def b64_stringify(self, img_array):
        processed_image = Image.fromarray(img_array)
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        img_str = base64.urlsafe_b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

    def preprocess_images_for_clf(self, images):
        '''
        Resize image to (240,320,3) and divide by 255
        Return a list
        '''
        
        processed_images = []
        
        for image in images:
            try:
                target_reshape = (self.TARGET_SIZE[0], self.TARGET_SIZE[1], image.shape[-1])
            # t0 = time.time()
                processed_image = np.reshape(image, target_reshape)
            except:
                processed_image = image
            # t1 = time.time()
            # x = processed_image.tolist()
            # t2 = time.time()
            # print(x)
            # processed_images.append(x)
            # t3 = time.time()
            # print('prep image time:', (t1 - t0), (t2 - t1), (t3 - t2))
            img_str = self.b64_stringify(processed_image)
            # print(img_str)
            processed_images.append(img_str)
        # input_array = np.array(processed_images)
        # logger.debug(f'Input array shape: {input_array.shape}') 
        return processed_images

    def image_prep(self, paths):
        '''
        TODO
        define your preprocessing here to suit your model input
        :param single path/list of paths to image:
        :return: input to model
        '''
        
        output = []
        if isinstance(paths, str):
            # it means there's only one path...
            paths = [paths]
        for path in paths:
            # Do we even need grayscale now?
            img = tf.keras.preprocessing.image.load_img(path,
                                                        target_size=self.TARGET_SIZE)
            img_arr = tf.keras.preprocessing.image.img_to_array(img)
            # print(np.amax(img_arr))
            output.append(self.b64_stringify(img_arr.astype(np.uint8)))
        # output = tf.expand_dims(output, 0).numpy()
        # output = np.array(output)
        # print(output.shape)
        # print(f'>>>> Input to model: {output}')
        # output = output.tolist()
        return output

    def array_rx(self, obj):
        t0 = time.time()
        if np.amax(obj) != 1:
            prep = self.preprocess_images_for_clf(obj)
            # print(prep.shape)
        output = prep
        t1 = time.time()
        elapsed_time = t1 - t0
        print('array_rx prep time,', elapsed_time)
        # print(len(output))
        return output

    def warmup(self, image='2905052.jpg'):
        # img = image_prep('./2905052.jpg')
        for _ in range(3):
            self.request(image)

    def request(self, obj):
        '''
        Sends API request to TFX
        Returns the result
        '''

        # Check type of input
        # Possible inputs:
        # 1. Single path 
        # 2. Single array (How would this be generated? CV2/VLC?)
        # 3. Multi path
        # 4. Multi array
        # image_prep if paths, array_rx if array

        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU
        
        if isinstance(obj, str) or (isinstance(obj, list) and isinstance(obj[0], str)):
            img = self.image_prep(obj)
        else:
            # Maybe do for multiple non-path images?
            img = self.array_rx(obj)
        
        # img = self.array_rx(obj)

        # print('IMG',img)
        
        # Compose a JSON Predict request
        # t0 = time.time()
        predict_request = json.dumps({
            'instances': img
        })
        headers = {"content-type": "application/json"}
        # t1 = time.time()
        # elapsed_time = t1 - t0
        # print('predict generation time:', elapsed_time)
        t0 = time.time()
        # print(requests.post(url=url, data=predict_request, headers=headers))
        response = requests.post(url=self.url, data=predict_request, headers=headers)
        t1 = time.time()
        elapsed_time = t1 - t0
        print('response time:', elapsed_time)
        print(response.text)

        prediction = response.json()['predictions']


        return prediction

    def predict(self, images):
        result = []
        for this_batch in batch(images, bs=8):
            t0 = time.time()
            batch_res = self.request(this_batch)
            # print('results?',batch_res)
            
            if len(result) > 0:
                result = np.concatenate((result, batch_res),axis=0)
            else:
                result = batch_res
            t1 = time.time()
            elapsed_time = t1 - t0
            print('result generation time:', elapsed_time)
        return result

# def postprocess(prediction):
#     '''
#     Not sure if this is needed, but..
#     Processes output of TFX depending on the model used
#     Modes of postprocessing: Detection/Classification/etc
#     '''

# def predict(inputs):



if __name__ == "__main__":
    import time
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    classifier = Classifier(
        weight_path = '/data/hiPPsaurus_old/classification/models/B6/B6_vertpad_998.h5',
        classes_path = '/data/shipserving/classes.txt',
        clf_thresh = 0.5,
        gpu_mem_limit = 1024,
        max_batch_size = 8
    )
    for n in range(3):
        classifier.predict(np.zeros((1,240,320,3),dtype=np.uint8))
    t0 = time.time()
    pred1 = classifier.predict(np.zeros((8,240,320,3),dtype=np.uint8))
    t1 = time.time()
    elapsed_time = t1 - t0
    print('time 1:', elapsed_time)
    print(pred1)
    
    # test = Serving_clf()
    # # test.warmup()
    # t0 = time.time()
    # pred2 = test.predict(np.zeros((8,240,320,3),dtype=np.uint8))
    # t1 = time.time()
    # elapsed_time = t1 - t0
    # print('time 2:', elapsed_time)
    # print(pred2)
    # test.predict(['2905052.jpg'])


    
    # test1 = np.array([[0.00188430294, 0.00032713628, 9.84914586e-05, 0.000393940456, 7.86710189e-06, 2.81793327e-05, 1.42693179e-05, 0.000200252078, 0.000487478828, 1.82934982e-05, 0.000238345587, 0.994563937, 3.3461587e-05, 0.00010461588, 0.000282816036, 0.000102636666, 0.00110567024, 0.000108313019]])
    # test2 = np.array([[1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]
    #                   [1.8843065e-03 3.2713689e-04 9.8491553e-05 3.9394063e-04 7.8671101e-06 2.8179362e-05 
    #                    1.4269347e-05 2.0025246e-04 4.8747906e-04 1.8293569e-05 2.3834580e-04 9.9456394e-01 
    #                    3.3461620e-05 1.0461588e-04 2.8281627e-04 1.0263676e-04 1.1056719e-03 1.0831333e-04]])
    # test3 = np.array([[0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05,
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332], 
    #                   [0.00188430655, 0.000327136891, 9.84915532e-05, 0.000393940631, 7.86711e-06, 2.81793618e-05, 
    #                    1.4269347e-05, 0.000200252456, 0.000487479061, 1.82935692e-05, 0.000238345805, 0.994563937, 
    #                    3.34616198e-05, 0.00010461588, 0.000282816269, 0.000102636761, 0.00110567186, 0.000108313332]])
    # print(test1 == test2)


