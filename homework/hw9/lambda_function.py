#!/usr/bin/env python
# coding: utf-8

# ### TensorFlow Lite Usage

# !pip install keras-image-helper
# !pip install tflite-runtime
# !pip install tensorflow

# In[1]:

import numpy as np
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def load_img(url,target_size=(150,150),rescale=1./255,dtype=np.float32):
    img = download_image(url)
    x = prepare_image(img, target_size)
    x = np.asarray(x,dtype=dtype)
    x = x.reshape((1,) + x.shape)
    x = x*rescale
    return x

interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.allocate_tensors()

def predict(url):
    x=load_img(url)
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return float_predictions

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



