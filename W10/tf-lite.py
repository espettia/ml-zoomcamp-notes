#!/usr/bin/env python
# coding: utf-8

# ### TensorFlow Lite Usage

# !pip install keras-image-helper
# !pip install tflite-runtime
# !pip install tensorflow

# In[1]:

import numpy as np
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception',target_size=(150,150))
image_url = 'http://bit.ly/mlbookcamp-pants'
interpreter = tflite.Interpreter(model_path='clothing-model.tflite')

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.allocate_tensors()

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'skirt',
     't-shirt'
]


def predict(url):
    X = preprocessor.from_url(image_url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



