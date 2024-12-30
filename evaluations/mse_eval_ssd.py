import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
import csv
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc
from datetime import datetime

dataset_path = 'D:/Omni Dataset/test'

image_path = os.path.join(dataset_path, 'images')
label_path = os.path.join(dataset_path, 'labels')

interpreter = Interpreter(model_path='models/ssd_model2.tflite')
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def extract_bboxes(img, thresh):
    # Load the Tensorflow Lite model into memory
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imH, imW, _ = img.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    bboxes = []

    for i in range(len(scores)):
        if ((scores[i] > thresh) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            bboxes.append([[xmin, ymin, xmax, ymax], scores[i]])
    
    return bboxes

MSE = 0

for file in os.listdir(image_path):

    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))
    bboxes = extract_bboxes(image, 0.2)

    label = os.path.join(label_path, file.replace('.jpg', '.txt'))
    f = open(label, "r")
    coordinates_list = f.read().split('\n')

    actual_num = len(coordinates_list)
    predict_num = len(bboxes)

    print(actual_num, predict_num)

    MSE += (actual_num - predict_num)**2

MSE = MSE / len(os.listdir(image_path))

print(MSE)



    