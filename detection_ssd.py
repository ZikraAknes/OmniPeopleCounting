# Script to run custom TFLite model on test images to detect objects
# Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py

# Import packages
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

# Font Variables
font = cv2.FONT_HERSHEY_DUPLEX
font_thickness = 1
font_size = 0.5

dataset_path = 'D:/Omni Dataset/valid'

image_path = os.path.join(dataset_path, 'images')
label_path = os.path.join(dataset_path, 'labels')

precision_list = []
recall_list = []
f1_list = []

eval_metrics = {
    'TP': [],
    'FP': [],
    'TN': [],
    'FN': []
}

FP_pic = []

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

# Label the image input
def label_img(img, bbox, num):
    (bbox_x1, bbox_y1, bbox_x2, bbox_y2), bbox_id = bbox

    text = 'Person - ' + str(num)

    (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)

    # Create bounding box
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x1 + w, bbox_y1 - h), (46, 100, 255), -1)
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (46, 100, 255), 3)
    # cv2.circle(img, (int(bbox_x2), int(bbox_y2)), 3, (0, 0, 255), thickness=-1)
    cv2.putText(img, text, (bbox_x1, bbox_y1), font, font_size, (255, 255, 255), font_thickness)

    return img


frame = cv2.imread('D:/Omni Dataset/test/images/omni_video_616.jpg') 

i = 1

for bbox in extract_bboxes(frame, 0.25):
    frame = label_img(frame, bbox, i)
    i+=1

cv2.imshow("image", frame)
cv2.waitKey(0)