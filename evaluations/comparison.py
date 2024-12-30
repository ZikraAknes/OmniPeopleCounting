import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import auc
from datetime import datetime
from tensorflow.lite.python.interpreter import Interpreter

# YOLO Model
model = YOLO('models/best11.pt')
model.to('cuda')

dataset_path = 'D:/Omni Dataset/full_dataset'

image_path = os.path.join(dataset_path, 'images')
label_path = os.path.join(dataset_path, 'labels')

# precision_list = []
# recall_list = []
# f1_list = []

# eval_metrics = {
#     'TP': [],
#     'FP': [],
#     'TN': [],
#     'FN': []
# }

# FP_pic = []

image_list = []

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

def extract_bboxes_ssd(img, thresh):
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

# Extract bounding box from model
def extract_bboxes_yolo(img, thresh):
    boxes = []
    results = model.track(img, verbose=False, classes=0, conf=thresh)

    for result in results:
        bbox_list = result.boxes.data.tolist()
        classes = result.names
        for bbox in bbox_list:

            box = [[int(data) for data in bbox[:4]], bbox[5]]
            boxes.append(box)
     
    return boxes

def label_img(image, bbox, color):
    img = np.copy(image)

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, 1)

    return img

def run_eval(iou_thresh, extract_bboxes, file):
    print(f"\nIOU_THRESHOLD: {iou_thresh}")

    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))

    bboxes = extract_bboxes(image, iou_thresh)

    for bbox in bboxes:
        image = label_img(image, bbox[0], (225, 0, 0))

    label = os.path.join(label_path, file.replace('.jpg', '.txt'))

    f = open(label, "r")
    coordinates_list = f.read().split('\n')

    for coordinates in coordinates_list:
        center_x, center_y, lab_w, lab_h = [int(image.shape[0]*float(data)) for data in coordinates.split(' ')[1:5]]
        
        lab_x1 = int(center_x - (lab_w/2))
        lab_x2 = int(center_x + (lab_w/2))
        lab_y1 = int(center_y - (lab_h/2))
        lab_y2 = int(center_y + (lab_h/2))

        image = label_img(image, [lab_x1, lab_y1, lab_x2, lab_y2], (0, 0, 255))

    image_list.append(image)

run_eval(0.65, extract_bboxes_yolo, 'omni_video_296.jpg')
run_eval(0.2, extract_bboxes_ssd, 'omni_video_296.jpg')

cv2.imshow('yolo', image_list[0])
cv2.imshow('ssd', image_list[1])
cv2.waitKey(0)

cv2.imwrite('evaluations/output image/omni_video_296_yolo.jpg', image_list[0])
cv2.imwrite('evaluations/output image/omni_video_296_ssd.jpg', image_list[1])