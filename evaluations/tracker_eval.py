import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import auc
from datetime import datetime
# from bytetracker import BYTETracker
from ultralytics import trackers
from centroid_tracker import CentroidTracker

# YOLO Model
model = YOLO('models/best11.pt')
model.to('cuda')

# tracker = sv.ByteTrack(track_activation_threshold = 0)
# tracker = trackers.byte_tracker.BYTETracker(frame_rate=30)
# Font Variables
font = cv2.FONT_HERSHEY_DUPLEX
font_thickness = 1
font_size = 0.5

dataset_path = 'D:/Omni Dataset/test'

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

tracker = CentroidTracker()

# Calculate distance between points
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Extract bounding box from model
def extract_bboxes(img, thresh):
    boxes = []
    # results = model.predict(img, verbose=False, classes=0, conf=0.65)
    results = model.predict(img, verbose=False, classes=0, conf=thresh)[0]    

    bbox_list = results.boxes.data.tolist()

    detections = tracker.update(bbox_list)

    # for i, bbox in enumerate(detections.xyxy):
    for i, bbox in enumerate(detections['tracked_bbox']):
        
        if bbox[0] < bbox[2]:
            center_x = bbox[0] + ((bbox[2] - bbox[0]) / 2)
            center_y = bbox[1] + ((bbox[3] - bbox[1]) / 2)
        else:
            center_x = bbox[2] + ((bbox[0] - bbox[2]) / 2)
            center_y = bbox[3] + ((bbox[1] - bbox[3]) / 2)

        # box = [[int(data) for data in bbox[:4]], [center_x, center_y], detections.tracker_id[i]]
        box = [[int(data) for data in bbox[:4]], [center_x, center_y], detections['tracked_id'][i]]
        boxes.append(box)
     
    return boxes

def label_img(image, bbox, color, bbox_id=None):
    img = np.copy(image)

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, 1)

    if bbox_id != None:
        text = 'Person - ' + str(bbox_id)
        (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)
        cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x1 + w, bbox_y1 - h), (255, 252, 46), -1)
        cv2.putText(img, text, (bbox_x1, bbox_y1), font, font_size, (255, 255, 255), font_thickness)
        
    return img

def run_eval(iou_thresh):
    print(f"\nIOU_THRESHOLD: {iou_thresh}")

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    GT = 0

    mme = 0

    dist = 0

    for file in tqdm(os.listdir(image_path)):
        current_TP = 0
        current_FP = 0
        current_TN = 0
        current_FN = 0

        image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))

        bboxes = extract_bboxes(image, iou_thresh)

        for bbox in bboxes:
            image = label_img(image, bbox[0], (255, 252, 46), bbox[2])

        label = os.path.join(label_path, file.replace('.jpg', '.txt'))

        f = open(label, "r")
        coordinates_list = f.read().split('\n')

        GT += len(coordinates_list)

        for coordinates in coordinates_list:
            try:
                center_x, center_y, lab_w, lab_h = [int(image.shape[0]*float(data)) for data in coordinates.split(' ')[1:5]]
            except:
                if len(bboxes) == 0:
                    TN += 1
                    current_TN += 1
                    break
            
            lab_x1 = int(center_x - (lab_w/2))
            lab_x2 = int(center_x + (lab_w/2))
            lab_y1 = int(center_y - (lab_h/2))
            lab_y2 = int(center_y + (lab_h/2))
            
            iou_list = []

            for bbox in bboxes:
                (det_x1, det_y1, det_x2, det_y2), _, bbox_id = bbox
                det_h = np.abs(det_y2-det_y1)
                det_w = np.abs(det_x2-det_x1)
                
                x1 = max(min(lab_x1, lab_x2), min(det_x1, det_x2))
                y1 = max(min(lab_y1, lab_y2), min(det_y1, det_y2))
                x2 = min(max(lab_x1, lab_x2), max(det_x1, det_x2))
                y2 = min(max(lab_y1, lab_y2), max(det_y1, det_y2))
                
                if x1<x2 and y1<y2:
                    inter_area = (x2-x1)*(y2-y1)
                    diff_area = (lab_w*lab_h - inter_area) + (det_w*det_h - inter_area)

                    iou_list.append(inter_area/(inter_area+diff_area))
                    # iou_list.append(bbox_conf)
                else:
                    iou_list.append(0)
            
            if len(iou_list) == 0:
                FN += 1
                current_FN += 1
            # elif max(iou_list) >= iou_thresh:
            #     TP += 1
            #     current_TP += 1
            #     del bboxes[np.argmax(iou_list)]
            elif max(iou_list) < 0.001:
                FN += 1
                current_FN += 1
            else:
                # FN += 1
                # current_FN += 1
                # del bboxes[np.argmax(iou_list)]
                TP += 1
                current_TP += 1

                (det_x1, det_y1, det_x2, det_y2), (det_center_x, det_center_y), bbox_id = bboxes[np.argmax(iou_list)]
                
                # dist += distance(center_x, center_y, det_center_x, det_center_y)
                dist += (1 - max(iou_list))

                del bboxes[np.argmax(iou_list)]



            image = label_img(image, [lab_x1, lab_y1, lab_x2, lab_y2], (0, 0, 255))

        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        # mme += int(input("mme_num = "))

    MOTP = dist / TP

    MOTA = 1 - ((FP + FN + mme) / GT)

    print(f"mme: {mme}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"GT: {GT}")
    print(f"MOTA: {MOTA}")

    print(f"Distance: {dist}")
    print(f"TP: {TP}")
    print(f"MOTP: {MOTP}")


run_eval(0.65)