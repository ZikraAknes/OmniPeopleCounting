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

# YOLO Model
model = YOLO('models/yolo11n_v4_bs32.pt')
model.to('cuda')

tracker = sv.ByteTrack()

dataset_path = 'D:/Omni Dataset/test'

image_path = os.path.join(dataset_path, 'images')
label_path = os.path.join(dataset_path, 'labels')

precision_list = []
recall_list = []
f1_list = []

used_class = 0

eval_metrics = {
    'TP': [],
    'FP': [],
    'TN': [],
    'FN': []
}

FP_pic = []

# Extract bounding box from model
def extract_bboxes(img, thresh):
    boxes = []
    results = model.predict(img, verbose=False, classes=used_class, conf=thresh)

    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        bbox_list = result.boxes.data.tolist()
        # print(len(bbox_list), len(detections.xyxy))
        classes = result.names
        # for i, bbox in enumerate(detections.xyxy):
        for bbox in bbox_list:
            # box = [[int(data) for data in bbox[:4]], detections.tracker_id[i]]
            box = [[int(data) for data in bbox[:4]], bbox[5]]
            boxes.append(box)
     
    return boxes

def label_img(image, bbox, color):
    img = np.copy(image)

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, 1)

    return img

def run_eval(iou_thresh):
    print(f"\nIOU_THRESHOLD: {iou_thresh}")

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for file in tqdm(os.listdir(image_path)):
        current_TP = 0
        current_FP = 0
        current_TN = 0
        current_FN = 0

        image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))

        bboxes = extract_bboxes(image, iou_thresh)

        for bbox in bboxes:
            image = label_img(image, bbox[0], (255, 252, 46))

        label = os.path.join(label_path, file.replace('.jpg', '.txt'))

        f = open(label, "r")
        coordinates_list = f.read().split('\n')

        for coordinates in coordinates_list:
            try:
                class_id, center_x, center_y, lab_w, lab_h = [int(image.shape[0]*float(data)) for data in coordinates.split(' ')[0:5]]
            except:
                if len(bboxes) == 0:
                    TN += 1
                    current_TN += 1
                    break
            
            if class_id != used_class:
                break

            lab_x1 = int(center_x - (lab_w/2))
            lab_x2 = int(center_x + (lab_w/2))
            lab_y1 = int(center_y - (lab_h/2))
            lab_y2 = int(center_y + (lab_h/2))
            
            iou_list = []

            for bbox in bboxes:
                (det_x1, det_y1, det_x2, det_y2), bbox_conf = bbox
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
                del bboxes[np.argmax(iou_list)]

            image = label_img(image, [lab_x1, lab_y1, lab_x2, lab_y2], (0, 0, 255))

        if len(bboxes) != 0:
            FP += len(bboxes)
            current_FP += 1

            if file not in FP_pic:
                FP_pic.append(file)

        # if current_FP != 0:
        # print('\nTP =', current_TP)
        # print('FP =', current_FP)
        # print('TN =', current_TN)
        # print('FN =', current_FN)


        # cv2.imshow('image', image)
        # cv2.waitKey(0)

    print('\nTP =', TP)
    print('FP =', FP)
    print('TN =', TN)
    print('FN =', FN)

    try:
        precision = TP / (TP + FP)
    except:
        precision = 1.0
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall)/(precision + recall))

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    eval_metrics['TP'].append(TP)
    eval_metrics['FP'].append(FP)
    eval_metrics['TN'].append(TN)
    eval_metrics['FN'].append(FN)

    # print('\nPrecision:', precision)
    # print('Recall:', recall)
    # print('F1 Score:', f1)

# thresh_list = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

thresh = 0.90
thresh_list = []

# for thresh in thresh_list:
#     run_eval(thresh)

while(thresh > 0):
    thresh_list.append(thresh)
    run_eval(thresh)
    thresh = round(thresh - 0.05, 2)

# print(precision_list)
# print(recall_list)
# print(FP_pic)

print('Average Precision (AP): {}'.format(auc(recall_list, precision_list)))

date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

with open(f'evaluations/reports/{date}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([['IoU Thresh', 'Precision', 'Recall', 'F1 Score', 'TP', 'FP', 'TN', 'FN']])
    for i in range(len(thresh_list)):
        writer.writerows([[thresh_list[i], 
                          precision_list[i], 
                          recall_list[i],
                          f1_list[i],
                          eval_metrics['TP'][i], 
                          eval_metrics['FP'][i], 
                          eval_metrics['TN'][i], 
                          eval_metrics['FN'][i]]])

plt.plot(recall_list, precision_list)
plt.show()