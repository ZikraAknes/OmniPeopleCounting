import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# YOLO Model
model = YOLO('models/best4.pt')
model.to('cuda')

iou_threshold = 0.6

image_path = os.path.join(os.path.dirname(__file__), 'dataset', 'images')
label_path = os.path.join(os.path.dirname(__file__), 'dataset', 'labels')

TP = 0
FP = 0
TN = 0
FN = 0

# Extract bounding box from model
def extract_bboxes(img):
    boxes = []
    results = model.track(img, persist=True, verbose=False, classes=0, stream_buffer=True, stream=True, tracker="bytetrack.yaml", conf=0.3)

    for result in results:
        bbox_list = result.boxes.data.tolist()
        classes = result.names
        for bbox in bbox_list:
            if len(bbox) != 7:
                continue

            box = [int(data) for data in bbox[:4]]
            boxes.append(box)
     
    return boxes

def label_img(img, bbox, color):
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    # text = 'Person - ' + bbox_id

    # (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)

    # Create bounding box
    # cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x1 + w, bbox_y1 - h), (255, 252, 46), -1)
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, 1)
    # cv2.drawMarker(img, (int(bbox_x2), int(bbox_y2)), (0, 0, 255), thickness=1)
    # cv2.putText(img, text, (bbox_x1, bbox_y1), font, font_size, (255, 255, 255), font_thickness)

    return img

for file in tqdm(os.listdir(image_path)):
    current_TP = 0
    current_FP = 0
    current_TN = 0
    current_FN = 0

    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))

    bboxes = extract_bboxes(image)

    for bbox in bboxes:
        image = label_img(image, bbox, (255, 252, 46))

    label = os.path.join(label_path, file.replace('.jpg', '.txt'))

    f = open(label, "r")
    coordinates_list = f.read().split('\n')

    if len(bboxes) == 0 and len(coordinates_list) == 0:
        TN += 1
        current_TN += 1

    for coordinates in coordinates_list:
        center_x, center_y, lab_w, lab_h = [int(image.shape[0]*float(data)) for data in coordinates.split(' ')[1:5]]
        lab_x1 = int(center_x - (lab_w/2))
        lab_x2 = int(center_x + (lab_w/2))
        lab_y1 = int(center_y - (lab_h/2))
        lab_y2 = int(center_y + (lab_h/2))
        
        iou_list = []

        for bbox in bboxes:
            det_x1, det_y1, det_x2, det_y2 = bbox
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
            else:
                iou_list.append(0)
        
        if len(iou_list) == 0:
            FN += 1
            current_FN += 1
        elif max(iou_list) >= iou_threshold:
            TP += 1
            current_TP += 1
            del bboxes[np.argmax(iou_list)]
        elif max(iou_list) < 0.1:
            FN += 1
            current_FN += 1
        else:
            FN += 1
            current_FN += 1
            del bboxes[np.argmax(iou_list)]

        image = label_img(image, [lab_x1, lab_y1, lab_x2, lab_y2], (0, 0, 255))

    if len(bboxes) != 0:
        FP += len(bboxes)
        current_FP += 1 

    # if current_FP != 0:
    #     print('TP =', current_TP)
    #     print('FP =', current_FP)
    #     print('TN =', current_TN)
    #     print('FN =', current_FN)

    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)

print('\nTP =', TP)
print('FP =', FP)
print('TN =', TN)
print('FN =', FN)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * ((precision * recall)/(precision + recall))

print('\nPrecision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)