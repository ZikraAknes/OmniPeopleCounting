import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# YOLO Model
model = YOLO('models/best4.pt')
model.to('cuda')

# Font Variables
font = cv2.FONT_HERSHEY_DUPLEX
font_thickness = 1
font_size = 0.5

iou_threshold = 0.6

image_path = os.path.join(os.path.dirname(__file__), '..', 'extracted_frames')
# label_path = os.path.join(os.path.dirname(__file__), 'dataset', 'labels')

TP = 0
FP = 0
TN = 0
FN = 0

# Extract bounding box from model
def extract_bboxes(img):
    boxes = []
    results = model.track(img, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml", conf=0.3)

    for result in results:
        bbox_list = result.boxes.data.tolist()
        # print(len(bbox_list))
        classes = result.names
        for bbox in bbox_list:
            if len(bbox) != 7:
                continue

            box = [[int(data) for data in bbox[:4]], str(int(bbox[4]))]
            boxes.append(box)
     
    return boxes

def label_img(img, bbox, color):
    (bbox_x1, bbox_y1, bbox_x2, bbox_y2), bbox_id = bbox

    text = 'Person - ' + bbox_id

    (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)

    # Create bounding box
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x1 + w, bbox_y1 - h), (255, 252, 46), -1)
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, 1)
    # cv2.drawMarker(img, (int(bbox_x2), int(bbox_y2)), (0, 0, 255), thickness=1)
    cv2.putText(img, text, (bbox_x1, bbox_y1), font, font_size, (255, 255, 255), font_thickness)

    return img

file_list = os.listdir(image_path)
file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in tqdm(file_list):
    current_TP = 0
    current_FP = 0
    current_TN = 0
    current_FN = 0

    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))

    bboxes = extract_bboxes(image)

    for bbox in bboxes:
        image = label_img(image, bbox, (255, 252, 46))

    cv2.imshow('image', image)
    # cv2.waitKey(0)

    input('MOTP: ')