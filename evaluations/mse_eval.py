import os
import cv2
from ultralytics import YOLO

# YOLO Model
model = YOLO('models/best11.pt')
model.to('cuda')

dataset_path = 'D:/Omni Dataset/test'

image_path = os.path.join(dataset_path, 'images')
label_path = os.path.join(dataset_path, 'labels')

def extract_bboxes(img, thresh):
    boxes = []
    results = model.track(img, verbose=False, classes=0, conf=thresh)

    for result in results:
        bbox_list = result.boxes.data.tolist()
        classes = result.names
        for bbox in bbox_list:

            box = [[int(data) for data in bbox[:4]], bbox[5]]
            boxes.append(box)
     
    return boxes

MSE = 0

for file in os.listdir(image_path):

    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path, file))
    bboxes = extract_bboxes(image, 0.5)

    label = os.path.join(label_path, file.replace('.jpg', '.txt'))
    f = open(label, "r")
    coordinates_list = f.read().split('\n')

    actual_num = len(coordinates_list)
    predict_num = len(bboxes)

    print(actual_num, predict_num)

    MSE += (actual_num - predict_num)**2

MSE = MSE / len(os.listdir(image_path))

print(MSE)



    