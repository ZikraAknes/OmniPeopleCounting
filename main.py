import cv2
import numpy as np
import time
import firebase_admin

from datetime import datetime
from ultralytics import YOLO
from tools.crop_image import CropImage
from tools.dewarp import DewarpImage
from firebase_admin import db

databaseURL = "https://omni-people-default-rtdb.asia-southeast1.firebasedatabase.app"
cred_obj = firebase_admin.credentials.Certificate("omni-people-firebase-adminsdk-mj65k-77786d42ec.json")
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':databaseURL})

ref = db.reference("User123")

# YOLO Model
model = YOLO('models/best8.pt')
model.to('cuda')

# Font Variables
font = cv2.FONT_HERSHEY_DUPLEX
font_thickness = 1
font_size = 0.5

# Center point
center = [488, 428]

# Detcted Persons Id
detected_persons = {}

# Calculate distance between points
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Extract bounding box from model
def extract_bboxes(img):
    boxes = []
    results = model.track(img, persist=True, verbose=False, classes=0, tracker="bytetrack.yaml", conf=0.6)

    for result in results:
        bbox_list = result.boxes.data.tolist()
        classes = result.names
        for bbox in bbox_list:
            if len(bbox) != 7:
                continue

            box = [[int(data) for data in bbox[:4]], str(int(bbox[4])), bbox[5]]
            boxes.append(box)

    ref.child('People Count').set(len(boxes))
     
    return boxes

# Label the image input
def label_img(img, bbox):
    (bbox_x1, bbox_y1, bbox_x2, bbox_y2), bbox_id, bbox_conf = bbox

    text = 'Person - ' + str(detected_persons[bbox_id][4])

    (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)

    # Create bounding box
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x1 + w, bbox_y1 - h), (255, 252, 46), -1)
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (255, 252, 46), 3)
    cv2.drawMarker(img, (int(bbox_x2), int(bbox_y2)), (0, 0, 255), thickness=1)
    cv2.putText(img, text, (bbox_x1, bbox_y1), font, font_size, (255, 255, 255), font_thickness)

    return img

# Draw marker
def draw_marker(center_x, center_y):
    date = datetime.now().strftime('%Y-%m-%d')

    # Read image previous image output
    tracked_img = cv2.imread(f'outputs/{date}.jpg')

    # Draw marker
    cv2.drawMarker(tracked_img, (int(center_x), int(center_y)), (0, 0, 255), thickness=1)

    # Rewrite image output to folder
    cv2.imwrite(f'outputs/{date}.jpg', tracked_img)


# People counting 
def count_people(frame):
    pred_img = np.copy(frame)

    for bbox in extract_bboxes(frame):
        
        (bbox_x1, bbox_y1, bbox_x2, bbox_y2), bbox_id, bbox_conf = bbox

        center_x = bbox_x1 + (bbox_x2 - bbox_x1)/2
        center_y = bbox_y1 + (bbox_y2 - bbox_y1)/2

        if detected_persons.get(bbox_id) == None:
            detected_persons[bbox_id] = [center_x, center_y, False, time.time(), len(detected_persons)+1]
        else:
            movement_dist = distance(center_x, center_y, detected_persons[bbox_id][0], detected_persons[bbox_id][1])
            dist_treshold = 20
            elapsed_time = (time.time() - detected_persons[bbox_id][3]) % 60
            if movement_dist < dist_treshold and not detected_persons[bbox_id][2] and elapsed_time > 5:
                draw_marker(detected_persons[bbox_id][0], detected_persons[bbox_id][1])
                detected_persons[bbox_id][2] = True
            elif movement_dist > dist_treshold:
                detected_persons[bbox_id] = [center_x, center_y, False, time.time(), detected_persons[bbox_id][4]]

        pred_img = label_img(pred_img, bbox)

    return pred_img     

# Retrieve video
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("inputs/Omni3.mp4")

# Get camera area
# cam_area = CropImage.get_cam_area(cap.read()[1])

# Saving current panorama image
date = datetime.now().strftime('%Y-%m-%d')
cv2.imwrite(f'outputs/{date}.jpg', CropImage.crop_center(cap.read()[1], center))
# panorama_img = DewarpImage.create_panorama(cap.read()[1], cam_area)
# cv2.imwrite(f'outputs/{date}.jpg', cap.read()[1][cam_area[0]:cam_area[1], cam_area[2]:cam_area[3]])
# cv2.imwrite(f'outputs/{date}.jpg', cap.read()[1])

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    
    # frame = frame[cam_area[0]:cam_area[1], cam_area[2]:cam_area[3]]
    frame = CropImage.crop_center(frame, center)

    if not ret:
        break

    pred_img = count_people(frame)

    cv2.imshow("image", pred_img)

    FPS = int(1 // (time.time() - start_time))

    print(f"FPS: {FPS}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        break