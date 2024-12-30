import cv2
import numpy as np
import time
import firebase_admin
import threading
import supervision as sv
import random

from datetime import datetime
from ultralytics import YOLO
from tools.crop_image import CropImage
from tools.dewarp import DewarpImage
from firebase_admin import db
from tools.centroid_tracker import CentroidTracker

databaseURL = "https://omni-people-default-rtdb.asia-southeast1.firebasedatabase.app"
cred_obj = firebase_admin.credentials.Certificate("omni-people-firebase-adminsdk-mj65k-6898755a38.json")
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':databaseURL})

ref = db.reference("User123")

uploading = False

# YOLO Model
model = YOLO('models/best11.pt')
model.to('cuda')

# tracker = sv.ByteTrack()
tracker = CentroidTracker()

# Font Variables
font = cv2.FONT_HERSHEY_DUPLEX
font_thickness = 1
font_size = 0.5

# Center point
center = [360, 360]

# Detcted Persons Id
detected_persons = {}

tracked_points = {}

# Calculate distance between points
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Extract bounding box from model
def extract_bboxes(img):
    boxes = []
    # results = model.predict(img, verbose=False, classes=0, conf=0.65)
    results = model.predict(img, verbose=False, classes=0, conf=0.65)[0]    

    bbox_list = results.boxes.data.tolist()

    detections = tracker.update(bbox_list)

    # for i, bbox in enumerate(detections.xyxy):
    for i, bbox in enumerate(detections['tracked_bbox']):
        box = [[int(data) for data in bbox[:4]], detections['tracked_id'][i]]
        boxes.append(box)
    
    if(not uploading):
        threading.Thread(target=upload_database, daemon=True, args=[len(boxes)]).start()
     
    return boxes

# Label the image input
def label_img(img, bbox):
    (bbox_x1, bbox_y1, bbox_x2, bbox_y2), bbox_id = bbox

    text = 'Person - ' + str(bbox_id)

    (w, h), _ = cv2.getTextSize(text, font, font_size, font_thickness)

    # Create bounding box
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x1 + w, bbox_y1 - h), (46, 100, 255), -1)
    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (46, 100, 255), 3)
    # cv2.circle(img, (int(bbox_x2), int(bbox_y2)), 3, (0, 0, 255), thickness=-1)
    cv2.putText(img, text, (bbox_x1, bbox_y1), font, font_size, (255, 255, 255), font_thickness)

    return img

# Draw marker
def draw_marker(center_x, center_y):
    date = datetime.now().strftime('%Y-%m-%d')

    # Read image previous image output
    tracked_img = cv2.imread(f'outputs/{date}.jpg')

    # Draw marker
    # cv2.drawMarker(tracked_img, (int(center_x), int(center_y)), (0, 0, 255), thickness=1)
    cv2.circle(tracked_img, (int(center_x), int(center_y)), 3, (0, 0, 255), thickness=-1)

    # Rewrite image output to folder
    cv2.imwrite(f'outputs/{date}.jpg', tracked_img)

def draw_polylines(center_x, center_y, prev_x, prev_y, color):
    date = datetime.now().strftime('%Y-%m-%d')

    # Read image previous image output
    tracked_img = cv2.imread(f'outputs/{date}.jpg')

    # print(center_x, center_y)

    # Draw marker
    # cv2.drawMarker(tracked_img, (int(center_x), int(center_y)), (0, 0, 255), thickness=1)
    # cv2.circle(tracked_img, (int(center_x), int(center_y)), 3, (0, 0, 255), thickness=-1)
    tracked_img = cv2.line(tracked_img, (int(center_x), int(center_y)), (int(prev_x), int(prev_y)), color, 3)

    # Rewrite image output to folder
    cv2.imwrite(f'outputs/{date}.jpg', tracked_img)

# People counting 
def count_people(frame):
    pred_img = np.copy(frame)

    for bbox in extract_bboxes(frame):
        
        (bbox_x1, bbox_y1, bbox_x2, bbox_y2), bbox_id = bbox

        center_x = bbox_x1 + (bbox_x2 - bbox_x1)/2
        center_y = bbox_y1 + (bbox_y2 - bbox_y1)/2

        if detected_persons.get(bbox_id) == None:
            # color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))
            color = (46, 100, 255)
            detected_persons[bbox_id] = [center_x, center_y, False, time.time(), len(detected_persons)+1, center_x, center_y, center_x, center_y, color]
        else:
            movement_dist = distance(center_x, center_y, detected_persons[bbox_id][0], detected_persons[bbox_id][1])
            dist_treshold = 10
            elapsed_time = (time.time() - detected_persons[bbox_id][3]) % 60
            if movement_dist < dist_treshold and not detected_persons[bbox_id][2] and elapsed_time > 5:
                draw_marker(detected_persons[bbox_id][0], detected_persons[bbox_id][1])
                detected_persons[bbox_id][2] = True
            elif movement_dist > dist_treshold:
                detected_persons[bbox_id] = [center_x, center_y, False, time.time(), detected_persons[bbox_id][4], detected_persons[bbox_id][5], detected_persons[bbox_id][6], detected_persons[bbox_id][7], detected_persons[bbox_id][8], detected_persons[bbox_id][9]]
            detected_persons[bbox_id] = [detected_persons[bbox_id][0], detected_persons[bbox_id][1], detected_persons[bbox_id][2], detected_persons[bbox_id][3], detected_persons[bbox_id][4], center_x, center_y, detected_persons[bbox_id][5], detected_persons[bbox_id][6], detected_persons[bbox_id][9]]

        pred_img = label_img(pred_img, bbox)

        # draw_polylines(detected_persons[bbox_id][5], detected_persons[bbox_id][6], detected_persons[bbox_id][7], detected_persons[bbox_id][8], detected_persons[bbox_id][9])

        if tracked_points.get(bbox_id) == None:
            tracked_points[bbox_id] = [[int(center_x), int(center_y)]]
        else:
            tracked_points[bbox_id].append([int(center_x), int(center_y)])

    return pred_img     

def upload_database(people_count):
    uploading = True
    ref.child('People Count').set(people_count)
    uploading = False

# Retrieve video
# cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture("inputs/OMNI_4.mp4")
cap = cv2.VideoCapture("D:/Omni Dataset/video/Scene3A_55.mp4")

# Get camera area
# cam_area = CropImage.get_cam_area(cap.read()[1])

# Saving current panorama image
date = datetime.now().strftime('%Y-%m-%d')
cv2.imwrite(f'outputs/{date}.jpg', CropImage.crop_center(cap.read()[1], center))
# panorama_img = DewarpImage.create_panorama(cap.read()[1], cam_area)
# cv2.imwrite(f'outputs/{date}.jpg', cap.read()[1][cam_area[0]:cam_area[1], cam_area[2]:cam_area[3]])
# cv2.imwrite(f'outputs/{date}.jpg', cap.read()[1])
try:
    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        
        # frame = cv2.imread('D:/Omni Dataset/full_dataset/images/omni_video_616.jpg') 
        
        # frame = frame[cam_area[0]:cam_area[1], cam_area[2]:cam_area[3]]
        frame = CropImage.crop_center(frame, center)
        # frame = DewarpImage.create_panorama(frame)

        if not ret:
            break

        pred_img = count_people(frame)

        cv2.imshow("image", pred_img)

        FPS = int(1 // (time.time() - start_time))

        print(f"FPS: {FPS}")

        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            break
except:
    date = datetime.now().strftime('%Y-%m-%d')

    # Read image previous image output
    tracked_img = cv2.imread(f'outputs/{date}.jpg')

    # print(center_x, center_y)

    # print(tracked_points)

    for data in tracked_points.values():
        # print(data)
    #     print(data)

        # Draw marker
        tracked_img = cv2.polylines(tracked_img, [np.array(data).reshape((-1, 1, 2))], False, (46, 100, 255), 3)

    # # Rewrite image output to folder
    cv2.imwrite(f'outputs/{date}.jpg', tracked_img)
    cv2.imshow('image', tracked_img)
    cv2.waitKey(0)