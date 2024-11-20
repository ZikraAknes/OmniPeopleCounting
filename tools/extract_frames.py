import cv2
import os

cap = cv2.VideoCapture("C:/Users/zikra/OneDrive/Pictures/Camera Roll/WIN_20241115_10_51_34_Pro.mp4")

i = 0
j = 171

fps = 15
start_sec = 11221

while cap.isOpened():
    ret, frame = cap.read()

    center_point = [1048, 659]

    closest = min([center_point[0], center_point[1], frame.shape[1]-center_point[0], frame.shape[0]-center_point[1]])

    frame = frame[center_point[1]-closest:center_point[1]+closest, center_point[0]-closest:center_point[0]+closest]

    print(fps*start_sec, i)

    if not ret or j > 370:
        break

    if i>=start_sec and i%10 == 0:
        while(os.path.isfile('extracted_frames/images/omni_video_' + str(j) + '.jpg')):
            j+=1    
        if j>280:
            break
        filename = 'extracted_frames/images/omni_video_' + str(j) + '.jpg'
        cv2.imwrite(filename, frame)
        j+=1

    cv2.imshow('images', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        break

    i+=1