import cv2
import os

cap = cv2.VideoCapture("C:/Users/zikra/OneDrive/Documents/Kuliah/Semester 7/Project Skripsi/OmniPeopleCounting/inputs/OMNI_4.mp4")

i = 0
j = 281

fps = 15
start_sec = 0

while cap.isOpened():
    ret, frame = cap.read()

    center_point = [303, 267]

    closest = min([center_point[0], center_point[1], frame.shape[1]-center_point[0], frame.shape[0]-center_point[1]])

    frame = frame[center_point[1]-closest:center_point[1]+closest, center_point[0]-closest:center_point[0]+closest]

    print(fps*start_sec, i)

    if not ret:
        break

    if i>=start_sec and i%10 == 0:
        while(os.path.isfile('D:/Omni Dataset/full_dataset/images/omni_video_' + str(j) + '.jpg')):
            j+=1
        filename = 'D:/Omni Dataset/full_dataset/images/omni_video_' + str(j) + '.jpg'
        cv2.imwrite(filename, frame)
        j+=1

    cv2.imshow('images', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        break

    i+=1