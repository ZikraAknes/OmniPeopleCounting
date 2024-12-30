import cv2
import os

output_dir = 'D:/Omni Dataset/full_dataset/images'
input_dir = "C:/Users/zikra/OneDrive/Pictures/Camera Roll/WIN_20241127_02_17_46_Pro.mp4"

def extract_frames(input_path):
    cap = cv2.VideoCapture(input_path)

    i = 0
    j = 800
    while cap.isOpened():
        ret, frame = cap.read()

        center_point = [1045, 657]

        closest = min([center_point[0], center_point[1], frame.shape[1]-center_point[0], frame.shape[0]-center_point[1]])

        frame = frame[center_point[1]-closest:center_point[1]+closest, center_point[0]-closest:center_point[0]+closest]

        if not ret:
            break

        if i%10 == 0:
            while(os.path.isfile(os.path.join(output_dir, 'omni_video_' + str(j) + '.jpg'))):
                j+=1
            filename = os.path.join(output_dir, 'omni_video_' + str(j) + '.jpg')
            cv2.imwrite(filename, frame)
            j+=1

        cv2.imshow('images', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            break

        i+=1

# for file in os.listdir(input_dir):
    # extract_frames(os.path.join(input_dir, file))

extract_frames(input_dir)

