import os

start_num = 71

image_path = os.path.join(os.path.dirname(__file__), '..', 'extracted_frames/images')
label_path = os.path.join(os.path.dirname(__file__), '..', 'extracted_frames/labels')

file_list = os.listdir(image_path)
file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for file in file_list:
    os.rename(os.path.join(image_path, file), image_path + '/omni_video_' + str(start_num) + '.jpg')
    os.rename(os.path.join(label_path, file.replace('.jpg', '.txt')), label_path + '/omni_video_' + str(start_num) + '.txt')
    start_num += 1

