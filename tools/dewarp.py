import cv2
import numpy as np

class DewarpImage():
    def create_panorama(img, cam_area=None):
        if cam_area == None:
            cropped_img = img
        else:
            cropped_img = img[cam_area[0]:cam_area[1], cam_area[2]:cam_area[3]]

        width = int(cropped_img.shape[0] / 2)

        Panorama = np.zeros((width, 4*width, 3), np.uint8)

        for i in range(width):
            for j in range(width*4):
                radius = width - i
                theta = 2*np.pi*(j/(4*width))

                x = width - int(round(radius * np.cos(theta)))
                y = width - int(round(radius * np.sin(theta)))

                if(x >= 0 and x < 2*width and y >=0 and y < 2*width):
                    Panorama[i][j] = cropped_img[x][y]

        return Panorama
    
    def dewarped_points(x, y, img, cam_area):
        cropped_img = img[cam_area[0]:cam_area[1], cam_area[2]:cam_area[3]]

        width = int(cropped_img.shape[0] / 2)
