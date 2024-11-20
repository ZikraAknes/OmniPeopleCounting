import numpy as np
import cv2

class CropImage:
    def get_cam_area(image):
        # Convert to grayscale. 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        # Blur 
        gray_blurred = cv2.medianBlur(gray, 25)
        
        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(gray_blurred,  
        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
        param2 = 35, minRadius = 10, maxRadius = 50) 

        if detected_circles is not None: 
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
            
            pt = detected_circles[0, 0]
            center_point = [pt[0], pt[1]]
            closest = min([center_point[0], center_point[1], image.shape[1]-center_point[0], image.shape[0]-center_point[1]])
        
        return [center_point[1]-closest, center_point[1]+closest, center_point[0]-closest, center_point[0]+closest]
    
    def crop_center(img, center_point):
        closest = min([center_point[0], center_point[1], img.shape[1]-center_point[0], img.shape[0]-center_point[1]])

        img = img[center_point[1]-closest:center_point[1]+closest, center_point[0]-closest:center_point[0]+closest]

        return img
