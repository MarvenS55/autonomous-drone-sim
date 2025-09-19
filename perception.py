import airsim
import cv2
import numpy as np

class Perception:
    def __init__(self, client):
        self.client = client

    def get_frame(self, camera_name):
        """Gets a single image frame from a specified camera."""
        request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        responses = self.client.simGetImages([request])
        response = responses[0]
        
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame = img1d.reshape(response.height, response.width, 3).copy()
        return frame

    def find_landing_pad(self, frame):
        """Analyzes a frame to find the RED landing pad."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV ranges for the color red
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        target_info = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 100: 
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx = x + w // 2
                cy = y + h // 2
                target_info = ((cx, cy), area) 

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        return target_info, frame