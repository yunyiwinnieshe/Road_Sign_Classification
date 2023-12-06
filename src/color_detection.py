import cv2
import numpy as np
from pytesseract import image_to_string
import matplotlib.pyplot as plt 


#pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def detect_traffic_light_color(image, bounding_box):
    x, y, w, h = bounding_box
    
    cropped_image = image[y:h, x:w]
    cv2.imwrite('bb1.png', cropped_image)    
    
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 
        1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        for (cx, cy, radius) in circles:
            # Define a smaller circle within the detected circle
            circle_img = np.zeros_like(blurred)  
            cv2.circle(circle_img, (cx, cy), radius, 255, thickness=-1)

            # Mask the original image to analyze the circle color
            masked_data = cv2.bitwise_and(cropped_image, cropped_image, mask=circle_img)

            # Calculate the average color of the masked area
            avg_color_per_row = np.average(masked_data, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            avg_color = avg_color.astype(int)

            # Determine the color
            red, green, blue = avg_color[2], avg_color[1], avg_color[0]
            tolerance = 40  # Adjust the tolerance level as needed

            if red > green + tolerance and red > blue + tolerance:
                return "Red"
            elif green > red + tolerance and green > blue + tolerance:
                return "Green"
            elif red > blue + tolerance and green > blue + tolerance:
                return "Yellow"
    # Use color detection if no colored circles are detected 
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Range of Red, Yellow and Green 
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([15, 100, 100])  
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([40, 40, 40])  
    green_upper = np.array([80, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    green_area = np.sum(green_mask)

    if red_area > yellow_area and red_area > green_area:
        return 'red'
    elif yellow_area > red_area and yellow_area > green_area:
        return 'yellow'
    elif green_area > red_area and green_area > yellow_area:
        return 'green'
    return "Unknown"

    
def main():   
    traffic_lights = ['road1.png', 'road2.png', 'road3.png', 'road5.png', 'road6.png', 'road10.png', 'road11.png', 'road11.png', 'road12.png', 'road13.png', 'road15.png','road19.png', 'road24.png', 
        'road32.png', 'road33.png', 'road34.png', 'road37.png', 'road39.png', 'road42.png', 'road43.png']
    ac_colors =['yellow', 'red', 'red', 'red', 'red', 'red', 'green', 'green', 'yellow', 'red','red', 'green','red','green','red','green', 'red', 'red', 'red', 'green']
    bbs = [[154,63,258,281],[144,270,174,352],[178,134,236,261],[46,30,221,292], [95,153,176,327], [106,3,244,263],[188,98,207,132],[317,172,331,200],[136,92,155,133], [75,74,195,348], [186,102,227,157], [99,104,119,157], 
        [70,174,128,286],[48,33,194,356], [23,6,178,387], [21,4,177,386], [116, 103, 156, 230], [89,9,218,366], [177,184,221,297], [177,184,221,297], [90,86,105,116]]
    count = 0 
    for x in range(len(traffic_lights)):
        image = cv2.imread(traffic_lights[x])
        bb_hat = bbs[x]
        color = detect_traffic_light_color(image, bb_hat)
        if color == ac_colors[x]: 
            count +=1 
        else: 
            print('p: ' + color + ' a: ' + ac_colors[x])
            print(traffic_lights[x])
    print('accuracy: ' + str(count) +'/' + str(len(traffic_lights)))
    return 


if __name__ == "__main__":
    main()
