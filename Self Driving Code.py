# -------------------------------------------------------------------------
# DAA Project
# Hough Transform (Lane Detection) Algorithm
# You Only Look Once (Object Detection) Algorithm
# Programmers :-
#       E006 - Vrushit Patel
#       E008 - Dhruv Pathak
# -------------------------------------------------------------------------

import cv2
import numpy as np
import time
from PIL import ImageGrab
import matplotlib.pyplot as plt


# Functions ------------------------------------------------------------------------------------------------------------


def canny(image):
    # Applying Gaussian Blur for noise reduction and smoothening
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Displaying Blurred Image
    # cv2.imshow("Blurred Image", blur_image)
    # cv2.waitKey(0)

    # Detecting edges in the image
    canny_image = cv2.Canny(blur_image, 50, 150)
    # Displaying Canny Image
    # cv2.imshow("Canny Image", canny_image)
    # cv2.waitKey(0)
    return canny_image


def region_of_interest(image):
    # Height of the image using numpy
    height = image.shape[0]
    # Creating the region of interest with reference to plotted image

    # -----------------------------------------------------------------------------------------
    polygons = np.array([[(0, 400), (325,215), (550,230), (790,390)]]) # ( Put the length of the lines to 1/2)

    # polygons = np.array([[(200, height), (1100, height), (550, 250)]])  # ( Put the length of the lines to 4/6)
    # -----------------------------------------------------------------------------------------

    # Creating a black image
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    # Displaying Mask Region
    # cv2.imshow("Masked Region", mask)
    # cv2.waitKey(0)
    masked_img = cv2.bitwise_and(image, mask)
    # Displaying Masked Image
    # cv2.imshow("Masked Image", masked_img)
    # cv2.waitKey(0)
    return masked_img


# Called inside average_slopes
def coordinates(image, line_para):
    slope, intercept = line_para
    y1 = image.shape[0]
    # Length of the line
    y2 = int(y1 * (1 / 2))
    # Obtaining values of x1 and x2 from y= mx + c
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # To determine the slope and intercept for a linear function of 1 degree
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # Separating the 2 lines in 2 different arrays w.r.t slope
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Averaging out the values
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    # Calling coordinates function
    left_line = coordinates(image, left_fit_avg)
    right_line = coordinates(image, right_fit_avg)
    return np.array([left_line, right_line])


def display_lines(image, lines):
    lines_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Reshaping 2D array lines to 1D
            x1, y1, x2, y2 = line.reshape(4)
            # Printing each coordinate on a black image with blue color and thickness 10
            cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
        # Displaying Lane Image
        # cv2.imshow("Lane Image", lines_img)
        # cv2.waitKey(0)
    return lines_img

# ----------------------------------------------------------------------------------------------------------------------
# Object Detection

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open('coco.names', 'r') as name_file:
    classes = name_file.read().splitlines()

last_time = time.time()

while True:
    try:
        game_video = np.array(ImageGrab.grab(bbox=(2, 40, 800, 620)))
        cap = cv2.cvtColor(game_video, cv2.COLOR_BGR2RGB)
        img = cap
        frame = cap

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True
                                 , crop=False)

        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i] * 100, 2))
            color = (0, 255, 255)

            if label == "car" or label == "truck":
                color = (255, 255, 255)
                if w > 300 or h > 180:
                    cv2.putText(img, "Warning !!!", (x + 50, y + 50), font,
                                2, (0, 0, 255), 2)

            if label != "motorbike" and label != "skateboard":
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                cv2.putText(img, label + " " + confidence, (x, y + 20), font,
                            1, (255, 255, 255), 1)

        # cv2.imshow("Test Image", img)
        key = cv2.waitKey(1)

        canny_img = canny(frame)
        cropped_img = region_of_interest(canny_img)
        lines_1 = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        avg_line = average_slope(frame, lines_1)
        line_img = display_lines(frame, avg_line)
        combo_image = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
        cv2.imshow("Lane Detection", combo_image)
        cv2.waitKey(1)

        print('Loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        if key == 27:
            break

    except Exception as e:
        print(e)
        pass

cv2.destroyAllWindows()
