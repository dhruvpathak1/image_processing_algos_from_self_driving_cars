# DAA Project
# Hough Transform (Lane Detection) Algorithm
# Dhruv Pathak

import cv2
import numpy as np
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
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
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
    y2 = int(y1 * (4 / 6))
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


# Main Code ------------------------------------------------------------------------------------------------------------

# Reading Images
# (Original)
org_img = cv2.imread("Lane1.png")
# (Grayscale) 0 in 2nd parameter makes it black and white
lane_img = cv2.imread("Lane1.png", 0)
# Another way to convert to grayscale
# cv2.cvtColor(lane_img,cv2.COLOR_RGB2GRAY)

# Resizing Images to 1290x705
cv2.resize(org_img, (1290, 705))
cv2.resize(lane_img, (1290, 705))

# Displaying Original Image
# cv2.imshow("Original Image", org_img)
# cv2.waitKey(0)

# Displaying Grayscale Image
# cv2.imshow("Grayscale Image", lane_img)
# cv2.waitKey(0)

# Lane Detection in images
# Calling canny function
canny_img = canny(lane_img)
# Calling region of interest function
cropped_img = region_of_interest(canny_img)

# plt.imshow(canny_img)
# plt.show()

# Applying Hough Transform
lines1 = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# Calling average_slope function
avg_line = average_slope(org_img, lines1)
# Calling display_lines function
line_img = display_lines(org_img, avg_line)

combo_image = cv2.addWeighted(org_img, 0.8, line_img, 1, 1)
# Displaying ane Line Image
# cv2.imshow("Lane Detected Image", line_img)
# cv2.waitKey(0)

# Displaying Final output image
cv2.imshow("Final Image", combo_image)
cv2.waitKey(0)

# # For lane detection in Videos
# vid = cv2.VideoCapture("test.mp4")
# while (vid.isOpened()):
#     try:
#         bool_value, frame = vid.read()
#         canny_img = canny(frame)
#         cropped_img = region_of_interest(canny_img)
#         lines_1 = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#         avg_line = average_slope(frame, lines_1)
#         line_img = display_lines(frame, avg_line)
#         combo_image = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
#         # cv2.imshow("Lane Detected Image", line_img)
#         cv2.imshow("Lane Detection", combo_image)
#         cv2.waitKey(100)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     except Exception as e:
#         print(e)
#         pass
# vid.release()
# cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------
