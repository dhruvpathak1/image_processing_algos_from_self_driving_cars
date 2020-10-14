import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# --> This is the function for loading Deep Learning network into
#     system(memory).
# --> It automatically detected configuration and framework
#     based on file name specified.
# --> After Detecting calls an appropriate function such as readNetFromCaffe,
#     readNetFromTensorflow, readNetFromTorch, or readNetFromDarknet.
# --> It returns an object.

classes = []
with open('coco.names', 'r') as name_file:
    # 'with' keyword opens the file, manipulates the file and closes it
    # If any error comes throws it after closing the file
    # 'open' takes 2 parameters filename and mode.
    # r - read (default) other are a - append, w- write, x - create
    classes = name_file.read().splitlines()


# For webcam write 0 in () below like - (0)
cap = cv2.VideoCapture('./Clips/gta v2.mp4')

# img = cv2.imread('test_image.jpg')
# Reads the image and stores in img
while True:
    try:
        _, img = cap.read()  # Capturing each frame
        height, width, _ = img.shape
        # mapping variables in height and weight and ignore any extra values with '_'

        # =============================================================================

        # 1. Resize the image to 416x416 to fit in Yolo-v3
        # 2. Normalize the image by dividing the image pixels by 255
        # 3. And also the values are intended to be RGB order since our image is
        #    in BGR order so swap it
        # 4. While doing the changes don't crop the image

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True
                                     , crop=False)

        # 	"Binary Large OBject."
        # 	So basically imagine an image with 5 people in a background and it’s got
        # 	a lot of features in it. Now you want to detect those features
        #   (lots of different shapes) how you do it, you compare it with a blob.

        # Creates 4-dimensional blob from image.
        # This the format in which the deep learning understands the input image.
        # Optionally resizes and crops image from center.
        # 1st parameter input image
        # 2nd parameter for normalization
        # 3rd parameter resize image size
        # 4th parameter is for mean subtraction but we hae nothing to do
        # 5th parameter is for swapping R B components to convert BGR to RGB
        # 6th parameter id for direct resize image without cropping

        # For an image of size 416 x 416, YOLO predicts
        # ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10647 bounding boxes.

        # for b in blob:
        #     for n, img_blob in enumerate(b):
        #         cv2.imshow(str(n), img_blob)

        # So finally blob contains 3 channel - red, green, blue

        # =============================================================================

        # Earlier we have stored our files in 'net'. So now we are going to set input
        # by passing the blob. It sets a new input value to the network.
        net.setInput(blob)

        # Returns names of layers with unconnected outputs.
        # Generally in a sequential CNN network there will be only one output layer
        # at the end. In the YOLO v3 architecture we are using there are multiple
        # output layers giving out predictions. Below 2 Lines gives
        # the names of the output layers. An output layer is not connected to any
        # next layer.

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        # First line helps in getting all the output layers
        # Second line - in this we just pass it into the variable 'layerOutputs'

        # =============================================================================

        # Initialise the lists
        boxes = []  # Create boxes list to extract their bounding boxes
        confidences = []  # confidences list for storing the confidence
        class_ids = []
        # class_id list to store the class IDs which represent the predicted classes

        # =============================================================================

        # In order for us to extract the bounding boxes, predicted classes and
        # confidences we have to run 2 for loops that helps us to loop over the
        # latest outputs.

        for output in layerOutputs:
            for detection in output:
                # For each detection in each output of layerOutputs it should
                # contain 4 hunting box offset, one box confidence, and
                # 80 class prediction.
                # Therefore for each detection it is a list of 85 elements
                # First 4 are location of the bounding boxes and 5th is
                # the box confidence, which indicate how accurate is the
                # bounding box. So the last 80 contains the class of predictions
                # the probability of that class
                # 5th -- The confidence score reflects how likely the box contains an
                # object (objectness) and how accurate is the boundary box.

                # We store all 80 class prediction so we start from
                # 6th element that is 5
                scores = detection[5:]
                # To extract highest score location which has the highest possibility
                class_id = np.argmax(scores)
                # Extract the highest score and assigned it to confidence
                confidence = scores[class_id]

                # Now we have to check that highest prediction which is stored
                # in the confidence is above 0.5 so we are sure that it is
                # predicted correctly

                if confidence > 0.5:
                    # First 4 values of detection are X, Y, W and H
                    # X & Y are the center coordinates of the object
                    # And the size is the width and height i.e. W & H

                    # Since we have normalised the image earlier so we
                    # have multiply by 255 here
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # YOLO detects with center of the bounding boxes.
                    # So we have to find out the upper left corner position in
                    # order for us to present them with help of openCV
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    # So now we have all the information about the bounding boxes,
                    # confidence and predicted class ID.

        # =============================================================================

        # In this we will write non maximum suppression i.e. to reduce no. of
        # redundant boxes.

        # print(len(boxes))   # To print number of boxes

        # Below line removes redundant boxes.
        # NMS - Non Maximum Suppression
        # Performs non maximum suppression given boxes and corresponding scores.
        # 1st parameter => a set of bounding boxes to apply NMS.
        # 2nd parameter => a set of corresponding confidences.
        # 3rd parameter => a threshold used to filter boxes by score usually
        # our 0.5 used in above 'if'.
        # 4th parameter => a threshold used in non maximum suppression
        # default value is 0.4. Which is a ratio defined by the area of the
        # current smallest region divided by the area of current bounding box

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Font to display
        font = cv2.FONT_HERSHEY_PLAIN

        # Also return random color for each box.
        # 1st parameter => Low value i.e. 0.
        # 2nd parameter => High value i.e. 255.
        # So range is 0 <= x <= 255.
        # 3rd parameter => The shape of the returned array. '3' is for 3 channels
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        # flatten() => Return a copy of the array collapsed into one dimension.
        # [[1, 2], [4, 3]] => [1, 2, 3, 4]
        for i in indexes.flatten():
            x, y, w, h = boxes[i]  # Coordinates of the box
            label = str(classes[class_ids[i]])  # Label to be printed
            confidence = str(round(confidences[i] * 100, 2))  # Confidence of the object
            color = colors[i]  # Random color to the box

            if label == "car" or label == "truck":
                if w > 300 or h > 180:
                    cv2.putText(img, "Warning !!!", (x + 50, y + 50), font,
                                1, (255, 255, 255), 1)

            # Now to draw a rectangle.
            # 1st parameter => Image to put on rectangle.
            # 2nd parameter => Coordinates of the rectangle.
            # 3rd parameter => Size of the rectangle.
            # 4th parameter => Color of the rectangle.
            # 5th parameter => Thickness of the rectangle.

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Now to put a text.
            # 1st parameter => Image to put on rectangle.
            # 2nd parameter => What text to put so label + confidence.
            # 3rd parameter => Coordinates of the text.
            # 4th parameter => Font of the text.
            # 5th parameter => Size of the font.
            # 6th parameter => Color of the font.
            # 7th parameter => Thickness of the font.
            cv2.putText(img, label + " " + confidence, (x, y + 20), font,
                        1, (255, 255, 255), 1)

        # =============================================================================

        cv2.imshow("Test Image", img)
        key = cv2.waitKey(1)

        # key == 27 because 27 is ascii value of 'ESC'
        if key == 27:
            break

    except Exception as e:
        print(e)
        pass


cap.release()
cv2.destroyAllWindows()
