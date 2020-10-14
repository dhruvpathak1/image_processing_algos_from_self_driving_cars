# Image Processing Algorithms For Self Driving Cars In GTA V

## We’re Building an Open Source Self-Driving Car
The aim of the project was to implement image processing algorithms used in self driving cars and to demonstrate how an automated car sees its environment and how it detects the surrounding obstacles and objects.

In this project we scan the whole frame once and detect lanes and other surrounding objects like car, person, traffic lights, etc. It also shows a heads-up “Warning” if the car is too close to any other object. 

We are performing You Only Look Once Algorithm and Hough Transform Algorithm for Object and Lane Detection Respectively. We have performed these algorithms in a game called GTA V, which is the closest real-life experience.

## Installation Check List
1. Download the following libraries in your Python Interpreter:

    - [ ] OpenCV
    
    - [ ] Numpy
    
    - [ ] Pyplot from Matplot Lib
    
    - [ ] Time
    
    - [ ] ImageGrab from PIL
    
2. Now download the attached 'class.names'.

3. Go to this link below and download '.cfg' and '.weights' file for YOLOv3-320
   (https://pjreddie.com/darknet/yolo/)
   
Make sure everthing is in the same folder.
   
## Arrangement For Running The Code
1. If you have a GTA V Game then :

    a.) Lower your game resolution in the settings to 800x600
    
    b.) Lower all your graphics to setting High like Shadow, Reflection, etc.
    
    c.) Now drag the window to the top left corner of your screen.
    
2. If you do not have the game then download our video and :

    a.) Resize the window to almost 20-25% of your screen size.
    
    b.) And now drag it to the top left corner of your screen.
    
## Now You Are Ready To Run !!!
Run the code and adjust the output screen in such a way that it does not overlap the game or the video.

## For More Information
For some detailed explaination on each of the algorithm 

    a.) Lane Detection - Hough Transform
    
    b.) Object Detection - You Look Only Once
   
Please visit our Blog Website 

https://btechblogs.herokuapp.com/blog/lanedetection

https://btechblogs.herokuapp.com/blog/objectdetection

