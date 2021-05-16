# Autonomous_golf_cart


# Introduction
KCIRI is a reasearch centre and I did an internship there. They bought an autonomous golf cart and tried to make it autonomous.


It consisted of many componenets including lane detection,localisation,behavioral cloning,CAN protocols,GPS tracking,Lidar.


Here, I've included the lane detection and behavioral cloning components.


# Advanced Lane Finding:
  In this module, we mainly used OpenCV. First converting image to grey and then applying sobel operators to get the edges in the image.
  Then region masking was done and applying perspective transformation to convert it into bird's view. Sliding window, technique is applied to
  this image and then the lane lines are got. Now, unwarping the image is done and radius of curvatue of the road is found using the distance of pixels 
  between the two lanes lines right and left. This whole pipeline is include in the extrapolating_lane_lines.py file.
  
  
# Behavioral Cloning:
    Basically, behavioral cloning means to mimic the human behavior. Here, we took the images and the steering angles
    from  the vehicles and  tried to build a deep learning model which tries to predict the steering angle which
    should be applied by the car.This is an end-to-end pipeline model where we just pass the images and it predicts
    the steering angle which should be applied.Tensorflow and keras are mainly used in this module. 
    The model can be found in the cnn_fully.py file.
