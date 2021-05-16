import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
image= cv2.imread("Resources/top.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
def make_coordinates(image,parameters):
    slope,intercept=parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    y2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(399, 270), (1200, 270), (imshape[1],imshape[0])]])
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 40    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100 #minimum number of pixels making up a line
max_line_gap = 5   # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
# left=[]
# right=[]
# for line in lines:
#    for x1,y1,x2,y2 in line:
#        parameters=np.polyfit((x1,x2),(y1,y2),1)
#        print(parameters)
#        slope=parameters[0]
#        intercept=parameters[1]
#        if slope<0:
#            left.append((slope,intercept))
#        else:
#            right.append((slope,intercept))
# print(left)
# print(right)
# left_avg=np.average(left,axis=0)
# right_avg=np.average(right,axis=0)
# print(left_avg,"left")
# print(right_avg,"right")
# left_line=make_coordinates(image,left_avg)
# right_line=make_coordinates(image,right_avg)
#
#
# avg_line=np.array([left_line,right_line])
# print(avg_line)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for  x1,y1,x2,y2 in line:
          print(x1, y1, x2, y2,)
          cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))



lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()
cv2.imshow("image",lines_edges)
cv2.waitKey(0)