import cv2
import os
import numpy as np
from cv2 import aruco, imshow
import math

image = cv2.imread("aruco_images/room50.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
image2 = cv2.imread("aruco_images/poster.jpg")
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2BGRA)

def image_shower(image):
    cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('window', image)
    # cv2.imwrite('saved_images',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

height, width, chhanel = image2.shape
print("Height:",height)
print("Width:",width)
topleft = [0, 0]
topright = [width, 0]
bottomright = [width, height]
bottomleft = [0, height] #left bottom

c = np.array([topleft, topright, bottomright, bottomleft], dtype=np.float32)
[x, y] = center = np.mean(c, axis=0)
print(center)
w = width/6
h = height/4
corner1 = (x - w / 2, y - h / 2)
corner2 = (x + w / 2, y - h / 2)
corner3 = (x + w / 2, y + h / 2)
corner4 = (x - w / 2, y + h / 2)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
parameters = aruco.DetectorParameters()
arucocorners, ids, Error = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

print('ArucoCorners: ',arucocorners)

Arucoc1 = arucocorners[0][0][0]
Arucoc2 = arucocorners[0][0][1]
Arucoc3 = arucocorners[0][0][2]
Arucoc4 = arucocorners[0][0][3]

input_corners = np.float32([corner1,corner2,corner3,corner4])
output_corners = np.float32([Arucoc1,Arucoc2,Arucoc3,Arucoc4])

M = cv2.getPerspectiveTransform(input_corners, output_corners)
print(M)
print(input_corners)
print(M.shape,input_corners.shape)

input_corners = np.float32([topleft,topright,bottomright,bottomleft])
final_corners = cv2.perspectiveTransform(input_corners.reshape(-1, 1, 2), M)
print("Final Corners: ",final_corners)
sac1=final_corners[0,0]
sac2=final_corners[1,0]
sac3=final_corners[2,0]
sac4=final_corners[3,0]


transformed_image2= cv2.warpPerspective(image2, M,(image.shape[1], image.shape[0]))
# image_shower(transformed_image2)
pts = np.array([sac1,sac2,sac3,sac4], np.int32)
print('Final Points',pts)
# Create mask with zeros
mask = np.zeros_like(image)
cv2.fillPoly(mask, [pts], (255, 255, 255,255))
image_shower(mask)

# Invert mask to keep everything outside the polygon area
mask = cv2.bitwise_not(mask)
# image_shower(mask)

# Apply mask to image to remove the polygon area
image = cv2.bitwise_and(image, mask)
image_shower(image)
final_image = cv2.bitwise_or(image,transformed_image2)
image_shower(final_image)
cv2.imwrite("aug_images/new_img1.jpg",final_image)

