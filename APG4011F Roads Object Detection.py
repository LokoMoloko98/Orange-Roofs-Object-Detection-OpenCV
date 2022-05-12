#!/usr/bin/env python
# coding: utf-8

#Object detection of similar color
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

# download the image using scikit-image
url = "https://molokomokubedi.s3.af-south-1.amazonaws.com/School/Map4.tif"

print("downloading image from: %s" % (url))

# Reading the image
imgRGB = io.imread(url)
#imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Showing the output
plt.imshow(imgRGB)
plt.title('Aerial Image')
plt.show()


# In[20]:


# convert to hsv colorspace
hsv = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2HSV)

#lower bound and upper bound for colors

#Provided Image bounds 
lower_bound = np.array([0, 50, 0])   
upper_bound = np.array([15, 255, 255])

#Resort Image bounds
#lower_bound = np.array([0, 0, 0]) #(5,50,50) - (15,255,255)
#upper_bound = np.array([20,255,255])

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

plt.imshow(mask)
plt.title('Aerial Image HSV mask')
plt.show()


# In[21]:


#define kernel size  
#kernel = np.ones((1,1),np.uint8)
#print(kernel)

# Remove unnecessary noise from mask
#se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se1)
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

plt.imshow(mask)
plt.title('Aerial Image Roise reduced mask')
plt.show()


# In[22]:


# Segment only the detected region
segmented_img = cv2.bitwise_and(imgRGB, imgRGB, mask=mask)

plt.imshow(segmented_img)
plt.title('Aerial Image Segmented')
plt.show()

plt.imshow(imgRGB)
plt.title('Aerial Image')
plt.show()


# In[24]:


# Find contours from the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

# Showing the output
plt.imshow(output)
plt.title('Output')
plt.show()


# In[ ]:




