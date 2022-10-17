#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:49:18 2022
Use simple template matching to search for logos.

Ref: https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/#pyis-cta-modal

Pros:
    easy, fast, straightforward
    can handle variations in translation, scaling, and perhaps some 
    kind of transparency and minor distotion
Cons:
    due to its nature, cannot handle rotation
    template matching does not do a good job of telling us if an object does not appear in an image. 
    Sure, we could tune a threshold on the correlation coefficient, but in practice this is not reliable and robust
    
"""

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# define input images
templatepath = "./images/nbc-logo.png"
sourceimgpath = "./images/nbc-large.png"

# load template image
template = cv2.imread(templatepath)
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.show()
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#plt.imshow(template)
#plt.show()
#plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
#plt.show()
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
#plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
#plt.show()

# load source image
image = cv2.imread(sourceimgpath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()


found = None
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
    ratio = gray.shape[1] / float(resized.shape[1])
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
    #clone = np.dstack([edged, edged, edged])
    #cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 20)
    #plt.imshow(clone)
    #plt.show()
    
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, ratio)
        
_, maxLoc, ratio = found
startX, startY = (int(maxLoc[0] * ratio), int(maxLoc[1] * ratio))
endX, endY = (int((maxLoc[0] + tW) * ratio), int((maxLoc[1] + tH) * ratio))
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 40)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

