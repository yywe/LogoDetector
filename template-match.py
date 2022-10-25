#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:49:18 2022
Use simple template matching to search for logos.
"""

import cv2
import imutils
import numpy as np

def search_template(sourceimgpath, templatepath, outfile='./images/result.png'):
    template = cv2.imread(templatepath)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    height, width = template.shape[:2]
    
    image = cv2.imread(sourceimgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        ratio = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < height or resized.shape[1] < width:
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, ratio)
        
    _, maxLoc, ratio = found
    startX, startY = (int(maxLoc[0] * ratio), int(maxLoc[1] * ratio))
    endX, endY = (int((maxLoc[0] + width) * ratio), int((maxLoc[1] + height) * ratio))
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 40)
    cv2.imwrite(outfile, image)
    
if __name__== '__main__':
    templatepath = "./images/nbc-logo.png"
    sourceimgpath = "./images/nbc-large.png"
    search_template(sourceimgpath, templatepath, outfile='./images/nbc-result-temp-match.png')

    