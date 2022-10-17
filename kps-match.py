#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:24:18 2022

Use key point matching to search for logos
Ref: https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

Pros:
    major advantage: 
        local feature invariant, robust to rotation, scaling, etc.
Cons:
    sometimes, it does not work well due to mismatch. (try test with cnn-news-small.png)
    also this alg has some kind of randomness due to Random sample consensus (RANSAC)
    also need to tune the param like how many good matches suffice to infer an object exist
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# define input images
templatepath = "./images/nbc-logo.png"
sourceimgpath = "./images/nbc-small.png"


# feature type, can be SIFT or ORB
feature_type = "ORB"

# minimal match count to be valid
MIN_MATCH_COUNT = 10

# max number of features note this param is imporant
nfeatures = 2000

# load template image (as gray mode)
template = cv2.imread(templatepath, 0)
plt.imshow(template)
plt.show()

# load source image, keep a copy of original BGR format
image = cv2.imread(sourceimgpath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

feature_detector = None
index_params = None
if feature_type == "ORB":
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    feature_detector = cv2.ORB_create(nfeatures)
elif feature_type == "SIFT":
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    feature_detector = cv2.SIFT_create(nfeatures)
else:
    print("unsupported feature type")
    
search_params = dict(checks = 50)

kp_template, des_template = feature_detector.detectAndCompute(template, None)
kp_image, des_image = feature_detector.detectAndCompute(gray, None)

matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(des_template,des_image,k=2)

# get good match
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp_template[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_image[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = template.shape
    # get the 4 corner points of the template
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # get the new corners applying the transformation matrix
    dst = cv2.perspectiveTransform(pts,M)
    # draw the box
    image = cv2.polylines(image,[np.int32(dst)],True,(0, 0, 255), thickness=30, lineType=cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

"""
# draw the match pairs
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
matchimg = cv2.drawMatches(template,kp_template,image,kp_image,good,None,**draw_params)
plt.imshow(cv2.cvtColor(matchimg, cv2.COLOR_BGR2RGB))
plt.show()
"""
