#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:24:18 2022
Use key point matching to search for logos
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def search_template(sourceimgpath, templatepath, feature_type = "ORB", min_match_count = 10, nfeatures = 2000,  plot_match = True, outfile='./images/result.png'):
    """search template in an image using key points match
    
    Parameters
    ----------
    sourceimgpath : string
        source image path.
    templatepath : string
        template image path.
    feature_type : string, optional
        feature type, can be SIFT or ORB. The default is "ORB".
    min_match_count : string, optional
        minimal number of key point matches for valid detect. The default is 10.
    nfeatures : string, optional
        max number of key point used. The default is 2000.
    outfile : string, optional
        output path. The default is './images/result.png'.

    Returns
    -------
    None.

    """
    
    template = cv2.imread(templatepath, 0)
    image = cv2.imread(sourceimgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    feature_detector = None
    index_params = None
    if feature_type == "ORB":
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, 
                       key_size = 12,   
                       multi_probe_level = 1)
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
    
    # get good match count
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    # valid detect
    if len(good)> min_match_count:
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
        cv2.imwrite(outfile, image)
    else:
        print( "Not enough matches found - {}/{}".format(len(good), min_match_count) )
        matchesMask = None
        
    if plot_match:
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
        matchimg = cv2.drawMatches(template,kp_template,image,kp_image,good,None,**draw_params)
        plt.imshow(cv2.cvtColor(matchimg, cv2.COLOR_BGR2RGB))
        plt.show()
        
if __name__== '__main__':
    templatepath = "./images/nbc-logo.png"
    sourceimgpath = "./images/nbc-rotated.png"
    search_template(sourceimgpath, templatepath, outfile='./images/nbc-result-keypoint-match.png')