# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:09:40 2018

@author: Manasa Sathyan
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_SNP(image):  
    row,col = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)

    # Salt Noise
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
    
    # Pepper Noise
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    
    cv2.imwrite("LennaSNP.jpeg", out)
    
    return out


def create_ROT(img):
    num_rows, num_cols = img.shape[:2]
    
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    
    cv2.imwrite("LennaROT.jpeg", img_rotation)
    
    return img_rotation


def match_features(img1, img2):
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img2, flags = 2)

    return img3


def display_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(img, kp, img)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(img_kp)
    plt.show()


if __name__=="__main__":
    img1 = cv2.imread('Lenna.jpeg',0)         
    img2 = create_SNP(img1) 
    img3 = create_ROT(img1)

    # Feature extraction 
    display_SIFT(img1)

    # Feature Matching 
    match_img = match_features(img1, img2)
    plt.imshow(match_img)
    plt.show()