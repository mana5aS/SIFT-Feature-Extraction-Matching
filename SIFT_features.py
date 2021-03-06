# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:09:40 2018

@author: Manasa Sathyan
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# define path
path = ''

"""Noisy Image"""
def create_SNP(image):  
    row,col = image.shape
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(image)

    # Salt Noise
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
    
    # Pepper Noise
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    
    cv.imwrite(path + 'LennaSNP.jpeg', out)
    
    return out


"""Rotated Image"""
def create_ROT(img):
    num_rows, num_cols = img.shape[:2]
    
    rotation_matrix = cv.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    img_rotation = cv.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    
    cv.imwrite(path + 'LennaROT.jpeg', img_rotation)
    
    return img_rotation


"""Feature Extraction"""
def display_SIFT(img):
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    img_kp = cv.drawKeypoints(img, kp, img)
    
    #plt.figure(figsize=(15, 15))
    plt.figure(figsize=(5, 5))
    plt.imshow(img_kp)
    plt.show()


"""Feature Matching"""
def match_features(img1, img2):
    
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.65*n.distance:
            good.append([m])
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,img2, flags = 2)

    return img3


if __name__ == '__main__':

    img1 = cv.imread(path + 'Lenna.png') 
    gray_img = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = create_SNP(gray_img) 
    img3 = create_ROT(gray_img)

    # Feature extraction 
    display_SIFT(gray_img)
    display_SIFT(img2)
    display_SIFT(img3)

    # Feature Matching 
    plt.figure(figsize=(10, 10))
    match_img1 = match_features(gray_img, img2)
    plt.imshow(match_img1)

    plt.figure(figsize=(10, 10))
    match_img2 = match_features(gray_img, img3)
    plt.imshow(match_img2)
    plt.show()
