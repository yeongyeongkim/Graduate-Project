#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import numpy as np

# Set RANSAC parameters (You can change this part)
num_trials = 2000
distance_threshold = 0.01
confidence = 0.99

# Load the images
img1 = cv2.imread('im0.png')
img2 = cv2.imread('im1.png')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect features in both images
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match features using the descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Extract the matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute the fundamental matrix using RANSAC
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold=distance_threshold, confidence=confidence, maxIters=num_trials)

# Extract the inlier matches
inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
# Extract the outlier matches
outlier_matches = [matches[i] for i in range(len(matches)) if not mask[i]]

# Draw the inlier matches
img_inlier_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Draw the outlier matches
img_outlier_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, outlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the image with the inlier matches
cv2.imwrite('./inlier_matches.png', img_inlier_matches)
# Display the image with the outlier matches
cv2.imwrite('./outlier_matches.png', img_outlier_matches)

##
# Write a code that compute the accuracy of the estimated fundamental matrix
# Note that the fundamental matrix F and all pairs of corresponding points holds:
# (x')^T F x = 0. (Lecture2 p.28)

# Compute the average reprojection error
avg_error = 0
for i in range(len(src_pts)):
    x1 = np.array([src_pts[i][0][0], src_pts[i][0][1], 1])
    x2 = np.array([dst_pts[i][0][0], dst_pts[i][0][1], 1])
    error = abs(np.dot(np.dot(x2, F), x1.T))
    avg_error += error
avg_error /= len(src_pts)

# Print the average reprojection error
print('Average reprojection error:', avg_error)


# In[ ]:




