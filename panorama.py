
"""
Stitch the images using automatic feature extraction and maching points with homography 
"""

import cv2
import numpy as np

# load the images that will compose the panorama
img1 = cv2.imread('right.png')
img2 = cv2.imread('left.png')

# let's create a feature extractor
orb = cv2.ORB_create()

# compute the features in both images
kpt1, desc1 = orb.detectAndCompute(img1,None)
kpt2, desc2 = orb.detectAndCompute(img2,None)

# create a matcher and match the keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(desc1,desc2,k=2)

# perform the ratio test:
# a match is correct if the
# ratio betwee the two closest points of a match
# is below a certain threshold
good_matches = []
for m,n in matches:
    if m.distance < 0.3 * n.distance:
        good_matches.append(m)

# check if we have at least 4 points
# REMEMBER: the two images are called queryImg and trainingImg
# so queryIdx is a point belonging to query Img,
# while trainingIdx is a point belonging to trainingImg
# In our case, queryImg is the left image, while trainingImg is the right image
if len(good_matches) > 4:
    # convert the points to float32
    src_points = np.float32([kpt1[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([kpt2[m.trainIdx].pt for m in good_matches])

    # compute the homography matrix
    M, mask = cv2.findHomography(src_points,dst_points)

    # transform the left image and stitch it together with the
    # right image
    dst = cv2.warpPerspective(img1,M,(img1.shape[1] + img2.shape[1], img1.shape[0]))
    dst[0:img2.shape[0],0:img2.shape[1]] = img2.copy()

#result shows black pixels on the right side since some pixels are overlaped. it is easy to remove it 
    cv2.namedWindow('Panorama',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Panorama',dst)
    cv2.waitKey(0)
    