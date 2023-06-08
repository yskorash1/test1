import cv2
import numpy as np


y_S = np.zeros((4, 1), np.int)
counter = 0
test_y = 2000


img1 = cv2.imread('C:/Users/HOME/Desktop/mmro/1/DSC02694.jpg', 0)
img2 = cv2.imread('C:/Users/HOME/Desktop/mmro/1/DSC02692.jpg', 0)
img1 = cv2.resize(img1, (1920, 1080))
img2 = cv2.resize(img2, (1920, 1080))
sift = cv2.SIFT_create()


kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)


matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []


for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)


pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]

h1, w1 = img1.shape
h2, w2 = img2.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
)


img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

cv2.imshow('image2', img1_rectified)
cv2.imshow('image2', img2_rectified)
cv2.waitKey(1)
