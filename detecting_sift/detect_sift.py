import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-image', dest = 'file_image1')
parser.add_argument('-image', dest = 'file_image2')
args = parser.parse_args()

input_file = args.file_image

image = cv2.imread(input_file)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray_image, None)

sift_image = np.copy(image)
cv2.drawKeypoints(image, keypoints, sift_image, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT features',sift_image)


button = cv2.waitKey(0)
if button == 27:
    cv2.destroyAllWindows()