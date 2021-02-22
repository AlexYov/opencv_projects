import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-image', dest = 'file_image', help = 'Файл с изображением')
args = parser.parse_args()
input_file= args.file_image

image = cv2.imread(input_file)


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
float_image = np.float32(gray_image)

harris_method_image = cv2.cornerHarris(float_image, 3, 5, 0.04)
dilated_image = cv2.dilate(harris_method_image, None)
image[dilated_image > 0.0 * dilated_image.max()] = [0,0,255]
cv2.imshow('Harris Corner', image)

button = cv2.waitKey(0)
if button == 27:
    cv2.destroyAllWindows()