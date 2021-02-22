import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-image', help = 'Файл с изображением', dest = 'file_image')
args = parser.parse_args()
input_file = args.file_image
image = cv2.imread(input_file)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_histeq_image = cv2.equalizeHist(gray_image)
new_gray_histeq_image = cv2.resize(gray_histeq_image,None,fx=0.2,fy=0.2)
cv2.imshow('Histogram equalized - grayscale', new_gray_histeq_image)

yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
yuv_image[:,:,2] = cv2.equalizeHist(yuv_image[:,:,2])
histeq_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
new_histeq_image = cv2.resize(histeq_image,None,fx=0.2,fy=0.2)
cv2.imshow('Histogram equalized - color', new_histeq_image)

button = cv2.waitKey(0)
if button == 27:
    cv2.destroyAllWindows()