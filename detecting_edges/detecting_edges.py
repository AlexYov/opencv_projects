import cv2 # библиотека opencv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-image', help = 'Файл с изображением', dest='file_image')
args = parser.parse_args()
input_file = args.file_image
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE) # переводим изображение в серые тона
canny_edge_detector_2 = cv2.Canny(image, 79, 177)
cv2.imshow('canny_edge_detector_2', canny_edge_detector_2)
button = cv2.waitKey(0)
if button == 27:
    cv2.destroyAllWindows()