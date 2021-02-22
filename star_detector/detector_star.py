import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-image', dest = 'file_image')
args = parser.parse_args()
input_file = args.file_image

class StarFeatureDetector(object):
    
    def __init__(self):
        self.detector = cv2.xfeatures2d.StarDetector_create()
        
    def detect(self, image):
        return self.detector.detect(image)
    
if __name__=='__main__':
    
    image = cv2.imread(input_file)
    keypoints = StarFeatureDetector().detect(image)
    cv2.drawKeypoints(image, keypoints, image, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('Star Feature Detector', image)
    button = cv2.waitKey(0)
    if button == 27:
        cv2.destroyAllWindows()

