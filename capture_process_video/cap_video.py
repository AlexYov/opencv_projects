import cv2

capture_video = cv2.VideoCapture(0)
size_window = 0.8
while True:
    _, frames = capture_video.read()
    new_frames = cv2.resize(frames, None, fx = size_window, fy = size_window, interpolation = cv2.INTER_AREA)
    cv2.imshow('Window', new_frames)
    stop_button = cv2.waitKey(1)
    if stop_button == 27:
        capture_video.release()
        break
cv2.destroyAllWindows()