import numpy as np
import cv2 as cv

# The given video and updated calibration data
video_file = r'C:\Users\p0105\Videos\chessboard.mp4'  # 비디오 파일 경로
K = np.array([[754.36932621, 0, 354.80740075],
              [0, 769.75820428, 440.99934945],
              [0, 0, 1]])  # 새로운 카메라 매트릭스
dist_coeff = np.array([2.77612703e-01, -2.93292907e+00, 1.32399985e-02, -2.81894361e-03, 9.97246553e+00])  # 새로운 왜곡 계수

# Open the video
video = cv.VideoCapture(video_file)
assert video.isOpened(), f'Cannot read the given input video file: {video_file}'

# Run distortion correction
map1, map2 = None, None
while True:
    # Read a frame from the video
    valid, original_img = video.read()
    if not valid:
        break

    # Perform distortion correction
    if map1 is None or map2 is None:
        map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (original_img.shape[1], original_img.shape[0]), cv.CV_32FC1)
    rectified_img = cv.remap(original_img, map1, map2, interpolation=cv.INTER_LINEAR)

    # Add text overlays
    cv.putText(original_img, "Original", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv.putText(rectified_img, "Rectified", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show both images in separate windows
    cv.imshow("Original Video", original_img)
    cv.imshow("Rectified Video", rectified_img)

    # Process key events
    key = cv.waitKey(10)
    if key == ord(' '):  # Space: Pause
        key = cv.waitKey()
    if key == 27:  # ESC: Exit
        break

video.release()
cv.destroyAllWindows()

 

