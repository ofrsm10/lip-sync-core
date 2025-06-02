import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('/Users/ofersimchovitch/PycharmProjects/lipSyncBeta/Utils/teeth.jpg')
resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.namedWindow('marking')

cv2.createTrackbar('H Lower', 'marking', 0, 179, nothing)
cv2.createTrackbar('H Higher', 'marking', 179, 179, nothing)
cv2.createTrackbar('S Lower', 'marking', 0, 255, nothing)
cv2.createTrackbar('S Higher', 'marking', 255, 255, nothing)
cv2.createTrackbar('V Lower', 'marking', 0, 255, nothing)
cv2.createTrackbar('V Higher', 'marking', 255, 255, nothing)

while True:

    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized_img, cv2.COLOR_RGB2HSV)

    hL = cv2.getTrackbarPos('H Lower', 'marking')
    hH = cv2.getTrackbarPos('H Higher', 'marking')
    sL = cv2.getTrackbarPos('S Lower', 'marking')
    sH = cv2.getTrackbarPos('S Higher', 'marking')
    vL = cv2.getTrackbarPos('V Lower', 'marking')
    vH = cv2.getTrackbarPos('V Higher', 'marking')

    LowerRegion = np.array([hL, sL, vL], np.uint8)
    upperRegion = np.array([hH, sH, vH], np.uint8)

    mask = cv2.inRange(hsv, LowerRegion, upperRegion)

    kernal = np.ones((1, 1), "uint8")

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    mask = cv2.dilate(mask, kernal, iterations=1)

    res = cv2.bitwise_and(resized_img, resized_img, mask=mask)

    cv2.imshow("Masking", res)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q' press
        break
    elif key == ord('r'):  # Reset on 'r' press
        cv2.setTrackbarPos('Tongue', 'Image', 0)
        cv2.setTrackbarPos('teeth', 'Image', 0)