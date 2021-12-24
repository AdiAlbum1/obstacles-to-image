import numpy as np
import cv2 as cv

if __name__ == "__main__":
    img = cv.imread("img.png")

    M = np.float32([
        [1, 0, 0],
        [0, 1, 40]
    ])

    shifted = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv.imshow("translated", shifted)
    # cv.imshow("img", img)
    cv.waitKey(0)