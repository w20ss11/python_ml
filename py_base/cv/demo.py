# -*- coding=utf-8 -*-
import cv2 as cv
import numpy as np


def create_image():
    img = np.ones([400, 400, 1], np.uint8)
    img = img * 127
    cv.imshow("img1", img)


create_image()
cv.waitKey(0)
cv.destroyAllWindows()
