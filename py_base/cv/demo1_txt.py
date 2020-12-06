# -*- coding=utf-8 -*-
import cv2 as cv
import numpy as np


matrix = np.loadtxt('matrix9.txt')
matrix = np.expand_dims(matrix, axis=2)
img = matrix  # * 255
print(img[0, 0])
print(img.shape)
print(np.max(img))
print(np.min(img))

print('------------------------------------')
hi = np.max(img)
lo = np.min(img)
mmax = 255
mmin = 0
img = (mmax - mmin) * (img - lo) / (hi - lo) + mmin
print(np.max(img))
print(np.min(img))

cv.imshow("sss", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

print('------------------------------------')
blurred = cv.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
cv.imshow("Image", img)  # 显示图像
(T, thresh) = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)  # 阈值化处理，阈值为：155
cv.imshow("Threshold Binary", thresh)

(T, threshInv) = cv.threshold(blurred, 200,
                              255, cv.THRESH_BINARY_INV)  # 反阈值化处理，阈值为：155
cv.imshow("Threshold Binary Inverse", threshInv)

#cv2.imshow("Coins", cv2.bitwise_and(image, image, mask =threshInv))
cv.waitKey(0)
