import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('CAM003.png', 0)
# img = cv2.medianBlur(img, 5)

matrix = np.loadtxt('matrix.txt')
pic = np.expand_dims(matrix, axis=2)

# https://blog.csdn.net/u010555688/article/details/38779447
hi = np.max(pic)
lo = np.min(pic)
mmax = 255
mmin = 0
img = (mmax - mmin) * (pic - lo) / (hi - lo) + mmin


# # https://blog.csdn.net/JNingWei/article/details/78213360
# pic *= (pic > 0)
# pic = pic * (pic <= 255) + 255 * (pic > 255)
# pic = pic.astype(np.uint8)
# img = pic

# https://blog.csdn.net/qq_30909117/article/details/78789311
ret, th1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['orgianl Image',
          'Gllobal Thresholding(v=127)', 'ADaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
cv2.imshow("sss", images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()