# author:Hurricane
# date:  2020/11/5
# E-mail:hurri_cane@qq.com

import cv2 as cv
import numpy as np
import os
orig_path = r"F:\PyCharm\Practice\hand_wrtten\real_img"
img_list = os.listdir(orig_path)
kernel = np.ones((3,3),np.uint8)
for img_name in img_list:
    img_path = os.path.join(orig_path, img_name)
    img = cv.imread(img_path)
    img_resize = cv.resize(img,(600,600))
    img_gray = cv.cvtColor(img_resize, cv.COLOR_RGB2GRAY)
    ret, img_bw = cv.threshold(img_gray, 200, 255,cv.THRESH_BINARY)

    # img_open = cv.morphologyEx(img_bw,cv.MORPH_CLOSE,kernel)
    img_open = cv.dilate(img_bw,kernel,iterations=3)
    # cv.imshow("open", img_open)
    num_labels, labels, stats, centroids = \
        cv.connectedComponentsWithStats(img_open, connectivity=8, ltype=None)
    for sta in stats:
        if sta[4] < 1000:
            cv.rectangle(img_open, tuple(sta[0:2]), tuple(sta[0:2] + sta[2:4]), (0, 0, 255), thickness=-1)
    # cv.imshow("img", img)
    # cv.imshow("gray", img_gray)
    # cv.imshow("bw", img_bw)
    # cv.imshow("dele", img_open)
    cv.imshow("img_resize",img_resize)
    cv.waitKey(0)
    print("*"*50)

cv.destroyAllWindows()



