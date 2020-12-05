# author:Hurricane
# date:  2020/11/6
# E-mail:hurri_cane@qq.com


import cv2 as cv
import numpy as np
import os


def get_number(img):

    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cv.imshow("gray", img_gray)
    img_gray_resize = cv.resize(img_gray, (600, 600))
    cv.imshow("img_gray_resize", img_gray_resize)
    ret, img_bw = cv.threshold(img_gray_resize, 200, 255, cv.THRESH_BINARY)
    cv.imshow("img_bw", img_bw)
    kernel = np.ones((3, 3), np.uint8)
    # img_open = cv.morphologyEx(img_bw,cv.MORPH_CLOSE,kernel)
    img_open = cv.dilate(img_bw, kernel, iterations=2)
    cv.imshow("img_open", img_open)
    num_labels, labels, stats, centroids = \
        cv.connectedComponentsWithStats(img_open, connectivity=8, ltype=None)
    for sta in stats:
        if sta[4] < 1000:
            cv.rectangle(img_open, tuple(sta[0:2]), tuple(sta[0:2] + sta[2:4]), (0, 0, 255), thickness=-1)
    return img_open

def get_roi(img_bw):
    img_bw_c = img_bw.sum(axis=1) / 255
    img_bw_r = img_bw.sum(axis=0) / 255
    all_sum = img_bw_c.sum(axis=0)
    if all_sum != 0:
        r_ind, c_ind = [], []
        for k, r in enumerate(img_bw_r):
            if r >= 5:
                r_ind.append(k)
        for k, c in enumerate(img_bw_c):
            if c >= 5:
                c_ind.append(k)
        img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]
        leng_c = len(c_ind)
        leng_r = len(r_ind)
        side_len = max(leng_c, leng_r) + 20
        if leng_c == side_len:
            add_r = int((side_len - leng_r) / 2)
            add_c = 10
        else:
            add_r = 10
            add_c = int((side_len - leng_c) / 2)
        img_bw_sg_bord = cv.copyMakeBorder(img_bw_sg, add_c, add_c, add_r, add_r, cv.BORDER_CONSTANT, value=[0, 0, 0])
        return img_bw_sg_bord
    else:
        return img_bw

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition