# author:Hurricane
# date:  2020/11/6
# E-mail:hurri_cane@qq.com

import cv2 as cv
import numpy as np
import os
from Pre_treatment import get_number as g_n
import predict as pt
from time import time
from Pre_treatment import softmax
net = pt.get_net()
orig_path = r"F:\PyCharm\Practice\hand_wrtten\real_img_resize"
img_list = os.listdir(orig_path)

# img_path = r'F:\PyCharm\Practice\hand_wrtten\real_img\7.jpg'

for img_name in img_list:
    since = time()
    img_path = os.path.join(orig_path, img_name)
    img = cv.imread(img_path)
    img_bw = g_n(img)
    img_bw_c = img_bw.sum(axis=1) / 255
    img_bw_r = img_bw.sum(axis=0) / 255
    r_ind, c_ind = [], []
    for k, r in enumerate(img_bw_r):
        if r >= 5:
            r_ind.append(k)
    for k, c in enumerate(img_bw_c):
        if c >= 5:
            c_ind.append(k)
    img_bw_sg = img_bw[ c_ind[0]:c_ind[-1],r_ind[0]:r_ind[-1]]
    leng_c = len(c_ind)
    leng_r = len(r_ind)
    side_len = leng_c + 20
    add_r = int((side_len-leng_r)/2)
    img_bw_sg_bord = cv.copyMakeBorder(img_bw_sg,10,10,add_r,add_r,cv.BORDER_CONSTANT,value=[0,0,0])
    # 展示图片
    cv.imshow("img", img_bw)
    cv.imshow("img_sg", img_bw_sg_bord)
    c = cv.waitKey(1) & 0xff

    img_in = cv.resize(img_bw_sg_bord, (28, 28))
    result_org = pt.predict(img_in,  net)
    result = softmax(result_org)
    best_result = result.argmax(dim=1).item()
    best_result_num = max(max(result)).cpu().detach().numpy()
    if best_result_num <= 0.5:
        best_result = None

    # 显示结果
    img_show = cv.resize(img, (600, 600))
    end_predict = time()
    fps = np.ceil(1 / (end_predict - since))
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img_show, "The number is:" + str(best_result), (1, 30), font, 1, (0, 0, 255), 2)
    cv.putText(img_show, "Probability is:" + str(best_result_num), (1, 60), font, 1, (0, 255, 0), 2)
    cv.putText(img_show, "FPS:" + str(fps), (1, 90), font, 1, (255, 0, 0), 2)
    cv.imshow("result", img_show)
    cv.waitKey(1)
    print(result)
    print("*" * 50)
    print("The number is:", best_result)
