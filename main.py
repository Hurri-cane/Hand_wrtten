# author:Hurricane
# date:  2020/11/6
# E-mail:hurri_cane@qq.com

import cv2 as cv
import numpy as np
import os
from Pre_treatment import get_number as g_n
from Pre_treatment import get_roi
import predict as pt
from time import time
from Pre_treatment import softmax

# 实时检测视频
capture = cv.VideoCapture(0,cv.CAP_DSHOW)
capture.set(3, 1920)
capture.set(4, 1080)
net = pt.get_net()

# img_path = r'F:\PyCharm\Practice\hand_wrtten\real_img\7.jpg'
while (True):
    ret, frame = capture.read()
    since = time()
    if ret:
        # frame = cv.imread(img_path)

        img_bw = g_n(frame)
        img_bw_sg = get_roi(img_bw)
        # 展示图片
        cv.imshow("img", img_bw_sg)
        c = cv.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
        img_in = cv.resize(img_bw_sg, (28, 28))
        result_org = pt.predict(img_in, net)
        result = softmax(result_org)
        best_result = result.argmax(dim=1).item()
        best_result_num = max(max(result)).cpu().detach().numpy()
        if best_result_num <= 0.5:
            best_result = None

        # 显示结果
        img_show = cv.resize(frame, (600, 600))
        end_predict = time()
        fps = round(1/(end_predict-since))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_show, "The number is:" + str(best_result), (1, 30), font, 1, (0, 0, 255), 2)
        cv.putText(img_show, "Probability is:" + str(best_result_num), (1, 60), font, 1, (0, 255, 0), 2)
        cv.putText(img_show, "FPS:" + str(fps), (1, 90), font, 1, (255, 0, 0), 2)
        cv.imshow("result", img_show)
        cv.waitKey(1)
        print(result)
        print("*" * 50)
        print("The number is:", best_result)


    else:
        print("please check camera!")
        break