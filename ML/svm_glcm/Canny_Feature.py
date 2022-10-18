# encoding: utf-8
# @author: Evan
# @file: Canny_Feature.py
# @time: 2022/10/18 19:02
# @desc: Canny算法——图像边缘轮廓提取

import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 生成高斯核
def gaussian_create():
    sigma1 = sigma2 = 1
    gaussian_sum = 0
    g = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            g[i, j] = math.exp(-1 / 2 * (np.square(i - 1) / np.square(sigma1)
                                         + (np.square(j - 1) / np.square(sigma2)))) / (
                              2 * math.pi * sigma1 * sigma2)
            gaussian_sum = gaussian_sum + g[i, j]
    g = g / gaussian_sum  # 归一化
    return g


# 产生灰度图
def gray_fuc(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# 高斯卷积
def gaussian_blur(gray_img, g):
    """
    gray_img:灰度图
    g:高斯核
    """
    gray_img = np.pad(gray_img, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = gray_img.shape
    new_gray_img = np.zeros([h - 2, w - 2])
    for i in range(h - 2):
        for j in range(w - 2):
            new_gray_img[i, j] = np.sum(gray_img[i:i + 3, j:j + 3] * g)
    return new_gray_img


# 求高斯偏导
def partial_derivative(new_gray_img):
    """
    new_gray_img:高斯卷积后的灰度图
    """
    new_gray_img = np.pad(new_gray_img, ((0, 1), (0, 1)), constant_values=0)  # 填充
    h, w = new_gray_img.shape
    dx_gray = np.zeros([h - 1, w - 1])  # 用来存储x方向偏导
    dy_gray = np.zeros([h - 1, w - 1])  # 用来存储y方向偏导
    df_gray = np.zeros([h - 1, w - 1])  # 用来存储梯度强度
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray[i, j] = new_gray_img[i, j + 1] - new_gray_img[i, j]
            dy_gray[i, j] = new_gray_img[i + 1, j] - new_gray_img[i, j]
            df_gray[i, j] = np.sqrt(np.square(dx_gray[i, j]) + np.square(dy_gray[i, j]))
    return dx_gray, dy_gray, df_gray


# 非极大值抑制
def non_maximum_suppression(dx_gray, dy_gray, df_gray):
    """
    dx_gray:x方向梯度矩阵
    dy_gray:y方向梯度矩阵
    df_gray:梯度强度矩阵
    """
    df_gray = np.pad(df_gray, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] != 0:
                gx = math.fabs(dx_gray[i - 1, j - 1])
                gy = math.fabs(dy_gray[i - 1, j - 1])
                if gx > gy:
                    weight = gy / gx
                    grad1 = df_gray[i + 1, j]
                    grad2 = df_gray[i - 1, j]
                    if gx * gy > 0:
                        grad3 = df_gray[i + 1, j + 1]
                        grad4 = df_gray[i - 1, j - 1]
                    else:
                        grad3 = df_gray[i + 1, j - 1]
                        grad4 = df_gray[i - 1, j + 1]
                else:
                    weight = gx / gy
                    grad1 = df_gray[i, j + 1]
                    grad2 = df_gray[i, j - 1]
                    if gx * gy > 0:
                        grad3 = df_gray[i + 1, j + 1]
                        grad4 = df_gray[i - 1, j - 1]
                    else:
                        grad3 = df_gray[i + 1, j - 1]
                        grad4 = df_gray[i - 1, j + 1]
                t1 = weight * grad1 + (1 - weight) * grad3
                t2 = weight * grad2 + (1 - weight) * grad4
                if df_gray[i, j] > t1 and df_gray[i, j] > t2:
                    df_gray[i, j] = df_gray[i, j]
                else:
                    df_gray[i, j] = 0
    return df_gray


# 双阈值过滤
def double_threshold(dx_gray, dy_gray, df_gray, low, high):
    """
    dx_gray:x方向梯度矩阵
    dy_gray:y方向梯度矩阵
    df_gray:梯度强度矩阵
    low:低阈值
    high:高阈值
    """
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] < low:
                df_gray[i, j] = 0
            elif df_gray[i, j] >= high:
                df_gray[i, j] = 1
                if dy_gray[i-1, j-1] * dx_gray[i-1, j-1] > 0:  # dx，dy同向
                    if df_gray[i - 1, j + 1] > low:
                        df_gray[i - 1, j + 1] = high
                    if df_gray[i + 1, j - 1] > low:
                        df_gray[i + 1, j - 1] = high
                    if dy_gray[i-1, j-1] > dx_gray[i-1, j-1]:
                        if df_gray[i, j + 1] > low:
                            df_gray[i, j + 1] = high
                        if df_gray[i, j - 1] > low:
                            df_gray[i, j - 1] = high
                    else:
                        if df_gray[i - 1, j] > low:
                            df_gray[i - 1, j] = high
                        if df_gray[i + 1, j] > low:
                            df_gray[i + 1, j] = high
                else:
                    if df_gray[i - 1, j - 1] > low:
                        df_gray[i - 1, j - 1] = high
                    if df_gray[i + 1, j + 1] > low:
                        df_gray[i + 1, j + 1] = high
                    if math.fabs(dy_gray[i-1, j-1]) > math.fabs(dx_gray[i-1, j-1]):
                        if df_gray[i, j + 1] > low:
                            df_gray[i, j + 1] = high
                        if df_gray[i, j - 1] > low:
                            df_gray[i, j - 1] = high
                    else:
                        if df_gray[i - 1, j] > low:
                            df_gray[i - 1, j] = high
                        if df_gray[i + 1, j] > low:
                            df_gray[i + 1, j] = high
            else:
                df_gray[i, j] = 0
    return df_gray


if __name__ == '__main__':

    fulims_path = '../garbage/full/'  # 图像数据集的路径
    empims_path = '../garbage/empty/'
    fims_list = os.listdir(fulims_path)
    fims_list.sort(key=lambda x: int(x.split('.')[0]))
    eims_list = os.listdir(empims_path)
    eims_list.sort(key=lambda x: int(x.split('.')[0]))
    imgnum = len(fims_list) + len(eims_list)  # 获取总个数

    for i in range(imgnum):
        # 读取图像
        if i < len(fims_list):
            img = cv2.imread('../garbage/full/' + fims_list[i])
        else:
            img = cv2.imread('../garbage/empty/' + eims_list[i - len(fims_list)])

        # 生成高斯核
        gaussian = gaussian_create()
        # 生成灰度图
        gray = gray_fuc(img)
        # 高斯卷积
        new_gray = gaussian_blur(gray, gaussian)
        # 求偏导
        d = partial_derivative(new_gray)
        dx = d[0]
        dy = d[1]
        df = d[2]
        # 非极大值抑制
        new_df = non_maximum_suppression(dx, dy, df)
        # 双阈值过滤,并将图像转换成转化二值图
        low_threshold = 0.15 * np.max(new_df)
        high_threshold = 0.2 * np.max(new_df)
        result = double_threshold(dx, dy, new_df, low_threshold, high_threshold)

        # 输出图像
        # plt.imshow(img)
        # plt.imshow(result, cmap="gray")
        # plt.axis("off")
        # plt.show()

        # 保存图像
        result *= 255
        result = result.astype(np.uint8)
        saveImg_path = '../garbage/canny/' + fims_list[i]
        cv2.imwrite(saveImg_path, result)
        # if i < len(fims_list):
        #     saveImg_path = '../garbage/canny/' + fims_list[i]
        #     cv2.imwrite(saveImg_path, result)
        # else:
        #     saveImg_path = '../garbage/canny/' + eims_list[i - len(fims_list)]
        #     cv2.imwrite(saveImg_path, result)

        # saveImg_path = "../garbage/canny/" + fims_list[i]
        # cv2.imwrite(saveImg_path, result)
        # plt.savefig(saveImg)


