# encoding: utf-8
# @author: Evan
# @file: figs_label.py
# @time: 2022/10/25 9:20
# @desc: 给图片打标签

import os
import pandas as pd

if __name__ == '__main__':  # 主函数
    # 获得图片列表
    fulims_path = '../full/'
    empims_path = '../empty/'
    fulims_list = os.listdir(fulims_path)
    fulims_list.sort(key=lambda x: int(x.split('.')[0]))
    empims_list = os.listdir(empims_path)
    empims_list.sort(key=lambda x: int(x.split('.')[0]))
    imgnums = len(fulims_list) + len(empims_list)
    # 创建标签列表
    data = pd.DataFrame(columns=['name', 'label'])
    # 制作标签
    for i in range(imgnums):
        if i < len(fulims_list):
            fulims_name = fulims_list[i]
            data.loc[i] = [fulims_name, 1]
        else:
            empims_name = empims_list[len(fulims_list) - i]
            data.loc[i] = [empims_name, 0]
    # 保存标签文件
    data.to_csv('garbage_label_data.csv')
