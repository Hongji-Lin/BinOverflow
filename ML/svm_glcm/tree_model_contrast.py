# encoding: utf-8
# @author: Evan
# @file: tree_model_contrast.py
# @time: 2022/10/20 21:08
# @desc:

# 模型二：随机森林，决策树，ExtraTree分类器
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# 读取数据
data = pd.read_csv('../data/garbage_33dim_sorted_data.csv')  # 49×16
data = np.array(data)
rate = []
epoch = 1
start_time = time.time()

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))
    # 划分训练集测试集
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 随机森林
    print('RandomForest train begin')
    clf1 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf1.fit(X_train[:, 1:-1], y_train)
    # joblib.dump(clf1, "RandomForest_garbage.model")        # 模型保存
    # clf = joblib.load("RandomForest_garbage.model")        # 模型加载
    y_pre = clf1.predict(X_test[:, 1:-1])
    print('Training finished')

    sum1 = 0
    err_idx = []
    for i in range(len(y_test)):
        if y_pre[i] == y_test[i]:
            sum1 += 1
        else:
            err_idx.append(i)
    print('分类错误的下标索引值：{}'.format(err_idx))

    err_imgidx = []
    ful_res = []
    emp_res = []
    err_dict = dict()
    for i in range(len(err_idx)):
        err_imgidx.append(X_test[err_idx[i], 0])
    err_imgidx.sort()

    for i in range(len(err_imgidx)):
        if err_imgidx[i] <= 309:
            ful_res.append(err_imgidx[i])
        else:
            emp_res.append(err_imgidx[i])
        err_dict['full'] = ful_res
        err_dict['empty'] = emp_res
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')

# 决策树
    print('DecisionTree train begin')
    clf2 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    clf2.fit(X_train[:, 1:-1], y_train)
    # joblib.dump(clf2, "DecisionTree_garbage.model")        # 模型保存
    # clf = joblib.load("DecisionTree_garbage.model")        # 模型加载
    y2_pre = clf2.predict(X_test[:, 1:-1])
    print('Training finished')

    # 计算准确率和找到分类错误的索引
    sum1 = 0
    err_idx = []
    for i in range(len(y_test)):
        if y2_pre[i] == y_test[i]:
            sum1 += 1
        else:
            err_idx.append(i)
    print('分类错误的下标索引值：{}'.format(err_idx))

    err_imgidx = []
    ful_res = []
    emp_res = []
    err_dict = dict()
    for i in range(len(err_idx)):
        err_imgidx.append(X_test[err_idx[i], 0])
    err_imgidx.sort()

    for i in range(len(err_imgidx)):
        if err_imgidx[i] <= 309:
            ful_res.append(err_imgidx[i])
        else:
            emp_res.append(err_imgidx[i])
        err_dict['full'] = ful_res
        err_dict['empty'] = emp_res
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')

# 极端随机树 ExtraTree
    print('ExtraTrees train begin')
    clf3 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    y_pre1 = clf3.fit(X_train[:, 1:-1], y_train)
    # joblib.dump(clf3, "ExtraTrees_garbage.model")        # 模型保存
    # clf = joblib.load("ExtraTrees_garbage.model")        # 模型加载
    y_pre = clf3.predict(X_test[:, 1:-1])
    print('Training finished')

    # 计算准确率和找到分类错误的索引
    sum1 = 0
    err_idx = []
    for i in range(len(y_test)):
        if y_pre[i] == y_test[i]:
            sum1 += 1
        else:
            err_idx.append(i)
    print('分类错误的下标索引值：{}'.format(err_idx))

    err_imgidx = []
    ful_res = []
    emp_res = []
    err_dict = dict()
    for i in range(len(err_idx)):
        err_imgidx.append(X_test[err_idx[i], 0])
    err_imgidx.sort()

    for i in range(len(err_imgidx)):
        if err_imgidx[i] <= 309:
            ful_res.append(err_imgidx[i])
        else:
            emp_res.append(err_imgidx[i])
        err_dict['full'] = ful_res
        err_dict['empty'] = emp_res
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')


