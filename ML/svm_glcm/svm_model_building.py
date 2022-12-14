import time

import joblib
import numpy as np
import pandas as pd
from sklearn import svm  # svm支持向量机
# from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# 模型一：SVM
# 读取数据
data = pd.read_csv('../data/garbage_33dim_sorted_data.csv')  # 49×16
data = np.array(data)
# print('data\n', data)
rate = []
Kern = 'linear'
epoch = 50

writer = SummaryWriter('runs/experiment')


start_time = time.time()
for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    # 划分训练集测试集
    x = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10, 50]}

    # 搭建svm模型
    svm_clf = svm.SVC(gamma='auto', class_weight='balanced')
    # svm_clf = svm.SVC(C=1.0, kernel=Kern, probability=True, tol=0.000001, max_iter=1000000, degree=3)
    clf = GridSearchCV(svm_clf, parameters, cv=5)

    # 训练模型
    # print('train begin...')
    # clf.fit(X_train[:, 1:-1], y_train)
    # joblib.dump(clf, "svm_garbage.model")
    # print('Training finished')

    # 测试模型
    clf = joblib.load("svm_garbage.model")
    y_pre = clf.predict(X_test[:, 1:-1])
    # print('y_pre', y_pre)
    # print('y_true', y_test)

    # 计算准确率和找到分类错误的索引
    sum1 = 0
    err_idx = []
    for i in range(len(y_test)):
        if y_pre[i] == y_test[i]:
            sum1 += 1
        else:
            err_idx.append(i)
    print(err_idx)

    err_num = []
    ful_res = []
    emp_res = []
    err_dict = dict()
    for i in range(len(err_idx)):
        err_num.append(X_test[err_idx[i], 0])
    err_num.sort()

    for i in range(len(err_num)):
        if err_num[i] <= 309:
            ful_res.append(err_num[i])
        else:
            emp_res.append(err_num[i])
        err_dict['full'] = ful_res
        err_dict['empty'] = emp_res

    print(err_dict)

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')


# PCA
"""
import Spectral_Feature as Spectral
# 读取数据
data = pd.read_csv('../data/garbage_33dim_data.csv')  # 49×11
data = np.array(data)
# print('data\n', data)
rate = []
Kern = 'linear'

for k in range(1):

    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(data[:, 1:17], data[:, 17], test_size=0.1, random_state=0)

    # 本数据集维度较高,最好进行PCA降维
    pca = PCA(n_components=15)
    # 也许一些原始特征也非常有用
    selection = SelectKBest(k="all")

    # 从主成分分析和单变量选择的建立评估器
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    # 使用组合特征来转换数据集
    X_features = combined_features.fit(X_train, y_train).transform(X_train)

    svm = svm.SVC(kernel="linear")

    # 进行网格搜索(over k, n_components and C)
    pipeline = Pipeline([("features", combined_features), ("svm", svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3],
                      features__univ_select__k=[1, 2],
                      svm__C=[0.1, 1, 10, 50])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_)
    joblib.dump(grid_search, "svm_garbage.model")
    print('Training finished')

    # 测试模型
    y_pre = grid_search.predict(X_test)
    print('y_pre', y_pre)
    print('y_true', y_test)
    print('X_test', X_test)

    # 计算准确率
    sum1 = 0
    for i in range(len(y_test)):
        if y_pre[i] == y_test[i]:
            sum1 += 1
    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    print('rate[', k, '] ', sum1 / len(y_test))
    print('')
"""