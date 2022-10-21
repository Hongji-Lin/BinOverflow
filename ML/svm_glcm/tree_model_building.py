# encoding: utf-8
# @author: Evan
# @file: tree_model_building.py
# @time: 2022/10/20 16:08
# @desc: Tree_Model

# 模型二：随机森林，决策树，ExtraTree分类器
import time
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# 读取数据
data = pd.read_csv('../data/garbage_33dim_sorted_data.csv')  # 49×16
data = np.array(data)
rate = []
epoch = 1
start_time = time.time()


def random_forest(X_train, y_train, X_test, y_test):
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
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')


# 决策树
def decision_tree(X_train, y_train, X_test, y_test):
    print('DecisionTree train begin')
    clf2 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    clf2.fit(X_train[:, 1:-1], y_train)
    # joblib.dump(clf2, "DecisionTree_garbage.model")        # 模型保存
    # clf = joblib.load("DecisionTree_garbage.model")        # 模型加载
    y2_pre = clf2.predict(X_test[:, 1:-1])
    print('Training finished')

    sum1 = 0
    err_idx = []
    for i in range(len(y_test)):
        if y2_pre[i] == y_test[i]:
            sum1 += 1
        else:
            err_idx.append(i)
    print('分类错误的下标索引值：{}'.format(err_idx))

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
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')


# 极端随机树 ExtraTree
def extra_tree(X_train, y_train, X_test, y_test):
    print('ExtraTrees train begin')
    clf3 = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    clf3.fit(X_train[:, 1:-1], y_train)
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
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')


# 梯度提升 GradientBoosting
def gradient_boost(X_train, y_train, X_test, y_test):
    print('GradientBoosting train begin')
    params = {
        "n_estimators": 500,
        "learning_rate": 0.01,
    }
    clf4 = ensemble.GradientBoostingClassifier(**params)
    clf4.fit(X_train[:, 1:-1], y_train)
    # joblib.dump(clf3, "ExtraTrees_garbage.model")        # 模型保存
    # clf = joblib.load("ExtraTrees_garbage.model")        # 模型加载
    y_pre = clf4.predict(X_test[:, 1:-1])
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
    print('分类错误的集合：{}'.format(err_dict))

    end_time = time.time()

    print('len(y_test)', len(y_test))
    print('right sum', sum1)
    # print('rate[', i, '] ', sum1 / len(y_test))
    print("测试精度：{}".format(sum1 / len(y_test)))
    print("训练时间：{}".format(end_time - start_time))
    print('')


if __name__ == '__main__':
    for i in range(epoch):
        print("第{}轮训练开始".format(i + 1))
        # 划分训练集测试集
        X = data[:, :-1]
        y = data[:, -1]
        X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, random_state=0)
        random_forest(X1_train, y1_train, X1_test, y1_test)
        decision_tree(X1_train, y1_train, X1_test, y1_test)
        extra_tree(X1_train, y1_train, X1_test, y1_test)
        gradient_boost(X1_train, y1_train, X1_test, y1_test)

        # 交叉验证
        # scores1 = cross_val_score(clf1, X, y)
        # print(scores1.mean())

