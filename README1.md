# BinOverflow
 垃圾桶满仓溢出检测

# 纹理特征提取方法：灰度共生矩阵GLCM
# 图像色彩空间（RGB, HSV, HLS)
RGB色彩模式是工业界的一种颜色标准，是通过对红®、绿(G)、蓝(B)三个颜色通道的变化以及它们相互之间的叠加来得到各式各样的颜色的，
RGB即是代表红、绿、蓝三个通道的颜色，这个标准几乎包括了人类视力所能感知的所有颜色，是运用最广的颜色系统之一。

HLS 也有三个分量，hue（色相）、saturation（饱和度）、lightness（亮度）。
HLS 和 HSV 的区别就是最后一个分量不同，HLS 中的 L 分量为亮度，亮度为100，表示白色，亮度为0，表示黑色
色相
就是颜色的色彩相貌，说简单点就是这个什么颜色，比如：蓝色、青色、紫色等就是色相
饱和度
就是色彩的纯度，饱和度越高色彩越浓、饱和度越低色彩越淡。
明度
就是色彩的明亮程度，色彩的明度越高，色彩越亮：色彩的明度越暗，色彩越暗。
H=色相决定是什么颜色；
S=纯度决定颜色浓淡；
L=明度决定照射在颜色上的白光有多亮。


Gabor_feature：这个函数是用来计算灰度共生矩阵的
（即对图像进行Gabor变换，得到处理后的图像之后，再进行对应特征提取

Spectral_Feature：
Feature_Color?

# svm_image_feature： 这个文件是用来提取特征并合并特征的
testfeature：调用feature_computer：返回4个变量
reRGBandHLS: 从RGB中读取通道，从HLS中读入通道。返回R,G,B、H、L、S: N×6的矩阵




svm_model_building


garbage_33dim_data.csv：这个文件是所有图片计算灰度共生矩阵之后的结果


# svm_model_building：
"""
svm_clf = 分类器对象
"""
# sklearn.model_selection.train_test_split 随机划分训练集和测试集
参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

# svm.SVC()参数说明
C ： float，可选(默认值= 1.0)
错误术语的惩罚参数C。
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
这样对训练集测试时准确率很高，但泛化能力弱。
C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

kernel ： string，optional(default =‘rbf’)
核函数类型，str类型，默认为’rbf’。可选参数为：
’linear’：线性核函数
‘poly’：多项式核函数
‘rbf’：径像核函数/高斯核
‘sigmod’：sigmod核函数

probability ： 布尔值，可选(默认=False)
是否启用概率估计，bool类型，可选参数，默认为False，这必须在调用fit()之前启用，并且会fit()方法速度变慢。

tol ： float，optional(默认值= 1e-3)
svm停止训练的误差精度，float类型，可选参数，默认为1e^-3。

max_iter ： int，optional(默认值= -1)
最大迭代次数，int类型，默认为-1，表示不限制。

degree ： int，可选(默认= 3)
多项式核函数的阶数，int类型，可选参数，默认为3。
这个参数只对多项式核函数有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数。

# GridSearchCV(estimator, param_grid, cv=None)网格搜索参数说明
estimator：选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数
param_grid：需要最优化的参数的取值，值为字典或者列表，例如：param_grid = param_test1, param_test1 = {'n_estimators' : range(10,71,10)}
cv = None：交叉验证参数，默认None，使用五折交叉验证。指定fold数量，默认为5(之前版本为3)，也可以是yield训练/测试数据的生成器。 

# Joblib
是一个可以简单地将Python代码转换为并行计算模式的软件包，它可非常简单并行我们的程序，从而提高计算速度。
保存训练好的Model：joblib.dump(clf, ‘save/clf.pkl’)
读取Model：clf2 = joblib.load(‘save/clf.pkl’)

clf.fit(X_train, y_train)： 用训练数据拟合分类器模型
clf.predict(X_test)：用训练好的分类器去预测[X_test]数据的标签[]