# 导包
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import svm


# Step1.数据准备
# 从指定路径下加载数据；
# 对加载的数据进行数据分割，x_train, x_test, y_train, y_test分别表示训练集特征、训练集标签、测试集特征、测试集标签


# 将字符串转为整型，便于数据加载 transform the dtype from string into integer
def iris_type(s):
	# 将鸢尾花的类别转换为数值标签
	# Error: it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
	it = {b'setosa': 0, b'versicolor': 1, b'virginica': 2}
	return it[s]


# load data
data_path = 'iris_data.csv'  # 数据文件的路径。注意修改路径
data = np.loadtxt(data_path, dtype=float, delimiter=',', converters={4: iris_type}
                  )  # 数据文件路径 数据类型 数据分隔符 将第5列使用函数iris_type进行转换

# print(data)
# data为二维数组，数据分割
# data.shape=(150, 5) print(data.shape) => (150, 5)


# 分割特征和标签
# 要切分的数组
x, y = np.split(data, (4,), axis=1)  # 沿轴切分的位置，第5列开始往后为y 代表纵向分割，按列分割
x = x[:, 1:3]  # 在X中我们取前两列作为特征，为了后面的可视化。x[:,0:4]代表第一维(行)全取，第二维(列)取0~2
# x = x[:, :2]
# print(x)


# 划分训练集和测试集 参数是(所要划分的样本特征集,所要划分的样本结果,随机数种子,测试样本占比)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)


# Step2.模型搭建
# C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
# C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
# kernel = 'linear'时，为线性核；kenrel="rbf"时，为高斯核
# decision_function_shape = 'ovr'时，为one v rest，即一个类别与其他类别进行划分，
# decision_function_shape = 'ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。

# 定义创建SVM分类器实例函数
def classifier():
	# clf = svm.SVC(C=0.8,kernel='rbf',  gamma=50,decision_function_shape='ovr')
	# 参数(误差项惩罚系数(default = 1), 线性核, 决策函数)
	clf = svm.SVC(C=0.2, kernel='linear', decision_function_shape='ovr')
	return clf


# 创建SVM分类器实例
# clf = classifier() # 调用classifier()函数创建 82开，线性核
clf = svm.SVC(C=0.5, kernel='rbf')  # 55开，高斯核


# Step3.模型训练
def train(clf, x_train, y_train):
	# 使用训练集对分类器进行训练
	clf.fit(x_train, y_train.ravel())


# 训练分类器
train(clf, x_train, y_train)


# Step4.模型评估
# Judge wether a and b is equal.To computer the accueacy  value.*************
def show_accuracy(a, b, tip):
	# 计算并打印分类准确率
	acc = a.ravel() == b.ravel()
	print('%s  Accuracy:%.3f' % (tip, np.mean(acc)))


# 打印训练集和测试集的预测准确率
def print_accuracy(clf, x_train, y_train, x_test, y_test):
	# score(x_train,y_train):denote the accuracy on model of x_train,y_train
	print('training prediction:%.3f' % (clf.score(x_train, y_train)))
	print('test data prediction:%.3f' % (clf.score(x_test, y_test)))
	# 原始结果与预测结果进行对比 predict()表示对x_train样本进行预测，返回样本类别
	# on training dataset
	show_accuracy(clf.predict(x_train), y_train, 'training data')
	# on text dataset
	show_accuracy(clf.predict(x_test), y_test, 'testing data')
	# 计算决策函数的值，表示x到各分割平面的距离
	print('decision_function:\n', clf.decision_function(x_train))


# Step5.模型使用
def draw(clf, x):
	iris_feature = 'sepal  length', 'sepal  width', 'petal  lenght', 'petal  width'
	# begin draw 开始绘图
	# 第一列特征的最小值和最大值
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # The  min  and  max  value  of  the  first  coloum
	# 第二列特征的最小值和最大值
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # The  min  and  max  value  of  the  second  coloum
	# 生成用于绘图的样本点
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # to  show  samples
	grid_test = np.stack((x1.flat, x2.flat), axis=1)  # stack(): 沿着新的轴加入一系列数组
	print('grid_test:\n', grid_test)
	# 计算数据点到决策超平面的距离 distance  between  data  to  the  hyperplane
	z = clf.decision_function(grid_test)
	print('the  distance  to  decision  plane:\n', z)
	
	grid_hat = clf.predict(grid_test)  # predict  result【0,0,  ....,  2,2,2】
	print('grid_hat:\n', grid_hat)
	grid_hat = grid_hat.reshape(x1.shape)  # reshape  grid_hat和x1形状一致
	# 若3*3矩阵e，则e.shape()为3*3,表示3行3列
	
	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
	
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # pcolormesh(x,y,z,cmap)这里参数代入
	# x1, x2, grid_hat, cmap= cm_light 绘制的是背景。
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
	plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)  # 测试点
	plt.xlabel(iris_feature[0], fontsize=20)
	plt.ylabel(iris_feature[1], fontsize=20)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.title('svm  in  iris  data  classification', fontsize=30)
	plt.grid()
	plt.show()


# applying  model
# 打印分类准确率和绘制决策边界 print_accuracy(clf, x_train, y_train, x_test, y_test)
draw(clf, x)
