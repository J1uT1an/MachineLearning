import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import svm


def iris_type(s):
	it = {b'setosa': 0, b'versicolor': 1, b'virginica': 2}
	return it[s]


data_path = 'iris_data.csv'
data = np.loadtxt(data_path, dtype=float, delimiter=',', converters={4: iris_type})

x, y = np.split(data, (4,), axis=1)
x = x[:, 1:3]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)


def train_linear_svm(x_train, y_train):
	# 创建线性 SVM 分类器实例
	clf = svm.SVC(C=0.2, kernel='linear', decision_function_shape='ovr')
	clf.fit(x_train, y_train.ravel())
	return clf


def train_kernelized_svm(x_train, y_train):
	# 创建核化 SVM 分类器实例
	clf = svm.SVC(C=0.5, kernel='rbf')
	clf.fit(x_train, y_train.ravel())
	return clf


# 训练线性 SVM 分类器
linear_svm = train_linear_svm(x_train, y_train)

# 训练核化 SVM 分类器
kernelized_svm = train_kernelized_svm(x_train, y_train)


def print_accuracy(clf, x_train, y_train, x_test, y_test):
	print('Training prediction accuracy: %.3f' % clf.score(x_train, y_train))
	print('Testing prediction accuracy: %.3f' % clf.score(x_test, y_test))


def draw(clf, x):
	iris_feature = ['sepal length', 'sepal width', 'petal length', 'petal width']
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
	grid_test = np.stack((x1.flat, x2.flat), axis=1)
	
	# 计算数据点到决策超平面的距离
	z = clf.decision_function(grid_test)
	grid_hat = clf.predict(grid_test)
	grid_hat = grid_hat.reshape(x1.shape)
	
	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
	
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)
	plt.xlabel(iris_feature[0], fontsize=20)
	plt.ylabel(iris_feature[1], fontsize=20)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.title('SVM Classification', fontsize=30)
	plt.grid()
	plt.show()


# 打印线性 SVM 的准确率和绘制决策边界
print("Linear SVM:")
print_accuracy(linear_svm, x_train, y_train, x_test, y_test)
draw(linear_svm, x)

# 打印核化 SVM 的准确率和绘制决策边界
print("\nKernelized SVM:")
print_accuracy(kernelized_svm, x_train, y_train, x_test, y_test)
draw(kernelized_svm, x)
