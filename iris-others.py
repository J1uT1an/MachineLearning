import numpy as np
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree


def iris_type(s):
	it = {b'setosa': 0, b'versicolor': 1, b'virginica': 2}
	return it[s]


data_path = 'iris_data.csv'
data = np.loadtxt(data_path, dtype=float, delimiter=',', converters={4: iris_type})

x, y = np.split(data, (4,), axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)


def train_knn(x_train, y_train):
	# 创建k近邻分类器实例
	clf = neighbors.KNeighborsClassifier(n_neighbors=3)
	clf.fit(x_train, y_train.ravel())
	return clf


def train_decision_tree(x_train, y_train):
	# 创建决策树分类器实例
	clf = tree.DecisionTreeClassifier()
	clf.fit(x_train, y_train.ravel())
	return clf


def train_naive_bayes(x_train, y_train):
	# 创建朴素贝叶斯分类器实例
	clf = naive_bayes.GaussianNB()
	clf.fit(x_train, y_train.ravel())
	return clf


# 训练k近邻分类器
knn = train_knn(x_train, y_train)

# 训练决策树分类器
decision_tree = train_decision_tree(x_train, y_train)

# 训练朴素贝叶斯分类器
naive_bayes = train_naive_bayes(x_train, y_train)


def print_accuracy(clf, x_train, y_train, x_test, y_test):
	print('Training prediction accuracy: %.3f' % clf.score(x_train, y_train))
	print('Testing prediction accuracy: %.3f' % clf.score(x_test, y_test))


# 打印k近邻分类器的准确率
print("K-Nearest Neighbors:")
print_accuracy(knn, x_train, y_train, x_test, y_test)

# 打印决策树分类器的准确率
print("\nDecision Tree:")
print_accuracy(decision_tree, x_train, y_train, x_test, y_test)

# 打印朴素贝叶斯分类器的准确率
print("\nNaive Bayes:")
print_accuracy(naive_bayes, x_train, y_train, x_test, y_test)
