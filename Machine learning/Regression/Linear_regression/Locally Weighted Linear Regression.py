import numpy as np
import matplotlib.pyplot as plt



'''
局部加权线性回归（Locally Weighted Linear Regression, LWLR）是一种非参数算法。
与k近邻类似，它并没有模型参数，也没有训练过程，而是直接使用整个数据集来做预测。
它的核心思想是：在做预测时，更多地参考距离预测样本近的已知样本，而更少地参考距离预测样本远的已知样本。

'''

m=100
np.random.seed(42)
x_value=100*np.random.rand(m,1)
x_value=np.sort(x_value,axis=0)
y=7*np.sin(0.12*x_value)+x_value+2*np.random.randn(m,1)
plt.figure(figsize=(10,5))
plt.plot(x_value,y,'b.')
plt.show()

X = np.c_[np.ones([m, 1]), x_value]



#k是一个超参数，描述了随着已知样本与预测样本距离的增加，其重要程度减少的速率

def calculate_theta(x_test, k):
    # 构造矩阵 W
    W = np.eye(m, m)
    for i in range(m):
        W[i, i] = np.exp(- np.sum(np.square(X[i] - x_test)) / (2 * k ** 2))

    # 应用局部加权线性回归，求解 theta
    theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)

    return theta


def predict(x_test, k):
    theta = calculate_theta(x_test, k)
    y_pred = theta[0] + x_test * theta[1]
    return y_pred


print(predict(50, 5))


#拟合图形
test_count = 50
x_test_values = np.linspace(0, 100, test_count)


def lwlr(x_test_values, k):
    left_values = x_test_values - m / test_count / 2
    right_values = x_test_values + m / test_count / 2
    X_tests = np.c_[np.ones(test_count), x_test_values.reshape(-1, 1)]

    x_plots = []
    y_plots = []

    for t, l, r in zip(X_tests, left_values, right_values):
        theta = calculate_theta(t, k)

        x_test_points = np.array([[l], [r]])
        X_test = np.c_[np.ones([2, 1]), x_test_points]
        y_test_points = X_test.dot(theta)

        x_plots.extend(x_test_points)
        y_plots.extend(y_test_points)

    plt.plot(x_value, y, "b.")
    plt.plot(x_plots, y_plots, 'r-', linewidth=2)
    plt.title("k={}".format(k))


plt.figure(figsize=(12, 8))
ks = [100, 5, 1]  #k=5是合适的拟合参数
for index in range(len(ks)):
    plt.subplot(len(ks), 1, index + 1)
    lwlr(x_test_values, ks[index])
plt.tight_layout()
plt.show()