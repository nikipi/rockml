import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 创建数据集
examDict = {"学习时间": [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75,
                     2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
            '分数': [10, 22, 13, 43, 20, 22, 33, 50, 62,
                   48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}

# 转换为DataFrame的数据格式


examDf = DataFrame(examDict)

print(examDf)

plt.scatter(examDf.分数,examDf.学习时间,color='b',label='Exam Data')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()

rDF=examDf.corr()
print(rDF)

'''
k 折交叉验证（k-fold cross validation）
静态的「留出法」对数据的划分方式比较敏感，有可能不同的划分方式得到了不同的模型。「k 折交叉验证」是一种动态验证的方式，这种方式可以降低数据划分带来的影响。具体步骤如下：

1.将数据集分为训练集和测试集，将测试集放在一边
2.将训练集分为 k 份
3.每次使用 k 份中的 1 份作为验证集，其他全部作为训练集。
4.通过 k 次训练后，我们得到了 k 个不同的模型。
5.评估 k 个模型的效果，从中挑选效果最好的超参数
6.使用最优的超参数，然后将 k 份数据全部作为训练集重新训练模型，得到最终模型。
'''

# 将原数据集拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(examDf.学习时间, examDf.分数, train_size=.8)
# X_train为训练数据标签,X_test为测试数据标签,exam_X为样本特征,exam_y为样本标签，train_size 训练数据占比

print("原始数据特征:", examDf.学习时间.shape,
      ",训练数据特征:", X_train.shape,
      ",测试数据特征:", X_test.shape)

print("原始数据标签:", examDf.分数.shape,
      ",训练数据标签:", Y_train.shape,
      ",测试数据标签:", Y_test.shape)

# 散点图
plt.scatter(X_train, Y_train, color="blue", label="train data")
plt.scatter(X_test, Y_test, color="red", label="test data")

# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Pass")
# 显示图像


plt.show()

linreg = LinearRegression()

# reshape如果行数=-1的话可以使我们的数组所改的列数自动按照数组的大小形成新的数组
# 因为model需要二维的数组来进行拟合但是这里只有一个特征所以需要reshape来转换为二维数组
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

model=linreg.fit(X_train, Y_train)
print (model)

model = LinearRegression()

# 对于模型错误我们需要把我们的训练集进行reshape操作来达到函数所需要的要求
# model.fit(X_train,Y_train)


model.fit(X_train, Y_train)


a=model.intercept_#截距
b=model.coef_  # 回归系数

print("最佳拟合线:截距", a, ",回归系数：", b)

# 训练数据的预测值
y_train_pred = model.predict(X_train)
# 绘制最佳拟合线：标签用的是训练数据的预测值y_train_pred
plt.plot(X_train, y_train_pred, color='black', linewidth=3, label="best line")

# 测试数据散点图
plt.scatter(X_train, Y_train, color="blue", label="train data")
plt.scatter(X_test, Y_test, color="red", label="test data")

# 添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
# 显示图像


plt.show()

score = model.score(X_test, Y_test)

print(score)


###############
#K FOLD########
###############

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))



##########################
#train/test split for reg#
##########################

# Import necessary modules
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create training and test sets
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squared Error: {}".format(rmse))
