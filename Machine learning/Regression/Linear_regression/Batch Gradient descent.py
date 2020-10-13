import pandas as pd
import numpy as np
import ssl
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os



ssl._create_default_https_context = ssl._create_unverified_context
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


url = 'https://raw.githubusercontent.com/emilmont/Artificial-Intelligence-and-Machine-Learning/master/ML/ex1/ex1data1.txt'
data = pd.read_csv(url, header=None, names=['Population','Profit'])


print(data.head())

print(data.describe())

'''
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8)) # for the dataframe contains only two columns
plt.show()
'''

def costfuncarion(X,y,theta):
 inner=np.power(((X*theta.T)-y),2)
 return np.sum(inner)/(2*len(X))

# append a ones column to the front of the data set
data.insert(0,'Ones',1) #inset(column index,'column name', real number)
print(data.head())

cols=data.shape[1]
X= data.iloc[:,0:cols-1]
y= data.iloc[:,cols-1:cols]
print(X.head())
print(y.head())




# convert from data frames to numpy matrices
X= np.matrix(X.values)

y= np.matrix(y.values)

theta= np.matrix(np.array([0,0]))
print(theta)

print(X.shape, theta.shape, y.shape)


print(costfuncarion(X,y,theta))


def gradientDescent(X, y, theta, alpha, iters):

  temp = np.matrix(np.zeros(theta.shape))
  parameters = int(theta.ravel().shape[1])
  cost = np.zeros(iters)

  for i in range(iters):
      error = (X * theta.T) - y

      for j in range(parameters):

         term = np.multiply(error, X[:, j])

         temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

      theta = temp

      cost[i] = costfuncarion(X, y, theta)

  return theta,cost

alpha=0.01
iters=1000

g, cost= gradientDescent(X, y, theta, alpha, iters)
print(g)
print(costfuncarion(X,y,g))

#使用numpy的“linspace”函数在我们的数据范围内创建一系列均匀间隔的点，然后用我们的模型“评估”这些点，看预期的利润会是多少
x = np.linspace(data.Population.min(), data.Population.max(), 100)

f = g[0, 0] + (g[0, 1] * x)
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

