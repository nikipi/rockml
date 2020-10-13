#######################
#GridSearchCV##########
#######################


# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
print(c_space)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


######################
##RandomizedSearch####
######################




# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],   # set list
              "max_features": randint(1, 9),  #set distribution
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree,param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


#criterion:特征选择的标准，有信息增益和基尼系数两种，使用信息增益的是ID3和C4.5算法（使用信息增益比），使用基尼系数的CART算法，默认是gini系数。

#splitter:特征切分点选择标准，决策树是递归地选择最优切分点，spliter是用来指明在哪个集合上来递归，有“best”和“random”两种参数可以选择，best表示在所有特征上递归，适用于数据集较小的时候，random表示随机选择一部分特征进行递归，适用于数据集较大的时候。

#max_depth:决策树最大深度，决策树模型先对所有数据集进行切分，再在子数据集上继续循环这个切分过程，max_depth可以理解成用来限制这个循环次数。

#min_samples_split:子数据集再切分需要的最小样本量，默认是2，如果子数据样本量小于2时，则不再进行下一步切分。如果数据量较小，使用默认值就可，如果数据量较大，为降低计算量，应该把这个值增大，即限制子数据集的切分次数。

#min_samples_leaf:叶节点（子数据集）最小样本数，如果子数据集中的样本数小于这个值，那么该叶节点和其兄弟节点都会被剪枝（去掉），该值默认为1。

#min_weight_fraction_leaf:在叶节点处的所有输入样本权重总和的最小加权分数，如果不输入则表示所有的叶节点的权重是一致的。

#max_features:特征切分时考虑的最大特征数量，默认是对所有特征进行切分，也可以传入int类型的值，表示具体的特征个数；也可以是浮点数，则表示特征个数的百分比；还可以是sqrt,表示总特征数的平方根；也可以是log2，表示总特征数的log个特征。