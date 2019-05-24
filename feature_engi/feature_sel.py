# -*- coding: utf-8 -*-

###得到返回至少含有90%特征信息的特征
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
iris = load_iris()
X, y = iris.data, iris.target
sp = SelectPercentile(f_classif, percentile= 90)
X_result = sp.fit_transform(X, y)
#可以看到哪些特征被保留
res = sp.get_support()

##计算特征相关系数
import pandas as pd
from sklearn import  datasets
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
iris = datasets.load_iris()
X = iris.data
y = iris.target
new_y = [y[i:i+1] for i in range(0, len(y), 1)]
data = np.hstack((X, new_y))
data_df = pd.DataFrame(data)
#0到3表示特征，4表示目标变量,画图查看相关性，如下图所示
sns.heatmap(data_df.corr(), annot= True, fmt= '.2f')

##使用GBDT选取重要特征
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
iris = datasets.load_iris()
gbdt_RFE = RFE(estimator=GradientBoostingClassifier(random_state= 123),n_features_to_select=2)
gbdt_RFE.fit(iris.data, iris.target)
gbdt_RFE.support_
#特征选择输出结果

##使用SVM选取重要特征
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
res = model.get_support()#the selected featrures

###使用ExtraTreesClassifier选取重要牲
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X,y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#作图观察特征重要性
plt.bar(range(10), importances[indices])
plt.xticks(range(X.shape[1]), indices)
plt.show()




##PipeLine
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)


from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
# generate some data to play with
X, y = samples_generator.make_classification(
    n_informative=5, n_redundant=0, random_state=42)
# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
prediction = anova_svm.predict(X)
anova_svm.score(X, y)  
# getting the selected features chosen by anova_filter
anova_svm['anova'].get_support()
# Another way to get selected features chosen by anova_filter
anova_svm.named_steps.anova.get_support()
# Indexing can also be used to extract a sub-pipeline.
sub_pipeline = anova_svm[:1]
sub_pipeline  
coef = anova_svm[-1].coef_
anova_svm['svc'] is anova_svm[-1]
coef.shape
sub_pipeline.inverse_transform(coef).shape
