
# coding: utf-8

# In[8]:


import numpy as np
import scipy.io as scio
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.externals import joblib
np.set_printoptions(suppress= True)


# In[2]:


# 加载数据
cloudy = scio.loadmat('data_cloud.mat')['cloudy']
clear = scio.loadmat('data_cloud.mat')['clear']
# cloudy和clear合并一个数组
data = np.concatenate((cloudy,clear),axis = 0)

# 提取分类与特征
classification = data[:,2].copy()
classification = np.where(classification > 0,1,0)
features = np.delete(data,[2,3],axis=1)
#交叉验证
x_train,x_test,y_train,y_test = train_test_split(features,classification,test_size = 0.25,random_state = 10)


# In[14]:


# 迭代次数与learning rate
param_test1 = {'n_estimators': list(range(100,201,50)),'learning_rate': np.linspace(0.05,0.3,6)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(max_depth= 5, 
#                                                   learning_rate= 0.1, 
#                                                   n_estimators= 150, 
                                                  n_jobs= -1, 
                                                  gamma= 0, 
                                                  reg_alpha= 0, 
                                                  reg_lambda= 1,
                                                  seed= 10),
                        param_grid = param_test1, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch1.fit(x_train,y_train)
gsearch1.cv_results_, gsearch1.best_score_, gsearch1.best_params_


# In[22]:


# max_depth
param_test2 = {'max_depth': np.arange(5,11)}
gsearch2 = GridSearchCV(estimator = XGBClassifier(
#                                                   max_depth= 5, 
                                                  learning_rate= 0.3, 
                                                  n_estimators= 200, 
                                                  n_jobs= -1, 
                                                  gamma= 0, 
                                                  reg_alpha= 0, 
                                                  reg_lambda= 1,
                                                  seed= 10),
                        param_grid = param_test2, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch2.fit(x_train,y_train)
gsearch2.cv_results_, gsearch2.best_score_, gsearch2.best_params_


# In[23]:


# min_child_weight
param_test3 = {'min_child_weight': np.arange(1,1002,200)}
gsearch3 = GridSearchCV(estimator = XGBClassifier(
                                                  max_depth= 10, 
                                                  learning_rate= 0.3, 
                                                  n_estimators= 200, 
                                                  n_jobs= -1, 
                                                  gamma= 0, 
                                                  reg_alpha= 0, 
                                                  reg_lambda= 1,
#                                                   min_child_weight = 300,
                                                  seed= 10),
                        param_grid = param_test3, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch3.fit(x_train,y_train)
gsearch3.cv_results_, gsearch3.best_score_, gsearch3.best_params_


# In[27]:


# gamma
param_test4 = {'gamma': np.linspace(0,0.5,5)}
gsearch4 = GridSearchCV(estimator = XGBClassifier(
                                                  max_depth= 10, 
                                                  learning_rate= 0.3, 
                                                  n_estimators= 200, 
                                                  n_jobs= -1, 
#                                                   gamma= 0, 
                                                  reg_alpha= 0, 
                                                  reg_lambda= 1,
                                                  min_child_weight = 200,
                                                  seed= 10),
                        param_grid = param_test4, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch4.fit(x_train,y_train)
gsearch4.cv_results_, gsearch4.best_score_, gsearch4.best_params_


# In[29]:


# subsample
param_test5 = {'subsample': [0.6,0.7,0.8,0.9,1.0]}
gsearch5 = GridSearchCV(estimator = XGBClassifier(
                                                  max_depth= 10, 
                                                  learning_rate= 0.3, 
                                                  n_estimators= 200, 
                                                  n_jobs= -1, 
                                                  gamma= 0.25, 
                                                  reg_alpha= 0, 
                                                  reg_lambda= 1,
                                                  min_child_weight = 200,
                                                  seed= 10),
                        param_grid = param_test5, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch5.fit(x_train,y_train)
gsearch5.cv_results_, gsearch5.best_score_, gsearch5.best_params_


# In[30]:


# colsample_bytree
param_test6 = {'colsample_bytree': [0.6,0.7,0.8,0.9,1.0]}
gsearch6 = GridSearchCV(estimator = XGBClassifier(
                                                  max_depth= 10, 
                                                  learning_rate= 0.3, 
                                                  n_estimators= 200, 
                                                  n_jobs= -1, 
                                                  gamma= 0.25, 
                                                  reg_alpha= 0, 
                                                  reg_lambda= 1,
                                                  min_child_weight = 200,
                                                  subsample = 1.0,
                                                  seed= 10),
                        param_grid = param_test6, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch6.fit(x_train,y_train)
gsearch6.cv_results_, gsearch6.best_score_, gsearch6.best_params_


# In[33]:


# 训练
xgb_skl = XGBClassifier(max_depth= 10, 
                        learning_rate= 0.1, 
                        n_estimators= 600, 
                        n_jobs= -1, 
                        gamma= 0.25, 
                        reg_alpha= 0, 
                        reg_lambda= 1,
                        min_child_weight = 200,
                        subsample = 1.0,
                        colsample_bytree = 1.0,
                        seed= 10)
xgb_skl.fit(x_train,y_train)
# 预测,测试集
y_pred = xgb_skl.predict(x_test)
accuracy_test = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy(test) : %.4f" %accuracy_test)
# 预测,训练集
y_pred = xgb_skl.predict(x_train)
accuracy_train = metrics.accuracy_score(y_train, y_pred)
print ("Accuracy(train) : %.4f" %accuracy_train)


# In[110]:


# 微调训练
xgb_skl = XGBClassifier(max_depth= 13, 
                        learning_rate= 0.1, 
                        n_estimators= 150, 
                        n_jobs= -1, 
                        gamma= 0.11, 
                        reg_alpha= 0, 
                        reg_lambda= 3,
                        min_child_weight = 1,
                        subsample = 1,
                        colsample_bytree = 0.9,
                        seed= 10)
xgb_skl.fit(x_train,y_train)
# 预测,测试集
y_pred = xgb_skl.predict(x_test)
accuracy_test = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy(test) : %.4f" %accuracy_test)
# 预测,训练集
y_pred = xgb_skl.predict(x_train)
accuracy_train = metrics.accuracy_score(y_train, y_pred)
print ("Accuracy(train) : %.4f" %accuracy_train)

