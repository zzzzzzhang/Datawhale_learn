
# coding: utf-8

# In[27]:


import numpy as np
import scipy.io as scio
from sklearn.ensemble import GradientBoostingClassifier
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


# In[3]:


# 提取分类与特征
classification = data[:,2].copy()
classification = np.where(classification > 0,1,0)
features = np.delete(data,[2,3],axis=1)
#交叉验证
x_train,x_test,y_train,y_test = train_test_split(features,classification,test_size = 0.25,random_state = 10)


# In[61]:


# 调参_1 n_estimators
param_test1 = {'n_estimators': list(range(100,201,50))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,min_samples_leaf=20,
                                                               max_depth=5,max_features=None, subsample= 1,random_state=10), 
                       param_grid = param_test1, scoring='accuracy',iid = False,n_jobs = 3)
gsearch1.fit(x_train,y_train)
# 显示调参结果
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# In[59]:


# 调参_2 max_depth
param_test2 = {'max_depth': list(range(4,11))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators = 200,learning_rate=0.1, min_samples_split=300,
                                                               min_samples_leaf=20,max_features=None, subsample= 1,random_state=10), 
                       param_grid = param_test2, scoring='accuracy',iid = False,n_jobs = -1)
gsearch1.fit(x_train,y_train)
# 显示调参结果
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# In[60]:


# 调参_3 subsample
param_test3 = {'subsample': [0.5,0.6,0.7,0.8,0.9,1.0]}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators = 200,learning_rate=0.1, min_samples_split=300,
                                                               min_samples_leaf=20, max_depth=5,max_features=None, random_state=10), 
                       param_grid = param_test3, scoring='accuracy',iid = False,n_jobs = -1)
gsearch1.fit(x_train,y_train)
# 显示调参结果
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# In[7]:


# 调参_4 min_samples_leaf min_samples_split
param_test4 = {'min_samples_leaf': list(range(30,31)),'min_samples_split': list(range(40,60,10))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators = 100,learning_rate=0.1,subsample= 1,
                                                               max_depth= 5, max_features=None, random_state=10), 
                       param_grid = param_test4, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch1.fit(x_train,y_train)
# 显示调参结果
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# In[24]:


# 调参_5 max_features
param_test5 = {'max_features': list(range(16,21,1))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators = 100,learning_rate=0.1,subsample= 0.9,
                                                               max_depth= 5, random_state=10,
                                                               min_samples_leaf = 100,min_samples_split = 300), 
                       param_grid = param_test5, scoring='accuracy',iid = False,n_jobs = -1,return_train_score = True)
gsearch1.fit(x_train,y_train)
# 显示调参结果
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# In[25]:


# 确定分类器与训练
gbdt = GradientBoostingClassifier(learning_rate = 0.05,random_state = 10,max_depth = 10,n_estimators = 200,min_samples_leaf = 20,
                                  min_samples_split = 50,subsample = 1,max_features = 20)
gbdt.fit(x_train,y_train)


# In[30]:


# 预测,测试集
y_pred = gbdt.predict(x_test)
accuracy_test = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy(test) : %.4g" %accuracy_test)
# 预测,训练集
y_pred = gbdt.predict(x_train)
accuracy_train = metrics.accuracy_score(y_train, y_pred)
print ("Accuracy(train) : %.4g" %accuracy_train)


# In[34]:


joblib.dump(gbdt,'gbdt_' + str(round(accuracy_test,4)) +  '.m')


# In[35]:


np.savez('data_cloud.npz',x_train,x_test,y_train,y_test)

