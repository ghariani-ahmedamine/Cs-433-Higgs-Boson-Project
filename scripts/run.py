#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Useful starting lines
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load the training data into feature matrix, class labels, and event ids:

# In[25]:


from proj1_helpers import *
from methods import *


# In[3]:




DATA_TRAIN_PATH = "../data/train.csv"  # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# ## Exploratory data analysis and preprocessing

# ### preliminary visualisation

# In[4]:


import pandas as pd 


# ### Dataset partition 

# the dataset comprises 30 features 29 continuous and 1 categorical.We divided the dataset according to the categorical feature, the partition will be further justified in the report.

# In[5]:


# -999 --> Nan
tX = missing_to_value(tX , np.nan)

#features partition
tX_0=tX[tX[:,22]==0]
tX_1=tX[tX[:,22]==1]
tX_2=tX[(tX[:,22]==2) | (tX[:,22]==3)]

#labels partition
y_0 =y[tX[:,22]==0]
y_1 =y[tX[:,22]==1]
y_2 =y[(tX[:,22]==2) | (tX[:,22]==3)]


# ### missing values

# In[39]:



df_0 = pd.DataFrame(tX_0)
df_1 = pd.DataFrame(tX_1)
df_2 = pd.DataFrame(tX_2)

missing_0 = df_0.isnull().sum() / df_0.shape[0]
missing_1 = df_1.isnull().sum() / df_1.shape[0]
missing_2 = df_2.isnull().sum() / df_2.shape[0]

#plot missing values for each partition

plt.figure()
missing_0.plot(kind="bar")
plt.title("Missing value ratio per feature (tx_0)")
plt.xlabel("features")
plt.ylabel("missing value ratio")

plt.figure()
missing_1.plot(kind="bar")
plt.title("Missing value ratio per feature (tx_1)")
plt.xlabel("features")
plt.ylabel("missing value ratio")

plt.figure()
missing_2.plot(kind="bar")
plt.title("Missing value ratio per feature (tx_2)")
plt.xlabel("features")
plt.ylabel("missing value ratio")


# In[6]:


#drop empty coloumns for each dataset
tX_0_prime = np.delete(tX_0,[0,4,5,6,12,23,24,25,26,27,28],axis=1)
tX_1_prime = np.delete(tX_1,[4,5,6,12,26,27,28],axis=1)

#replace missing values in the remaining coloumns by 
#the median of the coloumn for each dataset 

tX_0_non_missing = nan_to_median(tX_0_prime)
tX_1_non_missing = nan_to_median(tX_1_prime)
tX_2_non_missing = nan_to_median(tX_2)




# ### Removing highly correlated features (person_corr > 0.7)

# In[18]:


import seaborn as sns
 
df_0 = pd.DataFrame(tX_0_non_missing)
df_1 = pd.DataFrame(tX_1_non_missing)
df_2 = pd.DataFrame(tX_2_non_missing)

#computing and plotting the heatmap of the pearson correlation matrix 
#for each dataset

#computing the matrix
corr_0 = df_0.corr(method='pearson')
corr_1 = df_1.corr(method='pearson')
corr_2 = df_2.corr(method='pearson')

#plotting heatmaps

plt.figure(figsize=(30,15))
plt.title("Correlation matrix heatmap (tX_0)",size=(40))
sns.heatmap(corr_0, annot=True)
figure.savefig('heat_0.png', dpi=fig.dpi)

plt.figure(figsize=(30,15))
plt.title("Correlation matrix heatmap (tX_1)",size=(40))
sns.heatmap(corr_1, annot=True)
figure.savefig('heat_1.png', dpi=fig.dpi)

plt.figure(figsize=(30,15))
plt.title("Correlation matrix heatmap (tX_2)",size=(40))
sns.heatmap(corr_2, annot=True)
figure.savefig('heat_2.png', dpi=fig.dpi)


# In[19]:


# removing highly correlated features and apply standardization
# tX_0_final=standardize(np.delete(tX_0_non_missing,[2,5,6,18,19],axis=1))
# tX_1_final=standardize(np.delete(tX_1_non_missing,[2,6,12,17,18,19,22],axis=1))
# tX_2_final=standardize(np.delete(tX_2_non_missing,[2,5,9,16,19,23,29],axis=1))

# print(tX_0_final.shape,tX_1_final.shape,tX_2_final.shape)


# ## methods

# In[26]:





# ### PCA (least_squares)

# In[54]:


cross_validation_demo_least_squares(y_1,tX_1_non_missing,10,1,'least_squares')
cross_validation_demo_least_squares(y_2,tX_2_non_missing,10,1,'least_squares')
figure.savefig('pca.png', dpi=fig.dpi)


# ### ridge regression

# In[107]:


#ridge regression 
# tX_0 best lambda
lambdas = np.logspace(-8,-6,30)
mse_tr, mse_te,accuracy_,_= cross_validation_demo_lambdas(y_0,tX_0_non_missing,5,1,'ridge_regression',lambdas,0)
max_lambda = lambdas[np.argmax(accuracy_)]
print('tX_0: Best lambda:{} ; associated accuracy:{}'.format(max_lambda,accuracy_[np.argmax(accuracy_)]))         
        


# In[101]:


# tX_1 best lambda

lambdas = np.logspace(-8,-6,30)
mse_tr, mse_te,accuracy_,_= cross_validation_demo_lambdas(y_1,tX_1_non_missing,5,1,'ridge_regression',lambdas,0)
max_lambda = lambdas[np.argmax(accuracy_)]
print('tX_1: Best lambda:{} ; associated accuracy:{}'.format(max_lambda,accuracy_[np.argmax(accuracy_)]))         
        


# In[102]:


# tX_2 best lambda

lambdas = np.logspace(-8,-6,30)
mse_tr, mse_te,accuracy_,_= cross_validation_demo_lambdas(y_2,tX_2_non_missing,5,1,'ridge_regression',lambdas,0)
max_lambda = lambdas[np.argmax(accuracy_)]
print('tX_2: Best lambda:{} ; associated accuracy:{}'.format(max_lambda,accuracy_[np.argmax(accuracy_)]))         
        


# In[27]:


# tX_0 best degree

degrees = [3,4,5,6,7]
mse_tr, mse_te,accuracy_,_= cross_validation_demo_degrees(y_0, tX_0_non_missing ,5,1,'ridge_regression',1e-8,degrees)
max_degree = degrees[np.argmax(accuracy_)]
print('tX_0: Best degree:{} ; associated accuracy:{}'.format(max_degree,accuracy_[np.argmax(accuracy_)]))
a_0=accuracy_[np.argmax(accuracy_)]


# In[28]:


# tX_1 best degree

degrees = [3,4,5,6,7]
mse_tr, mse_te,accuracy_,_= cross_validation_demo_degrees(y_1, tX_1_non_missing ,5,1,'ridge_regression',1e-7,degrees)
max_degree = degrees[np.argmax(accuracy_)]
print('tX_1: Best degree:{} ; associated accuracy:{}'.format(max_degree,accuracy_[np.argmax(accuracy_)]))
a_1=accuracy_[np.argmax(accuracy_)]


# In[29]:


# tX_2 best degree

degrees = [3,4,5,6,7]
mse_tr, mse_te,accuracy_,_= cross_validation_demo_degrees(y_2, tX_2_non_missing ,5,1,'ridge_regression',1e-8,degrees)
max_degree = degrees[np.argmax(accuracy_)]
print('tX_2: Best degree:{} ; associated accuracy:{}'.format(max_degree,accuracy_[np.argmax(accuracy_)]))
a_2=accuracy_[np.argmax(accuracy_)]


# In[32]:



total_accuracy_expanded_ridge=(a_0 * np.shape(tX_0_non_missing)[0] + a_1 * np.shape(tX_1_non_missing)[0] + a_2 * np.shape(tX_2_non_missing)[0] ) / ( np.shape(tX_0_non_missing)[0] + np.shape(tX_1_non_missing)[0] + np.shape(tX_2_non_missing)[0])
print(total_accuracy_expanded_ridge)


# ### logistic regression

# In[83]:


degrees = [1,2]
mse_tr, mse_te,accuracy_,_= cross_validation_demo_degrees(y_0, tX_0_non_missing ,5,1,'logistic_regression',0,degrees)
max_degree = degrees[np.argmax(accuracy_)]
print(max_degree, accuracy_[np.argmax(accuracy_)] )


# ### Regularized logistic regression

# In[85]:


lambdas = np.logspace(-8,-6,2)
mse_tr, mse_te,accuracy_,_= cross_validation_demo_lambdas(y_0,tX_0_non_missing,5,1,'reg_logistic_regression',lambdas,0)
max_lambda = lambdas[np.argmax(accuracy_)]
print(max_lambda,accuracy_[np.argmax(accuracy_)])  


# In[86]:


degrees = [1,2]
mse_tr, mse_te,accuracy_,_= cross_validation_demo_degrees(y_0, tX_0_non_missing ,5,1,'reg_logistic_regression',1e-8,degrees)
max_degree = degrees[np.argmax(accuracy_)]
print(max_degree, accuracy_[np.argmax(accuracy_)] )    


# ## Generate predictions and save ouput in csv format for submission:

# In[33]:


DATA_TEST_PATH = "../data/test.csv"  # TODO: download train data and supply path here
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# In[41]:


#features partition
tX_0_test = np.array(tX_test[tX_test[:,22]==0])
tX_1_test = np.array(tX_test[tX_test[:,22]==1])
tX_2_test = np.array(tX_test[(tX_test[:,22]==2) | (tX_test[:,22]==3)])
#drop empty coloumns for each dataset
tX_0_prime_test = np.delete(tX_0_test,[0,4,5,6,12,23,24,25,26,27,28],axis=1)
tX_1_prime_test = np.delete(tX_1_test,[4,5,6,12,26,27,28],axis=1)

#replace missing values in the remaining coloumns by 
#the median of the coloumn for each dataset 

tX_0_non_missing_test = nan_to_median(tX_0_prime_test)
tX_1_non_missing_test = nan_to_median(tX_1_prime_test)
tX_2_non_missing_test = nan_to_median(tX_2_test)


# In[43]:


tX_0_poly = build_poly_dataset(tX_0_non_missing,7)
tX_0_poly_test = build_poly_dataset(tX_0_non_missing_test,7)
tX_1_poly = build_poly_dataset(tX_1_non_missing,4)
tX_1_poly_test = build_poly_dataset(tX_1_non_missing_test,4)
tX_2_poly = build_poly_dataset(tX_2_non_missing,7)
tX_2_poly_test = build_poly_dataset(tX_2_non_missing_test,7)

w_0 , _ = ridge_regression(y_0,tX_0_poly,1.1721022975334793e-08)
w_1 , _ = ridge_regression(y_1,tX_1_poly,1.7433288221999873e-07)
w_2 , _ = ridge_regression(y_2,tX_2_poly,2.2122162910704503e-08)


# In[45]:


y_pred=np.empty(tX_test.shape[0])
y_pred [tX_test[:,22]==0] = predict_labels(w_0,tX_0_poly_test)
y_pred [tX_test[:,22]==1] = predict_labels(w_1,tX_1_poly_test)
y_pred [(tX_test[:,22]==2) | (tX_test[:,22]==3)] = predict_labels(w_2,tX_2_poly_test)


# In[ ]:





# In[54]:


OUTPUT_PATH = "/res.csv"  # TODO: fill in desired name of output file for submission

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


# In[ ]:




