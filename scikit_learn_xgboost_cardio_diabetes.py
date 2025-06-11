#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE

# ![image-2.png](attachment:image-2.png)

# ![image.png](attachment:image.png)

# # TASK #2: IMPORT LIBARIES AND DATASETS

# In[40]:


# Binary classification problem (0, 1)
# Goal: To predict whether a patient has a cardiovascular disease or not
# import the necessary libraries. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#comment out when using light backgrounds
#from jupiterthemes import jtplot
#jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False)


# In[41]:


# read the csv file 
df = pd.read_csv("cardio_train.csv", sep=";")


# In[42]:


#print the first 5 rows
df.head()

#print the last 5 rows
#df.tail()

#print the last 10 rows
#df.tail(10)


# **PRACTICE OPPORTUNITY #1 [OPTIONAL]:**
# - **Display the last 5, 8, and 10 rows in the df DataFrame**

# In[ ]:





# # TASK #3: PERFORM EXPLORATORY DATA ANALYSIS

# In[43]:


# Drop id
df = df.drop(columns = 'id')


# In[44]:


#static check that id column was removed
df.head()


# In[45]:


# since the age is given in days, we convert it into years
df['age'] = df['age']/365


# In[46]:


df.head()


# In[47]:


# Statistical summary of the dataframe
df.describe()


# In[49]:


df.hist(bins = 30, figsize = (20,20), color = 'b')
plt.show()


# In[16]:


# get the correlation matrix
corr_matrix = df.corr()
corr_matrix


# In[52]:


# plotting the correlation matrix
plt.figure(figsize = (16,16))
sns.heatmap(corr_matrix, annot = True) #annot = False if no numbers
plt.show()


# # TASK #4: CREATE TRAINING AND TESTING DATASET

# In[53]:


df


# In[54]:


# split the dataframe into target and features
y = df['cardio']
X = df.drop(columns =['cardio'])


# In[55]:


X.shape


# In[56]:


y.shape


# In[57]:


#spliting the data in to test and train sets; standard 80% for training 20% for testing (hold-out dataset)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[58]:


#sanity check; train shuffles data -set to False if no shuffling
X_train.shape


# In[59]:


y_train.shape


# In[60]:


X_test.shape


# In[61]:


y_test.shape


# In[62]:


X_train


# # TASK #5: UNDERSTAND XG-BOOST ALGORITHM TO SOLVE CLASSIFICATION TYPE PROBLEMS

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #6: TRAIN AN XG-BOOST CLASSIFIER IN SK-LEARN

# In[64]:


get_ipython().system('pip install xgboost')


# In[99]:


from xgboost import XGBClassifier


# In[100]:


# Train an XGBoost classifier model
# hyper-parameter defs (many more available)
# eval_metric = error; to reduce the errors
# learning rate = scalng factor, how agressive you are trying to learn a certian task
# max depth of tree; 1 is shallow
# n_estimates is the number of trees
xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 20, n_estimators = 10)
xgb_classifier.fit(X_train, y_train)


# # TASK #7: TEST XGBOOST CLASSIFIER TO PERFORM INFERENCE

# In[101]:


# predict the score of the trained model using the testing dataset
result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))


# In[102]:


# make predictions on the test data
y_predict = xgb_classifier.predict(X_test)
y_predict
# 0 => based on these features, a guess that the patient does NOT have cardio. disease
# 1 => the patient, in fact, had cardiovascular disease


# In[103]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# precision is the ratio of TP/(TP+FP)
# recall is the ratio of TP/(TP+FN)
# F-beta score can be interpreted as a weighted harmonic mean of the precision and recall
# where an F-beta score reaches its best value at 1 and worst score at 0. 


# In[104]:


from sklearn.metrics import confusion_matrix
# visualization of model performance
# prediction = rows, ground truth = columns, (0,0) & (1,1) are the correctly classified results
# (0,1) & (1,0) are misclassified
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)


# **PRACTICE OPPORTUNITY #2 [OPTIONAL]:**
# - **Try a larger max_depth and retrain the model**
# - **Assess the performance of the trained model**
# - **What do you conclude?**

# In[ ]:





# # FINAL CAPSTONE PROJECT

# In[ ]:





# Using "Diabetes.csv" dataset, perform the following:
# - 1. Load the “diabetes.csv” dataset using Pandas
# - 2. Split the data into 80% for training and 20% for testing 
# - 3. Train an XG-Boost classifier model using SK-Learn Library
# - 4. Assess trained model performance
# - 5. Plot the confusion matrix
# - 6. Print the classification report

# In[112]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes.csv", sep= ',')


# In[113]:


# split the dataframe into target and features
y = df['Outcome']
X = df.drop(columns =['Outcome'])


# In[118]:


#y.shape
#X.shape


# In[119]:


#spliting the data in to test and train sets; standard 80% for training 20% for testing (hold-out dataset)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[120]:


from xgboost import XGBClassifier


# In[128]:


# Train an XGBoost classifier model
# hyper-parameter defs (many more available)
# eval_metric = error; to reduce the errors
# learning rate = scalng factor, how agressive you are trying to learn a certian task
# max depth of tree; 1 is shallow
# n_estimates is the number of trees
xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 5, n_estimators = 10)
xgb_classifier.fit(X_train, y_train)


# In[129]:


# predict the score of the trained model using the testing dataset
result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))


# In[130]:


y_predict = xgb_classifier.predict(X_test)
y_predict


# In[131]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[127]:


from sklearn.metrics import confusion_matrix
# visualization of model performance
# prediction = rows, ground truth = columns, (0,0) & (1,1) are the correctly classified results
# (0,1) & (1,0) are misclassified
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)


# # PRACTICE OPPORTUNITIES SOLUTION

# **PRACTICE OPPORTUNITY #1 SOLUTION:**
# - **Display the last 5, 8, and 10 rows in the df DataFrame**

# In[109]:


df.tail()


# In[110]:


df.tail(8)


# In[111]:


df.tail(10)


# **PRACTICE OPPORTUNITY #2 SOLUTION:**
# - **Try a much larger max_depth and retrain the model**
# - **Assess the performance of the trained model**
# - **What do you conclude?**

# In[ ]:


# Train an XGBoost classifier model 

xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 10, n_estimators = 10, use_label_encoder=False)
xgb_classifier.fit(X_train, y_train)


# make predictions on the test data
y_predict = xgb_classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)


# # FINAL CAPSTONE PROJECT SOLUTION

# In[132]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[133]:


# You have to include the full link to the csv file containing your dataset
df = pd.read_csv('diabetes.csv')


# In[134]:


df.info()


# In[135]:


# Plot Histogram
df.hist(bins = 30, figsize = (20,20), color = 'b');


# In[136]:


# Plot the correlation matrix
correlations = df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True);


# In[137]:


y = df['Outcome']
y


# In[138]:


X = df.drop(['Outcome'], axis = 1)
X


# In[139]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[140]:


X_train.shape


# In[141]:


X_test.shape


# In[142]:


# Train an XGBoost classifier model 

xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 1, n_estimators = 10, use_label_encoder=False)
xgb_classifier.fit(X_train, y_train)


# In[143]:


# predict the score of the trained model using the testing dataset
result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))


# In[144]:


# make predictions on the test data
y_predict = xgb_classifier.predict(X_test)
y_predict


# In[145]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# precision is the ratio of TP/(TP+FP)
# recall is the ratio of TP/(TP+FN)
# F-beta score can be interpreted as a weighted harmonic mean of the precision and recall
# where an F-beta score reaches its best value at 1 and worst score at 0. 


# In[146]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)


# # EXCELLENT JOB!
