#!/usr/bin/env python
# coding: utf-8

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

# read the csv file 
df = pd.read_csv("cardio_train.csv", sep=";")

#print the first 5 rows
df.head()

#print the last 5 rows
#df.tail()

#print the last 10 rows
#df.tail(10)



# PERFORM EXPLORATORY DATA ANALYSIS

# Drop id
df = df.drop(columns = 'id')

#static check that id column was removed
df.head()

# since the age is given in days, we convert it into years
df['age'] = df['age']/365

#check age has been edited
df.head()

# Statistical summary of the dataframe
df.describe()

#make histogram plots
df.hist(bins = 30, figsize = (20,20), color = 'b')
plt.show()

# get the correlation matrix
corr_matrix = df.corr()
corr_matrix

# plotting the correlation matrix
plt.figure(figsize = (16,16))
sns.heatmap(corr_matrix, annot = True) #annot = False if no numbers
plt.show()


# CREATE TRAINING AND TESTING DATASET

df

# split the dataframe into target and features
y = df['cardio']
X = df.drop(columns =['cardio'])

#check the shapes
X.shape

y.shape

#spliting the data in to test and train sets; standard 80% for training 20% for testing (hold-out dataset)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#sanity check; train shuffles data -set to False if no shuffling
X_train.shape

y_train.shape

X_test.shape

y_test.shape

X_train


# XG-BOOST ALGORITHM TO SOLVE CLASSIFICATION TYPE PROBLEMS

# TRAIN AN XG-BOOST CLASSIFIER IN SK-LEARN


get_ipython().system('pip install xgboost')

from xgboost import XGBClassifier

# Train an XGBoost classifier model
# hyper-parameter defs (many more available)
# eval_metric = error; to reduce the errors
# learning rate = scalng factor, how agressive you are trying to learn a certian task
# max depth of tree; 1 is shallow
# n_estimates is the number of trees
xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 20, n_estimators = 10)
xgb_classifier.fit(X_train, y_train)


# TEST XGBOOST CLASSIFIER TO PERFORM INFERENCE

# predict the score of the trained model using the testing dataset
result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))

# make predictions on the test data
y_predict = xgb_classifier.predict(X_test)
y_predict
# 0 => based on these features, a guess that the patient does NOT have cardio. disease
# 1 => the patient, in fact, had cardiovascular disease


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# precision is the ratio of TP/(TP+FP)
# recall is the ratio of TP/(TP+FN)
# F-beta score can be interpreted as a weighted harmonic mean of the precision and recall
# where an F-beta score reaches its best value at 1 and worst score at 0. 


from sklearn.metrics import confusion_matrix
# visualization of model performance
# prediction = rows, ground truth = columns, (0,0) & (1,1) are the correctly classified results
# (0,1) & (1,0) are misclassified
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)



#Another classification example
# Goals: Using "Diabetes.csv" dataset, perform the following:
# - 1. Load the “diabetes.csv” dataset using Pandas
# - 2. Split the data into 80% for training and 20% for testing 
# - 3. Train an XG-Boost classifier model using SK-Learn Library
# - 4. Assess trained model performance
# - 5. Plot the confusion matrix
# - 6. Print the classification report

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Include the full link to the csv file containing your dataset
df = pd.read_csv('diabetes.csv')

#table info
df.info()

# Plot Histogram
df.hist(bins = 30, figsize = (20,20), color = 'b');

# Plot the correlation matrix
correlations = df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True);

y = df['Outcome']
y

X = df.drop(['Outcome'], axis = 1)
X


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#sanity check
X_train.shape
X_test.shape

# Train an XGBoost classifier model 

xgb_classifier = XGBClassifier(objective ='binary:logistic', eval_metric = 'error', learning_rate = 0.1, max_depth = 1, n_estimators = 10, use_label_encoder=False)
xgb_classifier.fit(X_train, y_train)

# predict the score of the trained model using the testing dataset
result = xgb_classifier.score(X_test, y_test)
print("Accuracy : {}".format(result))

# make predictions on the test data
y_predict = xgb_classifier.predict(X_test)
y_predict

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# precision is the ratio of TP/(TP+FP)
# recall is the ratio of TP/(TP+FN)
# F-beta score can be interpreted as a weighted harmonic mean of the precision and recall
# where an F-beta score reaches its best value at 1 and worst score at 0. 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, fmt = 'd', annot = True)



