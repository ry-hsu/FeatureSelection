# -*- coding: utf-8 -*-
"""
Ryan Hsu
Class: CS 677
Date: 4/24/2022
Homework 6 Problem 1
Description of Problem (just a 1-2 line summary!):
Run Linear, Gaussian and Polynomial SVMs on seeds_dataset.txt

@author rhsu
The follow code uses helper.py to create dataframes and do most computing on data.
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
#from helper import *
from tools import *
from featureselection import *


plt.rcParams["figure.figsize"] = (16,12)

file_name = 'DATA.CSV'
df = pd.read_csv('DATA.csv',sep=';',index_col=0)
df.drop(['COURSE ID'],axis = 1, inplace=True)

# Classification models to use (as well as Poly SVC initated below)
nb_class = MultinomialNB()
tree_class = DecisionTreeClassifier()
k_class = KNeighborsClassifier()

# Dictionaries to hold accuracy at a feature # across models
nb_dict = {'SelectK': [], 'RFE': []}
dt_dict = {'SelectK': [], 'RFE': []}
kn_dict = {'SelectK': [], 'RFE': []}
sv_dict = {'SelectK': [], 'RFE': []}

# Get the selected featuers from SelectKBest feature selector
fit,skb = kbest_select(df,5)

# preprocess data to be pass/fail vs 6 grades
df['PassFail'] = np.where(df['GRADE'] > 2, 1, 0)
df.drop(['GRADE'],axis=1,inplace=True)

# Feature count to iterate over
for features in range(3,31,1): 
    # get features from RFE selection
    rfe,feats = rfe_select(df,features)

    # run classification models for each selection type
    for title,l in [['SelectK',skb[:features,0]],['RFE',feats]]:

        # Add the True Label to the subset of columns of selected features
        l = np.append(l,30)
        # setup as list of INT for index location in subset_columns
        l = [int(x) for x in l]
        #print(f'Features selected: {l}')
        df_subset = subset_columns(df,list(l))

        # Setup train test model
        X = df_subset.iloc[:,:-1]
        Y = df_subset.iloc[:,-1:]   
        X_sc = StandardScaler().fit_transform(X)
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.6,
                        random_state=3,stratify=Y)

        ##### Naive Bayesian Model Classification
        nb_class.fit(x_train,y_train.values.ravel())
        y_pred = nb_class.predict(x_test)
        #print(f'NaiveBayes {title}:{accuracy_score(y_pred,y_test)}')
        nb_dict[title].append([features,accuracy_score(y_pred,y_test)])

        ##### Decision Tree Model Classification
        tree_class.fit(x_train,y_train.values.ravel())
        y_pred = nb_class.predict(x_test)
        #print(f'DecisionTree {title}:{accuracy_score(y_pred,y_test)}')
        dt_dict[title].append([features,accuracy_score(y_pred,y_test)])        

        #### Polynomical SVM Classification
        # List of kernel type and internal kernel type argument for SVC to run for data
        c_val = 10 # Regularization parameter for SVC
        lin_svm = Pipeline([('scaler',StandardScaler()),
                            ('svm',SVC(kernel='rbf',C=c_val))])    
        # fit model to training values - ravel y_train to shape (#,)
        lin_svm.fit(x_train,y_train.values.ravel())
        # predict on x_test values
        y_pred = lin_svm.predict(x_test)
        #print(f'{title} kernel SVM: {acc}')
        sv_dict[title].append([features,accuracy_score(y_pred,y_test)])        

        #### K Neighbors Classification

        accuracy = {} #dictionary of results - key is the k_value, value is the accuracy
        # use kNN classifier for k = 3,5,7,9,11 and select best
        for i in range(3,12,2):
            k_class = KNeighborsClassifier(n_neighbors=i) # start with k=3, using L^2 norm
            k_class.fit(x_train,y_train.values.ravel())
            y_pred =  k_class.predict(x_test)
            accuracy[i] = accuracy_score(y_test,y_pred)

        # Get best K value with highest accuracy
        max_k_value = max(accuracy,key=accuracy.get)
        #print(f'Features: {features}, {max_k_value}')
        # Run classifier with best K value
        k_class = KNeighborsClassifier(n_neighbors=max_k_value) 
        k_class.fit(x_train,y_train.values.ravel())
        y_pred = k_class.predict(x_test)
        #print(f'KNeighbors k={max_k_value} {title}:{accuracy_score(y_pred,y_test)}')
        kn_dict[title].append([features,accuracy_score(y_pred,y_test)])        

# Create two plots, one for Select K type selection and the other for RFE
fig,(ax1,ax2) = plt.subplots(1,2)

# For each type of selection plot the four types of classification accuracies
for title,model in [['Naive',nb_dict],
              ['DecisionTree',dt_dict],
              ['KNeighbor',kn_dict],
              ['Poly SVM',sv_dict]]:

    selplot = np.array(model.get('SelectK')) # Select K accuracies in dict
    rfeplot = np.array(model.get('RFE'))    # RFE accuracies in dict

    X = selplot[:,0]    # num of features
    selY = selplot[:,1] # accuracy of select K features
    rfeY = rfeplot[:,1] # accuracy of rfe features

    ax1.plot(X,selY,label=title)
    ax2.plot(X,rfeY,label=title)

ax1.set(xlabel='Features',ylabel='Accuracy')
ax2.set(xlabel='Features',ylabel='Accuracy')
ax1.legend()
ax2.legend()
ax1.set_title('Select K Best Selection')
ax2.set_title('RFE Selection')
plt.show()

fig,[[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
# For each type of selection plot the four types of classification accuracies
for ax,title,model in [[ax1,'Naive',nb_dict],
              [ax2,'DecisionTree',dt_dict],
              [ax3,'KNeighbor',kn_dict],
              [ax4,'Poly SVM',sv_dict]]:

    selplot = np.array(model.get('SelectK')) # Select K accuracies in dict
    rfeplot = np.array(model.get('RFE'))    # RFE accuracies in dict

    X = selplot[:,0]    # num of features
    selY = selplot[:,1] # accuracy of select K features
    rfeY = rfeplot[:,1] # accuracy of rfe features

    ax.plot(X,selY,label='SelectK')
    ax.plot(X,rfeY,label='RFE')
    ax.set(xlabel='Features',ylabel='Accuracy')
    ax.legend() 
    ax.set_title(title)

plt.show()
    
# %%
