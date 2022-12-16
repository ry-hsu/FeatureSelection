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
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from helper import *
from tools import *


def kbest_select(df,k_val):
    """
    Runs SelectKBest with chi2 score function on a dataframe and returns
    a list of results in ascending order of fit scores
    ...
    Parameters
    ----------
        df : pandas Dataframe
            dataframe to select features from with the last column being labels
        k_val : int
            numver of features to select   
            *** NOT NECESSARY BECAUSE WE WANT ALL VALUES***         
    Returns
        fit : SelectKBest.fit
            SelectKBest.fit object 
        results : list
            in form [[label name,score]...] in ascending order
    """    
    # Setup X and Y as features and labels respectively
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1:]

    # Scale features
    X_sc = StandardScaler().fit_transform(X)

    # Setup SelectKBest to use chi2 and the # of features to select
    test = SelectKBest(score_func=chi2, k = k_val)
    fit = test.fit(X,Y) # fit data

    # Print out scores
    np.set_printoptions(precision=3)    # set precision to 3 decimals for readibility
    #print(f'\nScores from Kbest select')
    #print(fit.scores_)

    # # Uncomment to get the k_val features chosen
    # features = fit.transform(X)
    # # Summarize selected features
    # print(features[0:5,:])

    # Map scores to their label and sort in ascending order
    scores = fit.scores_
    # add labels keys in DATA.csv are numeric 1-32
    #key_Scores = np.array([[i,x] for i,x in enumerate(scores,1)])
    # get Dataframe index instead of labels
    key_Scores = np.array([[i,x] for i,x in enumerate(scores)])
    # reverse sort
    key_Scores = key_Scores[key_Scores[:,1].argsort()[::-1]]
    #reset label to a str for dataframe label
    result = [[str(int(i)),x] for i,x in key_Scores]
    #result = [[int(i),x] for i,x in key_Scores]

    print(f"Best Features (trimmed to 5 items for display): {result[:5]}...")
    return fit,np.array(result)
    #return fit,np.array(result)


def rfe_select(df,num_feat):
    """
    Runs RFE on a dataframe and returns best selected features
    ...
    Parameters
    ----------
        df : pandas Dataframe
            dataframe to select features from with the last column being labels
        num_feat : int
            number of features to select            
    Returns
        log_fit : RFE.fit 
            RFE.fit object 
        best_ind_offset : list
            Labels of best selected features
    """     
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1:]

    X_sc = StandardScaler().fit_transform(X)
    log_reg = LogisticRegression()
    rfe = RFE(log_reg, n_features_to_select=num_feat)
    log_fit = rfe.fit(X_sc,Y.values.ravel())

    # print(f'\nRFE Selection')
    # print("Num Features: %s" % (log_fit.n_features_))
    # print("Selected Features: %s" % (log_fit.support_))
    # print("Feature Ranking: %s" % (log_fit.ranking_))

    best_ind = np.where(log_fit.ranking_ == 1)
    best_ind_offset = best_ind[0] + 1    
    c_dict = column_dict()
    best_feat_names = list(map(lambda x: c_dict.get(x),best_ind_offset))
    
    # print(f'Best Features: {best_feat_names}')
    # print(f'Indicies: {best_ind[0]}')
    return log_fit,best_ind[0]
    #return log_fit,best_ind_offset

def print_corr(df,label,limit):
    """
    Prints correlation plots based on a label of DATA.csv by restricting
    multi labels to a binomial label pass or fail
    ...
    Parameters
    ----------
        df : pandas Dataframe
            dataframe to select features from with the last column being labels
        label : list
            what column to compare to limit for labels
        limit : int
            limit to sepparate labels > and < limit value to GOOD and BAD labels
    """     
    # See correlation matrix as a heatmap
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')    
    # See if there is any more correlation based on splitting the label into a binomial
    fig = plt.figure(figsize = (20, 25))
    j = 0
    for i in df.columns:
        plt.subplot(8, 5, j+1)
        j += 1
        sns.histplot(df[i][df[label] > limit], color='g', label = 'Pass')
        sns.histplot(df[i][df[label] < limit], color='r', label = 'Fail')
        plt.legend(loc='best')
    fig.suptitle('Performance Data Analysis')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()

def print_heat_pair(df,type):
    """
    Prints either a heatmap or pairwise plot or both
    ...
    Parameters
    ----------
        df : pandas Dataframe
            dataframe to select features from with the last column being labels
        type : list
            what to print ['heat','pair'] options
    """         
    if 'heat' in type:
        # See correlation matrix as a heatmap
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn') 
    if 'pair' in type:
        sns.pairplot(df,hue='GRADE')        

########################################################################
# Below is for testing
# plt.rcParams["figure.figsize"] = (16,12)
# file_name = 'DATA.CSV'
# df = pd.read_csv('DATA.csv',sep=';',index_col=0)
# df.drop(['COURSE ID'],axis = 1, inplace=True)

# log_fit,feats = rfe_select(df,5)
# fit,key_Scores = kbest_select(df,5)


# c_dict = column_dict()
# best_feat_names = list(map(lambda x: c_dict.get(x),feats+1))

# print(f'Best Features: {best_feat_names}')
#print_heat_pair(df,['heat'])
    
# corr_feat = list(map(lambda x: c_dict.get(x),[2,11,12,29,30]))

# print(corr_feat)
# print([str(x) for x in [2,11,12,29,30]])
########################################################################


# %%
