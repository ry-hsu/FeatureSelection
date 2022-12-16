import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def column_dict():
    columns = [
        'Student Age','Sex','HS Type',
        'Scholorship','Additional Work','Reg Art or Sport',
        'Partner','Salary','Transportation','Accommodation',
        'Mother Education','Father Education','Siblings',
        'Parental Status','Mother Occupation','Father Occupation',
        'Weekly Study Hours','Reading (NonSci) Frequency','Reading (Sci) Frequency',
        'Attendance to seminars','Impact of projects','Attendance to classes',
        'Prep to midterm exams','Prep2 to midterm exams','Taking notes in class',
        'Listening in class','Disucssion improves success','Flip-classroom',
        'GPA last semester','Exepected GPA at grad','Course ID',
        'Grade'
        ]

    columns = [
        'Student Age','Sex','HS Type',
        'Scholorship','Additional Work','Reg Art or Sport',
        'Partner','Salary','Transportation','Accommodation',
        'Mother Education','Father Education','Siblings',
        'Parental Status','Mother Occupation','Father Occupation',
        'Weekly Study Hours','Reading (NonSci) Frequency','Reading (Sci) Frequency',
        'Attendance to seminars','Impact of projects','Attendance to classes',
        'Prep to midterm exams','Prep2 to midterm exams','Taking notes in class',
        'Listening in class','Disucssion improves success','Flip-classroom',
        'GPA last semester','Exepected GPA at grad','PassFail'
        ]

    keys = [*range(1,31)]

    col_dict = dict(zip(keys,columns))

    return col_dict

def subset_columns(df,columns):
    """
    Creates a pandas dataframe from a subset of columns in a dataframe
    ...
    Parameters
    ----------
        columns : list of str
            columns within a dataframe
    """
    try:
        #new_df = df[columns].copy()
        new_df = df.iloc[:,columns].copy()
        return new_df
    except:
        print('Columns titles do not exist in dataframe')
    
def subset_columns_lab(df,columns):
    """
    Creates a pandas dataframe from a subset of columns in a dataframe
    ...
    Parameters
    ----------
        columns : list of str
            columns within a dataframe
    """
    try:
        new_df = df[columns].copy()
        #new_df = df.iloc[:,columns].copy()
        return new_df
    except:
        print('Columns titles do not exist in dataframe')

di = column_dict()

