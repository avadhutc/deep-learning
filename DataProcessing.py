# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:06:22 2018

@author: Nwh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("F:/Udacity ML A_Z\Machine_Learning_AZ/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

""" #for missing value preprocessing
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)              
imputer = imputer.fit(X[:,1:3])
X[:,1:3]  = imputer.transform(X[:,1:3])      

#taking care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#X1 = X
#X1 = pd.get_dummies(X1)

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""

# splitting train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                 random_state=0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""