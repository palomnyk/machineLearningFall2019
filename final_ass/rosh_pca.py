#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:50:53 2019

@author: aaronyerke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("processed.cleveland.data", sep=',', names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13", "14"])

X = pd.read_csv('processed.cleveland.data', names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'])

#print(df)
#del cp
del df["3"]
#del restecg
del df["7"]
#del exang
del df["9"]
#del slope
del df["11"]
#del ca
del df["12"]
#del thal
del df["13"]
#del num
del df["14"]
# for col in df.columns:
#     print(col)
#print(df)
#preprocessing
from sklearn.preprocessing import StandardScaler
# Standardizing the features
x = StandardScaler().fit_transform(df)
#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
finalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
print(finalDf)

for col in range(X.shape[1]):
    if col not in [11,12]:
        plt.scatter('principal component 1', 'principal component 2', data=finalDf, c=X.iloc[:,col], label=[set(X.iloc[:,col])], cmap=plt.cm.autumn)
        plt.title(list(X)[col])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(ticks=[min(set(X.iloc[:,col])), max(set(X.iloc[:,col]))])
#        plt.legend([min(set(X.iloc[:,col])), max(set(X.iloc[:,col]))])
        plt.show()