#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:08:29 2019

@author: aaronyerke
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#1 PCA and linear regression
############################
##preprocessing
############################
dataset=pd.read_csv("linear_regression_test_data.csv")
x = dataset.drop("y_theoretical", axis=1)
x = x.drop("Unnamed: 0", axis=1)
#ytheoretical
y = dataset["y_theoretical"]
#normalize with sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# ##y = sc.fit_transform(y)
############################
#PCA
# SO I THINK PCA() JUST USES DEFAULT 1 PRINCIPLE COMPONENT
############################
from sklearn.decomposition import PCA
from sklearn import decomposition
pca = decomposition.PCA(n_components=1)
print(x)
pc1 = pca.fit_transform(x)
print("-----------")
#plot = dataset.plot.scatter(x='x', y='y', c='DarkBlue')
plt.scatter( 'x', 'y', data=dataset, marker='p', linewidth=2, label="x vs y")
plt.scatter( 'x', 'y_theoretical', data=dataset, marker='p', linewidth=2, linestyle='dashed', label="x vs  y theoretical")
plt.scatter(dataset["x"], pc1, color='skyblue', label="x vs PC1")
plt.scatter(pc1[0], pc1[1], color='red', label="pc1 vs y")
plt.legend()
plt.show()