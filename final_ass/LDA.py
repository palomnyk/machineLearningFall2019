#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:17:36 2019

LDA for Final Project in Machine Learning

2.  (30 points)PCA and LDAIn datasetdataset1.csv, columns correspond to variables and there are two variables namedV1andV2.
(1)  PlotV2vsV1.  Do you see a clear separation of the raw data?
(2)  Apply PCA to this dataset without scaling the two variables.  Project the raw data onto your first principal component axis, i.e.  the PC1 axis.  Do you still see a clear separation of the datain PC1, i.e. in projections of your raw data on the PC1 axis?
(3)  Add the PC1 axis to the plot you obtained in (1).
(4)  Apply  LDA  to  this  dataset  and  obtainW.   The  class  information  of  each  data  point  is  in  thelabelcolumn.
(5)  Project your raw data ontoW.  Do you see a clear separation of the data in the projection ontoW?
(6)  Add theWaxis to your plot.  At this point, your plot should contain the raw data points, thePC1 axis you obtain from the PCA analysis, and theWaxis you obtain from the LDA analysis.
(7)  Compute  the  variance  of  the  projections  onto  PC1  and  PC2  axes.   What  is  the  relationshipbetween these two variances and the eigenvalues of the covariance matrix you use for computingPC1 and PC2 axes?
(8)  Compute the variance of the projections onto theWaxis.
(9)  What message can you get from the above PCA and LDA analyses?

@author: aaronyerke
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data = pd.read_csv("dataset_1.csv",header=0)

#code for part2.1
plt.scatter( 'V2', 'V1', data=data, marker='p', c="label", cmap=plt.cm.autumn, linewidth=2, label="V2 vs V1", )
plt.title(label="Part2.2")
plt.legend()
plt.show()
#I can make out a line that I think would separate them quite well.

#code for part2.2
from sklearn.decomposition import PCA
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca = pca.fit_transform(data)
pc_df2 = pd.DataFrame(data=pca,columns = ['PC1','PC2'])
pc_df2['label'] = data['label']
#plt.scatter( 'V2', 'V1', data=data, marker='p', c="label", cmap=plt.cm.autumn, linewidth=2, label="V2 vs V1", )
plt.scatter( 'PC1', 'PC2', data=pc_df2, marker='p', c="label", cmap=plt.cm.flag, linewidth=2, )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(label="Part2.2\nPC1 vs PC2")
plt.show()

#code for part2.3
plt.scatter( 'PC1', 'PC2', data=pc_df2, marker='p', c="label", cmap=plt.cm.flag, linewidth=2, label="V2 vs V1", )
plt.title(label="Part2.2")
plt.legend()
plt.show()







#X_train = data.iloc[0]
#y_train = data.iloc[1]
#
#X_train = X_train.reshape(-1, 1)
#
#lda = LDA(n_components=1)
#X_train = lda.fit_transform(X_train, y_train)
