# Aaron Yerke, HW 2 for ML 2019
# 1. (50 points) Implement gradient descent-based logistic regression in Python. Use
# ∆J = 0.00001 as the stopping criterion.
# 2. (50 points total distributed as below) Apply your code from question 2 to the iris virginica and virsicolor flowers. Specifically, randomly select 99 of these flowers for training your logistic model and use the remaining one flower for testing. You only need to do training once and testing once with your specific choice of the training flowers and testing flowers. That is to say, you don’t need to do the leave-one-out cross validation 100 times.
# (a) (15 points) After your training, plot the total cost J vs iterations for your 99 training flowers for four scenarios.
# (b) (20 points) Predict the flower type of your testing flower for each of the four scenarios.
# (c) (15 points) Apply sklearn.linear model.LogisticRegression to your specific choice of training flowers. With the intercept and coefficients produced by sklearn, calculate the total final cost J for your 99 flowers.

# --------------------------------------------------------------------------
# Import external libraries
# --------------------------------------------------------------------------
# for my code
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
# for comparison
import scipy.stats as stats
from sklearn import linear_model
from sklearn import datasets
# for dataset
from sklearn.datasets import load_iris

# --------------------------------------------------------------------------
# Some useful functions
# --------------------------------------------------------------------------
def logit(p):
    return np.log(p/(1-p))

def sigmoid(line):
    return 1/(1 + np.exp(-line))

def cost_j(X,y, coef, intercept):
    total_cost = 0
    for row in range(X.shape[0]):
        term = np.log(sigmoid(-(np.dot( skl_log.coef_[0].T, np.array(X.iloc[row,])))))
        total_cost += y.iloc[row,] * term + (1-y.iloc[row,] * (term))
    return - total_cost/X.shape[0]
# --------------------------------------------------------------------------
# set up log reg class
# --------------------------------------------------------------------------
class my_logistic_reg:
    def __init__(self, lr = 0.001, n_iter = 5000, dj_stop = 0.00001):
        self.slopes = None
        self.y_intercepts = None
        self.lr = lr
        self.n_iter = n_iter
        self.dj_stop = dj_stop

    def fit_model(self, my_x, my_y):
        n_samples, n_features = my_x.shape
        # init parameters
        self.slopes = np.zeros(n_features)
        self.y_intercept = 0
        prev_cost = 1
        #stop the grad descent ∆J = 0.00001
        dj = 1
        di = 1
        gd_iter = 0
        
        print(f'{self.dj_stop} and {self.dj_stop} {gd_iter} {self.n_iter}')

        # gradient descent
        while_criteria = [dj >= self.dj_stop, di >= self.dj_stop, gd_iter <= self.n_iter]
        while any(while_criteria):
            # approximate y with linear combination of slope and x, plus y_intercept
            linear_model = np.dot(my_x, self.slopes) + self.y_intercept
            # apply sigmoid function
            y_predicted = sigmoid(linear_model)

            # compute gradients
            d_slope = (1 / n_samples) * np.dot(my_x.T, (y_predicted - my_y))
            d_intercept = (1 / n_samples) * np.sum(y_predicted - my_y)
            
            # update parameters
            self.slopes -= self.lr * d_slope
#            print(self.slopes)
            self.y_intercept -= self.lr * d_intercept
            gd_iter+=1
            di = d_intercept
            print(d_slope)
#        print(f'done with while iters:{gd_iter}')
        print(dj)
        # print(self.slope)
        # print(self.y_intercept = 0)
        
    def test(self):
        pass
# --------------------------------------------------------------------------
# create training and testingdatasets
# --------------------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris['feature_names'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df = df[df.species != "setosa"]
df = df.reset_index(drop = True)
df['species'] = df['species'].map({'versicolor': 1, 'virginica': 0})
df.to_csv('testing.tsv', sep='\t')
#print(df.shape[0])

df_x = df.iloc[:,list(range(df.shape[1]-1))]
print(f'df_x.shape: {df_x.shape}')
df_y = df.iloc[:,df.shape[1]-1]
print(f'df_y.shape: {df_y.shape}')

test_index = np.random.randint(low = 0, high = df.shape[0] - 1, size = 1 )[0]
print(test_index)
train_index = filter(lambda a: a != test_index, range(df.shape[0]))
print(train_index)

train_x = df_x.drop(test_index, axis = 0)
print(f'train_x.shape: {train_x.shape}')
train_y = df_y.drop(test_index, axis = 0)
print(f'train_y.shape: {train_y.shape}')
test_x = df_x.drop(train_index, axis = 0)
print(f'test_x.shape: {test_x.shape}')
test_y = df_y.drop(filter(lambda a: a != test_index, range(df.shape[0])), axis = 0)
print(f'test_y.shape: {test_y.shape}')

# --------------------------------------------------------------------------
# main method
# --------------------------------------------------------------------------

my_lr = my_logistic_reg()
my_lr.fit_model(train_x, train_y)

 


# --------------------------------------------------------------------------
# run professional log reg model
# --------------------------------------------------------------------------
skl_log = linear_model.LogisticRegression(solver="lbfgs")
skl_log.fit(X=train_x, y=train_y)
skl_pred = skl_log.predict(test_x)
print(f'sklearn log rediction: {skl_pred}')
print(f'Correct answer: {test_y}, {test_y == skl_pred}')
print(skl_log.coef_)
print(skl_log.intercept_)
        
my_row = np.array(train_x.iloc[1,])

skl_log_cost = cost_j(train_x, train_y, skl_log.coef_, skl_log.intercept_)
print(f'2(c) the total final cost J: {skl_log_cost}')
   
