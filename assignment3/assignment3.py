#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:59:32 2019

@author: aaronyerke

Machine Learning with Dr. Du, Dec 2019

Homework3:

Create a neural net...
"""

# --------------------------------------------------------------------------
# Import external libraries
# --------------------------------------------------------------------------
# for our code
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
# for comparison
import scipy.stats as stats
from sklearn import linear_model
# for dataset
from sklearn.datasets import load_iris
from sklearn import preprocessing

# --------------------------------------------------------------------------
# set up plotting parameters
# --------------------------------------------------------------------------
line_width_1 = 2
line_width_2 = 2
marker_1 = '.' # point
marker_2 = 'o' # circle
marker_size = 12
line_style_1 = ':' # dotted line
line_style_2 = '-' # solid line

# --------------------------------------------------------------------------
# Some useful functions
# --------------------------------------------------------------------------
def logit(p):
    return np.log(p/(1-p))

def sigmoid(line):
    return 1/(1 + np.exp(-line))

def loss_j(y, y_hat):
    return -np.mean(y*np.log(y_hat) + (1-y) * np.log(1-y_hat))

# --------------------------------------------------------------------------
# set up log reg class
# --------------------------------------------------------------------------
class my_logistic_reg:
    def __init__(self, lr = 0.001, n_iter = 1000, dj_stop = 0.0001):
        self.slopes = None
        self.y_intercept = None
        self.lr = lr
        self.n_iter = n_iter
        self.dj_stop = dj_stop
        
    def _loss_j(self, y, y_hat):
        return -np.mean(y*np.log(y_hat) + (1-y) * np.log(1-y_hat))

    def fit_model(self, my_x, my_y):
        n_samples, n_features = my_x.shape
        # init parameters
        self.slopes = np.zeros(n_features)
        self.y_intercept = 1
        self.cost_j = []

        # gradient descent   
        while len(self.cost_j) < self.n_iter:#this is basically a for-loop that will go n iterations
            # approximate y with linear combination of slope and x, plus y_intercept
            lin_model = np.dot(my_x, self.slopes) + self.y_intercept
            # apply sigmoid function
            y_predicted = sigmoid(lin_model)
            loss = self._loss_j(my_y, y_predicted)          
            # compute gradients
            dz = y_predicted -my_y
            d_slope = (1 / n_samples) * np.matmul(my_x.T, dz)
            d_intercept = np.sum(dz)
            
            # update parameters
            self.slopes -= self.lr * d_slope
            self.y_intercept -= self.lr * d_intercept
            if len(self.cost_j) == 0:
                self.cost_j.append(loss)
            else:
                self.cost_j.append(loss)
                if abs(loss - self.cost_j[-2]) < self.dj_stop:
                    break#get out of while loop
        print("Fit completed!")
                
    def test_model(self, my_x, y_val = None):
        lin_model = np.dot(my_x, self.slopes) + self.y_intercept
        y_predicted = sigmoid(lin_model)
        if y_predicted > 0.5:
            model_prediction = 1
        else:
            model_prediction = 0
            
        if y_val != None:
            print(f"The model prediction: {model_prediction}\nThe correct value: {y_val}")
        return model_prediction
        
        
    def plot_cost(self):
        if len(self.cost_j) != 0:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter( range(0, len(self.cost_j) ), self.cost_j, label=f'Cost function values', color='yellow', marker=marker_1, linewidth=line_width_1)
#            ax.plot(self.x, self.y_hat, color='blue', label='model from training data', linewidth=line_width_1)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Cost values')
#            ax.set_title("Linear regression of diabetes data with d_lm class")
#            ax.legend(loc='lower right', fontsize=9)
            fig.show()

# --------------------------------------------------------------------------
# set up neural net class
# --------------------------------------------------------------------------
class neural_net (input_layer, hidden_layer, output_layer):
    def __init__(self, lr = 0.001, n_iter = 1000, dj_stop = 0.0001):
        self.slopes = None
        self.y_intercepts = None
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        
    def back_prop(self):
        pass
    
    def forward_prop(self):
        pass
    
    def classify_unknown(self, unknown):
        pass
    
    