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
# for comparison
import scipy.stats as stats
from sklearn import linear_model
from sklearn import datasets
# for dataset
from sklearn.datasets import load_iris

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


def logit(p):
    return np.log(p/(1-p))

def sigmoid(line):
    return 1/(1 + np.exp(-line))

print(logit(0.5))
print(sigmoid(0.000001))

# --------------------------------------------------------------------------
# set up log reg class
# --------------------------------------------------------------------------
class my_logistic_reg:
    def __init__(self, my_data, lr = 0.01, n_iter = 1000, dj_stop = 0.00001):
        self.my_data = my_data
        self.lr = lr
        self.n_iter = n_iter
        self.dj_stop = dj_stop

    def log_model(self):
        n_samples, n_features = self.my_data.shape

        # init parameters
        self.slope = np.zeros(n_features)
        self.y_intercept = 0

        #stop the grad descent ∆J = 0.00001
        dj = 1
        gd_iter = 0

        # gradient descent
        while dj >= self.dj_stop or gd_iter >= self.n_iter:
            # approximate y with linear combination of slope and x, plus y_intercept
            linear_model = np.dot(my_data, self.slope) + self.y_intercept
            # apply sigmoid function
            y_predicted = sigmoid(linear_model)

            # compute gradients
            d_slope = (1 / n_samples) * np.dot(data.T, (y_predicted - y))
            d_intercept = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.slope -= self.lr * d_slope
            self.y_intercept -= self.lr * d_intercept
            gd+=1
            dj = d_slope
        print('done with while')
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



# training_data = df.drop(testing_index, axis = 0)
# print(f'training_data.shape: {training_data.shape}')
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




# --------------------------------------------------------------------------
# run professional log reg model
# --------------------------------------------------------------------------
skl_log = linear_model.LogisticRegression(solver="lbfgs")
skl_log.fit(X=df_x, y=df_y)
# sk_log.predict(testing_x)


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(training_x, training_y, label=f'training data', color='yellow', marker=marker_1, linewidth=line_width_1)
# ax.scatter(testing_x, testing_y, label=f'true vals of testing data', color='red', marker=marker_1, linewidth=line_width_1)
# ax.scatter(testing_x, skl_lm.predict(testing_x), color='green', label='predicted vals of testing data', linewidth=line_width_1)
# ax.plot(training_x, skl_lm.predict(training_x), color='blue', label='model from training data', linewidth=line_width_1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title("Linear regression of diabetes data with sklearn lm")
# ax.legend(loc='lower right', fontsize=9)
# fig.show()
