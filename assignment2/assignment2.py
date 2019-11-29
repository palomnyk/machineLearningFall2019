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

def loss_j(X,y, coef):
    return 1/len(y) * np.sum((y @ np.log(sigmoid(X@coef)) - (1-y) @ np.log(1-sigmoid(X@coef)) ))

# --------------------------------------------------------------------------
# set up log reg class
# --------------------------------------------------------------------------
class my_logistic_reg:
    def __init__(self, lr = 0.1, n_iter = 1000, dj_stop = 0.00001):
        self.slopes = None
        self.y_intercept = None
        self.lr = lr
        self.n_iter = n_iter
        self.dj_stop = dj_stop

    def fit_model(self, my_x, my_y):
        n_samples, n_features = my_x.shape
        # init parameters
        self.slopes = np.zeros(n_features)
        self.y_intercept = 1
        #stop the grad descent ∆J = 0.00001
        self.cost_j = []

        # gradient descent   
        while len(self.cost_j) < self.n_iter:#this is basically a for-loop that will go n iterations
            # approximate y with linear combination of slope and x, plus y_intercept
            lin_model = np.dot(my_x, self.slopes) + self.y_intercept
            # apply sigmoid function
            y_predicted = sigmoid(lin_model)
            loss = loss_j(my_x, my_y, self.slopes)          
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
            ax.scatter( range(0, len(self.cost_j) ), self.cost_j, label=f'Cost function values', color='black', marker=marker_1, linewidth=line_width_1)
#            ax.plot(self.x, self.y_hat, color='blue', label='model from training data', linewidth=line_width_1)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Cost values')
#            ax.set_title("Linear regression of diabetes data with d_lm class")
#            ax.legend(loc='lower right', fontsize=9)
            fig.show()

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
#normalize data to max value
x = df_x.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_x = pd.DataFrame(x_scaled)
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
my_lr.plot_cost()
my_lr.test_model(test_x, test_y.iloc[0])

# --------------------------------------------------------------------------
# run professional log reg model
# --------------------------------------------------------------------------
print('pro_stuff')
skl_log = linear_model.LogisticRegression(solver="lbfgs")
skl_log.fit(X=train_x, y=train_y)
skl_pred = skl_log.predict(test_x)
print(f'sklearn log rediction: {skl_pred}')
print(f'Correct answer: {test_y}, {test_y == skl_pred}')
print(skl_log.coef_)
print(skl_log.intercept_)
skl_log_cost = loss_j(train_x, train_y, skl_log.coef_.T)
print(f'2(c) the total final cost J: {skl_log_cost}')

