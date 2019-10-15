# Aaron Yerke, HW 1 for ML 2019
# 1. (50 points) Revise the linear regression code we went through in class by replacing the current function with a class.
# 2. (50 points) Perform linear regression on the diabetes dataset using both the linear regression class you obtain in problem 1 and sklearn.
# This dataset can be accessed via the following python code:
# from sklearn import datasets diabetes = datasets.load diabetes()
# and the linear model module in sklearn can be accessed via the following python code:
# from sklearn import linear model
# This diabetes dataset contains 10 features/variables. Select diabetes.data[:,2] as x for linear regression. The dependent variable y is diabetes.target. Split x and y into training and testing sets by randomly selecting 20 points for testing and the remaining for training. Plot the testing x vs testing y, and the testing x vs predicted y in the same plot.

import os, sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#TODO:
  #class for OLS
  #class for gradient descent

#gradient descent
class gradi_desc(object):
  def __init__(self, data_table):
        self.data_table = data_table

  # def plot_data():
  #   fig = plt.figure()
  #   ax = fig.add_subplot(1, 1, 1)
  #   ax.scatter(dataIn['size'], dataIn['price'], marker='.')
  #   ax.set_xlabel('size')
  #   ax.set_ylabel('price')
  #   fig.show()

  def normalize_data():
    # normalize variables to make them have similar scale
    return (self.data_table - self.data_table.mean()) / self.data_table.std()


in_file_name = "home_price.csv"
in_file_full_name = os.path.join(data_absolute_path, in_file_name)

dataIn = pd.read_csv(in_file_full_name)


# one variable
X = data_normalized.iloc[:, 0:1]
X = X.values

number_of_samples = X.shape[0]

X0 = np.ones((number_of_samples, 1))
my_X = np.concatenate((X0, X), axis=1)

number_of_variables = my_X.shape[1] # including X0

my_y = data_normalized.iloc[:, 2]

my_delta_J_threshold = 0.001

my_initial_theta = np.zeros((number_of_variables, 1))

my_learning_rate = 0.001

obj_MLR = d_multivariate_linear_regression.MLR(X=my_X,
                                               y=my_y,
                                               delta_J_threshold = my_delta_J_threshold,
                                               initial_theta=my_initial_theta,
                                               learning_rate=my_learning_rate)

optimal_theta, J = obj_MLR.do_gradient_descent()

y_hat = np.zeros(number_of_samples)
for i in range(number_of_samples):
    y_hat[i] = optimal_theta[0] + optimal_theta[1] * X[i]

y_hat_restored = y_hat * dataIn.iloc[:, 2].std() + dataIn.iloc[:, 2].mean()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(dataIn['size'], dataIn['price'], marker='.', color='blue')
ax.plot(dataIn.iloc[:, 0], y_hat_restored, color='red')
ax.set_xlabel('size')
ax.set_ylabel('price')
fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(range(len(J)), J, marker='.', color='blue')
ax.set_xlabel('iterations')
ax.set_ylabel('J')
fig.show()

# two variables
X = data_normalized.iloc[:, 0:2]

number_of_samples = X.shape[0]
number_of_variables = X.shape[1] # including X0

X0 = np.ones((number_of_samples, 1))
my_X = np.concatenate((X0, X), axis=1)

my_y = data_normalized.iloc[:, 2]

my_delta_J_threshold = 0.001

my_initial_theta = np.zeros((number_of_variables, 1))

my_learning_rate = 0.01

obj_MLR = d_multivariate_linear_regression.MLR(X=my_X,
                                               y=my_y,
                                               delta_J_threshold = my_delta_J_threshold,
                                               initial_theta=my_initial_theta,
                                               learning_rate=my_learning_rate)

optimal_theta, J = obj_MLR.do_gradient_descent()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(len(J)), J, color='b')
ax.plot(range(len(J)), J, marker='.')
ax.set_xlabel('iterations')
ax.set_ylabel(r'$J$')
fig.show()

xx = 1



# --------------------------------------------------------------------------
# set up paths
# --------------------------------------------------------------------------
# get the directory path of the running script
working_dir_absolute_path = os.path.dirname(os.path.realpath(__file__))

toolbox_absolute_path = os.path.join(working_dir_absolute_path, "ML_toolbox")
data_absolute_path = os.path.join(working_dir_absolute_path, "data")

sys.path.append(toolbox_absolute_path)
sys.path.append(data_absolute_path)



from ML_toolbox import d_lm
from ML_toolbox import d_multivariate_linear_regression

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
# other settings
# --------------------------------------------------------------------------
boolean_using_existing_data = False

if boolean_using_existing_data:
    in_file_name = "linear_regression_test_data.csv"
    in_file_full_name = os.path.join(data_absolute_path, in_file_name)

    dataIn = pd.read_csv(in_file_full_name)
    x = np.array(dataIn['x'])
    y = np.array(dataIn['y'])
    y_theoretical = np.array(dataIn['y_theoretical'])
else:
    n = 20
    # np.random.seed(0)

    x = -2 + 4 * np.random.rand(n)
    x = np.sort(x)

    beta_0 = 1.0
    beta_1 = 1.5
    sigma = 1.0

    epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)

    y_theoretical = beta_0 + beta_1 * x
    y = beta_0 + beta_1 * x + epsilon

# --------------------------------------------------------------------------
# linear regression using OLS
# --------------------------------------------------------------------------
n = len(x)

x_bar = np.mean(x)
y_bar = np.mean(y)

# do linear regression using my own function
lm_d_result = d_lm.d_lm(x, y)

# plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)
ax.plot(x, lm_d_result['y_hat'], color='blue', label='predicted', linewidth=line_width_1)
ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=line_width_1)
ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Linear regression")
ax.legend(loc='lower right', fontsize=9)
fig.show()

# --------------------------------------------------------------------------
# cost function
# --------------------------------------------------------------------------
all_beta_1 = np.arange(start=beta_1 - 2.0, stop=beta_1 + 2.0, step=0.01)
if beta_0 == 0:     # cost J is a function of beta_1 only
    J_vec = np.zeros(len(all_beta_1))

    for i in range(len(all_beta_1)):
        current_beta_1 = all_beta_1[i]

        for j in range(n):
            current_y_hat = current_beta_1 * x[j]

            J_vec[i] = J_vec[i] + (current_y_hat - y[j])**2

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(all_beta_1, J_vec)
    ax.set_xlabel(r'$\theta_{1}$')
    ax.set_ylabel(r'$J(\theta_1)$')
    fig.show()
    fig.savefig('cost function_1 variable.pdf', bbox_inches='tight')

else:   # cost J is a function of beta_0 and beta_1
    all_beta_0 = np.arange(start=beta_0 - 2.0, stop=beta_0 + 2.0, step=0.1)

    beta_0_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))
    beta_1_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))

    J_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))

    for i in range(len(all_beta_1)):
        current_beta_1 = all_beta_1[i]

        for j in range(len(all_beta_0)):
            current_beta_0 = all_beta_0[j]

            beta_0_matrix[i, j] = current_beta_0
            beta_1_matrix[i, j] = current_beta_1

            for k in range(n):
                current_y_hat = current_beta_0 + current_beta_1 * x[k]

                J_matrix[i, j] = J_matrix[i, j] + (current_y_hat - y[k])**2

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(beta_0_matrix, beta_1_matrix, J_matrix, cmap=cm.coolwarm)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta_0, \theta_1)$')
    fig.show()
    fig.savefig('cost function_2 variables.pdf', bbox_inches='tight')

# --------------------------------------------------------------------------
# linear regression using gradient descent
# --------------------------------------------------------------------------
