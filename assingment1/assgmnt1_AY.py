# Aaron Yerke, HW 1 for ML 2019
# 1. (50 points) Revise the linear regression code we went through in class by replacing the current function with a class.
# 2. (50 points) Perform linear regression on the diabetes dataset using both the linear regression class you obtain in problem 1 and sklearn.
# This dataset can be accessed via the following python code:
# from sklearn import datasets diabetes = datasets.load diabetes()
# and the linear model module in sklearn can be accessed via the following python code:
# from sklearn import linear model
# This diabetes dataset contains 10 features/variables. Select diabetes.data[:,2] as x for linear regression. The dependent variable y is diabetes.target. Split x and y into training and testing sets by randomly selecting 20 points for testing and the remaining for training. Plot the testing x vs testing y, and the testing x vs predicted y in the same plot.

import numpy as np
import scipy.stats as stats
from sklearn import linear_model
from sklearn import datasets
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class d_lm:
    def __init__(self, x, y, confidence = 0.95):
        self.confidence = confidence
        self.n = len(x)
        self.x_bar = np.mean(x)
        self.y_bar = np.mean(y)
        self.S_yx = np.sum((y - self.y_bar) * (x - self.x_bar))
        self.S_xx = np.sum((x - self.x_bar)**2)
        # ====== estimate beta_0 and beta_1 ======
        self.beta_1_hat = self.S_yx / self.S_xx # also equal to (np.cov(x, y))[0, 1] / np.var(x)
        self.beta_0_hat = self.y_bar - self.beta_1_hat * self.x_bar
        # ====== estimate sigma ======
        # residual
        self.y_hat = self.beta_0_hat + self.beta_1_hat * x
        self.r = y - self.y_hat
        self.sigma_hat = np.sqrt(sum(self.r**2) / (self.n-2))
        # ====== estimate sum of squares ======
        # total sum of squares
        self.SS_total = np.sum((y - self.y_bar)**2)
        # regression sum of squares
        self.SS_reg = np.sum((self.y_hat - self.y_bar)**2)
        # residual sum of squares
        self.SS_err = np.sum((y - self.y_hat)**2)
        # ====== estimate R2: coefficient of determination ======
        self.R2 = self.SS_reg / self.SS_total
        # ====== R2 = correlation_coefficient**2 ======
        self.correlation_coefficient = np.corrcoef(x, y)
        self.delta = self.correlation_coefficient[0, 1]**2 - self.R2
        # ====== estimate MS ======
        # sample variance
        self.MS_total = self.SS_total / (self.n-1)
        self.MS_reg = self.SS_reg / 1.0
        self.MS_err = self.SS_err / (self.n-2)
        # ====== estimate F statistic ======
        self.F = self.MS_reg / self.MS_err
        self.F_test_p_value = 1 - stats.f._cdf(self.F, dfn=1, dfd=self.n-2)
        # ====== beta_1_hat statistic ======
        self.beta_1_hat_var = self.sigma_hat**2 / ((self.n-1) * np.var(x))
        self.beta_1_hat_sd = np.sqrt(self.beta_1_hat_var)
        # confidence interval
        self.z = stats.t.ppf(q=0.025, df=self.n-2)
        self.beta_1_hat_CI_lower_bound = self.beta_1_hat - self.z * self.beta_1_hat_sd
        self.beta_1_hat_CI_upper_bound = self.beta_1_hat + self.z * self.beta_1_hat_sd
        # hypothesis tests for beta_1_hat
        # H0: beta_1 = 0
        # H1: beta_1 != 0
        self.beta_1_hat_t_statistic = self.beta_1_hat / self.beta_1_hat_sd
        self.beta_1_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(self.beta_1_hat_t_statistic), df=self.n-2))
        # ====== beta_0_hat statistic ======
        self.beta_0_hat_var = self.beta_1_hat_var * np.sum(x**2) / self.n
        self.beta_0_hat_sd = np.sqrt(self.beta_0_hat_var)
        # confidence interval
        self.beta_0_hat_CI_lower_bound = self.beta_0_hat - self.z * self.beta_0_hat_sd
        self.beta_1_hat_CI_upper_bound = self.beta_0_hat + self.z * self.beta_0_hat_sd
        self.beta_0_hat_t_statistic = self.beta_0_hat / self.beta_0_hat_sd
        self.beta_0_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(self.beta_0_hat_t_statistic), df=self.n-2))
        # confidence interval for the regression line
        self.sigma_i = 1.0/self.n * (1 + ((x - self.x_bar) / np.std(x))**2)
        self.y_hat_sd = self.sigma_hat * self.sigma_i
        self.y_hat_CI_lower_bound = self.y_hat - self.z * self.y_hat_sd
        self.y_hat_CI_upper_bound = self.y_hat + self.z * self.y_hat_sd

    def predict_y_from_model(self, x_vals):
        y_vals = self.beta_0_hat + self.beta_1_hat * x_vals
        return y_vals

    def test_model(self, test_x, test_y):
        pass

    def plot_model_w_data(self):
        pass

diabetes = datasets.load_diabetes()
x = diabetes.data[:,2]
y = diabetes.target
n_samples = len(diabetes.target)
random_samples = np.random.randint(low = 0, high = n_samples, size = 20)
testing_x = np.asarray(x[random_samples])
training_x = np.asarray([x[i] for i in range(0, n_samples) if i not in random_samples])
testing_y = np.asarray(y[random_samples])
training_y = np.asarray([y[i] for i in range(0, n_samples) if i not in random_samples])
print(testing_y[0])
print(training_y[0])
print(y[0])
print(len(testing_y))
print(len(training_y))
print(len(testing_x))
print(len(training_x))

print(training_y[1:10])
print(training_x[1:10])
my_lm = d_lm(training_x, training_y)

print(my_lm.predict_y_from_model(testing_x))



from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm



from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#TODO:
  #class for OLS
  #class for gradient descent
  #class for d_lm.p

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

  def normalize_data(self):
    # normalize variables to make them have similar scale
    return (self.data_table - self.data_table.mean()) / self.data_table.std()


# in_file_name = "home_price.csv"
# in_file_full_name = os.path.join(data_absolute_path, in_file_name)

# dataIn = pd.read_csv(in_file_full_name)


# # one variable
# X = data_normalized.iloc[:, 0:1]
# X = X.values

# number_of_samples = X.shape[0]

# X0 = np.ones((number_of_samples, 1))
# my_X = np.concatenate((X0, X), axis=1)

# number_of_variables = my_X.shape[1] # including X0

# my_y = data_normalized.iloc[:, 2]

# my_delta_J_threshold = 0.001

# my_initial_theta = np.zeros((number_of_variables, 1))

# my_learning_rate = 0.001

# obj_MLR = d_multivariate_linear_regression.MLR(X=my_X,
#                                                y=my_y,
#                                                delta_J_threshold = my_delta_J_threshold,
#                                                initial_theta=my_initial_theta,
#                                                learning_rate=my_learning_rate)

# optimal_theta, J = obj_MLR.do_gradient_descent()

# y_hat = np.zeros(number_of_samples)
# for i in range(number_of_samples):
#     y_hat[i] = optimal_theta[0] + optimal_theta[1] * X[i]

# y_hat_restored = y_hat * dataIn.iloc[:, 2].std() + dataIn.iloc[:, 2].mean()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(dataIn['size'], dataIn['price'], marker='.', color='blue')
# ax.plot(dataIn.iloc[:, 0], y_hat_restored, color='red')
# ax.set_xlabel('size')
# ax.set_ylabel('price')
# fig.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(range(len(J)), J, marker='.', color='blue')
# ax.set_xlabel('iterations')
# ax.set_ylabel('J')
# fig.show()

# # two variables
# X = data_normalized.iloc[:, 0:2]

# number_of_samples = X.shape[0]
# number_of_variables = X.shape[1] # including X0

# X0 = np.ones((number_of_samples, 1))
# my_X = np.concatenate((X0, X), axis=1)

# my_y = data_normalized.iloc[:, 2]

# my_delta_J_threshold = 0.001

# my_initial_theta = np.zeros((number_of_variables, 1))

# my_learning_rate = 0.01

# obj_MLR = d_multivariate_linear_regression.MLR(X=my_X,
#                                                y=my_y,
#                                                delta_J_threshold = my_delta_J_threshold,
#                                                initial_theta=my_initial_theta,
#                                                learning_rate=my_learning_rate)

# optimal_theta, J = obj_MLR.do_gradient_descent()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(range(len(J)), J, color='b')
# ax.plot(range(len(J)), J, marker='.')
# ax.set_xlabel('iterations')
# ax.set_ylabel(r'$J$')
# fig.show()

# xx = 1



# # --------------------------------------------------------------------------
# # set up paths
# # --------------------------------------------------------------------------
# # get the directory path of the running script
# working_dir_absolute_path = os.path.dirname(os.path.realpath(__file__))

# toolbox_absolute_path = os.path.join(working_dir_absolute_path, "ML_toolbox")
# data_absolute_path = os.path.join(working_dir_absolute_path, "data")

# sys.path.append(toolbox_absolute_path)
# sys.path.append(data_absolute_path)



# from ML_toolbox import d_lm
# from ML_toolbox import d_multivariate_linear_regression

# # --------------------------------------------------------------------------
# # set up plotting parameters
# # --------------------------------------------------------------------------
# line_width_1 = 2
# line_width_2 = 2
# marker_1 = '.' # point
# marker_2 = 'o' # circle
# marker_size = 12
# line_style_1 = ':' # dotted line
# line_style_2 = '-' # solid line

# # --------------------------------------------------------------------------
# # other settings
# # --------------------------------------------------------------------------
# boolean_using_existing_data = False

# if boolean_using_existing_data:
#     in_file_name = "linear_regression_test_data.csv"
#     in_file_full_name = os.path.join(data_absolute_path, in_file_name)

#     dataIn = pd.read_csv(in_file_full_name)
#     x = np.array(dataIn['x'])
#     y = np.array(dataIn['y'])
#     y_theoretical = np.array(dataIn['y_theoretical'])
# else:
#     n = 20
#     # np.random.seed(0)

#     x = -2 + 4 * np.random.rand(n)
#     x = np.sort(x)

#     beta_0 = 1.0
#     beta_1 = 1.5
#     sigma = 1.0

#     epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)

#     y_theoretical = beta_0 + beta_1 * x
#     y = beta_0 + beta_1 * x + epsilon

# # --------------------------------------------------------------------------
# # linear regression using OLS
# # --------------------------------------------------------------------------
# n = len(x)

# x_bar = np.mean(x)
# y_bar = np.mean(y)

# # do linear regression using my own function
# lm_d_result = d_lm.d_lm(x, y)

# # plot
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
# ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)
# ax.plot(x, lm_d_result['y_hat'], color='blue', label='predicted', linewidth=line_width_1)
# ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=line_width_1)
# ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title("Linear regression")
# ax.legend(loc='lower right', fontsize=9)
# fig.show()

# # --------------------------------------------------------------------------
# # cost function
# # --------------------------------------------------------------------------
# all_beta_1 = np.arange(start=beta_1 - 2.0, stop=beta_1 + 2.0, step=0.01)
# if beta_0 == 0:     # cost J is a function of beta_1 only
#     J_vec = np.zeros(len(all_beta_1))

#     for i in range(len(all_beta_1)):
#         current_beta_1 = all_beta_1[i]

#         for j in range(n):
#             current_y_hat = current_beta_1 * x[j]

#             J_vec[i] = J_vec[i] + (current_y_hat - y[j])**2

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(all_beta_1, J_vec)
#     ax.set_xlabel(r'$\theta_{1}$')
#     ax.set_ylabel(r'$J(\theta_1)$')
#     fig.show()
#     fig.savefig('cost function_1 variable.pdf', bbox_inches='tight')

# else:   # cost J is a function of beta_0 and beta_1
#     all_beta_0 = np.arange(start=beta_0 - 2.0, stop=beta_0 + 2.0, step=0.1)

#     beta_0_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))
#     beta_1_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))

#     J_matrix = np.zeros((len(all_beta_1), len(all_beta_0)))

#     for i in range(len(all_beta_1)):
#         current_beta_1 = all_beta_1[i]

#         for j in range(len(all_beta_0)):
#             current_beta_0 = all_beta_0[j]

#             beta_0_matrix[i, j] = current_beta_0
#             beta_1_matrix[i, j] = current_beta_1

#             for k in range(n):
#                 current_y_hat = current_beta_0 + current_beta_1 * x[k]

#                 J_matrix[i, j] = J_matrix[i, j] + (current_y_hat - y[k])**2

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.plot_surface(beta_0_matrix, beta_1_matrix, J_matrix, cmap=cm.coolwarm)
#     ax.set_xlabel(r'$\theta_0$')
#     ax.set_ylabel(r'$\theta_1$')
#     ax.set_zlabel(r'$J(\theta_0, \theta_1)$')
#     fig.show()
#     fig.savefig('cost function_2 variables.pdf', bbox_inches='tight')

# --------------------------------------------------------------------------
# linear regression using gradient descent
# --------------------------------------------------------------------------
