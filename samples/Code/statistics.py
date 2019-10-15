import matplotlib.pyplot as plt
import numpy as np

# Random variable X = N(1, 1)
x = np.random.normal(loc=1.0, scale=1.0, size=1000)
plt.scatter(range(len(x)), x, s=4)
plt.show()

# Expectation
print('E[X] =', x.mean(), '(python)')
print('E[X] =', x.sum() / len(x), '(formula)')
print()

# Variance
print('Var[X] = ', x.var(), '(python)')
print('Var[X] = ', np.mean(x * x) - np.power(x.mean(), 2), '(formula)')
print('Var[X] = ', np.sum(np.power(x - x.mean(), 2)) / (len(x) - 1), '(sample)')
print()

# Standard deviation
print('Std[X] = ', x.std(), '(python)')
print('Std[X] = ', np.sqrt(np.mean(x * x) - np.power(x.mean(), 2)), '(formula)')
print('Std[X] = ', np.sqrt(np.sum(np.power(x - x.mean(), 2)) / (len(x) - 1)), '(sample)')
print()

# Random variable Y = 2 * X + eps
y = 2 * x + np.random.normal(loc=0.0, scale=0.5, size=1000)
plt.scatter(x, y, s=4)
plt.show()

# Covariance
print('Cov[X, Y] = ', np.mean(x * y) - x.mean() * y.mean())
print('Corr[X, Y] = ', (np.mean(x * y) - x.mean() * y.mean()) / (x.std() * y.std()))
print()

# Linear Regression
p = np.polyfit(x, y, deg=1)
y_hat = p[0] * x + p[1]

plt.scatter(x, y, s=4)
plt.plot(x, y_hat, c='red')
plt.show()

# Coefficient of determination
r2 = np.sum(np.power(y_hat - y.mean(), 2)) / np.sum(np.power(y - y.mean(), 2))
print('Coefficient of determination', r2)
print()

# Covariance matrix
print('Covariance matrix = \n', np.cov(x, y))
