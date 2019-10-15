import numpy as np

from numpy import linalg as la

a = np.array([[2, -4], [-1, -1]])
print(a)

l, v = la.eig(a)
v = v / np.abs(v[[1], :])
print('Eigenvalues:')
print(l)

print('Eigen vectors')
print(v)

