import matplotlib.pyplot as plt
import numpy as np

vector1 = np.array([1, 0])
vector2 = np.array([2, 2])
vector3 = np.array([0, 2])

# Orthogonal matrices
# m = np.array([[1, 0], [0, 1]])
theta = np.pi / 4
m = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# m = np.array([[1, 0], [0, -1]])

print("Mapping matrix")
print(m)

# Mapping
vector1 = np.dot(m, vector1)
vector2 = np.dot(m, vector2)
vector3 = np.dot(m, vector3)
print("Vectors after mapping")
print(vector1)
print(vector2)
print(vector3)

# Plot
plt.axes().set_aspect('equal')

plt.arrow(0, 0, *vector1, ec='red', lw=2)
plt.arrow(0, 0, *vector2, ec='blue', lw=2)
plt.arrow(0, 0, *vector3, ec='green', lw=2)

plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.grid()
plt.show()
