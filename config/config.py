import numpy as np

# Parameters
GAMMA = 0.9  # Discount factor
K = 2  # Number of physical machines (nodes)
J = 3  # Number of resource types (Disk Space, CPU and Memory)
I = 2  # Number of deployment request types

# Node capacities
# rows are nodes and columns are resources
C = np.array([
    [10, 8, 6],
    [12, 10, 9]
])

# Demand matrix: Each row is the demand for one request type, each column is for a resource type
D = np.array([
    [1, 2, 1],
    [2, 1, 2]
])

# Arrival rates, lifetime rates, and profit rates
LAMBDA = np.array([5, 4])
MU = np.array([7, 6])
R = np.array([10, 15])