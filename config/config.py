import numpy as np

# Parameters
GAMMA = 0.9  # Discount factor
K = 2  # Number of physical machines (nodes)
J = 3  # Number of resource types (Disk Space, CPU and Memory)
I = 2  # Number of deployment request types which vary in different aspects

# Node capacities
# rows are nodes and columns are resources
# c_k_j is the capacity of node k for resource j. node k capacity is c_k_j units for resource j
C = np.array([
    [10, 8, 6],
    [12, 10, 9]
])

# Demand matrix: Each row denotes one request type and each column denotes a resource type
# d_i_j is the demand by request type i for the resource type j. request i demands d_i_j units of resource j
# Each deployment consumes a certain amount of resources during the time it is hosted
D = np.array([
    [1, 2, 1],
    [2, 1, 2]
])

# Arrival rates, lifetime rates, and profit rates
LAMBDA = np.array([5, 4]) # Modeled the arrivals of each type i = 1, 2, ...., I as a Poisson Process with rate lambda_i
MU = np.array([7, 6]) # Each deployment has a life time assumed to follow exponential distribution with type-dependent mu_i.
R = np.array([10, 15]) #  Each deployment of type i has a profit rate r_i, such that a successful deployment is awarded with a profit r_i times the time units it is deployed on the cloud.

# It is assumed that the life time of a deployment is known at the time it is submitted to the data center.
# The reward rate for each customer is determined by contract, and is typically correlated with the resource requirements.