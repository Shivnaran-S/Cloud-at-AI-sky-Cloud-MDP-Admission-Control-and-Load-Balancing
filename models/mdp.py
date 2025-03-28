import numpy as np
import math
from scipy.optimize import linprog
from itertools import product
from config.config import GAMMA, K, J, I, C, D, LAMBDA, MU, R

##### STATES #####
def max_request():
    num_unknowns = K * I
    objective = -np.ones(num_unknowns)  # linprog minimizes by default, the objective is negated to maximize the number of deployments

    # Preparing constraints for s(unknowns) * d <= c
    A_ub = []
    b_ub = []

    for row in range(K):
        for j in range(J):
            constraint_row = np.zeros(num_unknowns)
            for i in range(I):
                constraint_row[row * I + i] = D[i, j]
            A_ub.append(constraint_row)
            b_ub.append(C[row, j])

    # Convert lists to numpy arrays for linprog
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds for each entry in s to be non-negative i.e., (0, None) means each variable has a lower bound of 0 and no upper bound (None)
    bounds = [(0, None) for _ in range(num_unknowns)]

    # Run linear programming optimization
    result = linprog(c=objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    # 'highs' 'highs-ds' 'highs-ipm' 'simplex' 'interior-point'

    if result.success:
        return int(result.x.max())
    else:
        # Fallback if linprog fails
        denominator = np.dot(np.ones((K, I)), D)
        denominator[denominator == 0] = np.inf  # Avoid division by zero
        fallback_value = C / denominator
        return math.ceil(np.nanmax(fallback_value))   # Use nanmax to ignore NaN values

def check_capacity(node_requests):
    return np.all(np.dot(node_requests, D) <= C)

def enumerate_states():
    states = []
    # states is the set of states that the system can reach in each point in time
    # states is a 3d-array with many state each of size (K+1 * I)
    # each state has the following architecture:
    # zero-th row is the pending request p_
    # rest rows 1 to K: s_k_i denotes the number of requests of type i hosted on node k (i.e., s_k_i constitute the unraveling of the matrix of number of requests of type i on machine k)
    
    #       { i, 1 <= i <= I, pending deployment request type 
    # p_ =  {  
    #       { 0, no pending request
    # Generate all possible configurations for the pending request (0 means no pending request)
    for p_ in range(I + 1):
        s = np.zeros((K + 1, I), dtype=int)  # Initialize the state matrix

        # Set the pending request type
        if p_ != 0:
            s[0, p_ - 1] = 1  # Set pending request type to p_

        # Generate all combinations for s[1:, :] where each entry can range from 0 to a reasonable maximum
        max_requests = max_request()
        all_combinations = product(range(max_requests + 1), repeat=K * I)

        for comb in all_combinations:
            node_requests = np.array(comb).reshape(K, I)

            # Check if this configuration satisfies the capacity constraint
            if check_capacity(node_requests):
                s[1:, :] = node_requests
                states.append(np.copy(s))

    return states


##### ACTION #####
def action(s):
    p = 0 # Initially pending request is set to zero
    k_ = []
    p__ = s[0]
    i_ = np.where(p__ != 0)[0]  # np.where(p__ != 0) returns a tuple with numpy array at index 0, the numpy array contains the indices of non-zero elements
    if i_.size>0:
        p = i_[0] + 1  # It is the pending request
        for k in range(1, K+1):  # Check each node
            ru = np.zeros(J)  # Resource utilized by each node, for each resource type j
            for i in range(I):
                ru += s[k,i]*D[i] 
            ru += D[p-1] 
            if np.all(ru<=C[k-1]):
                k_.append(k)
    return k_, p

def action_admission(s):
    k_, _ = action(s)
    if len(k_)>0:
        return 1
    return 0

def action_placement(s):
    # Extract the current capacities and demands
    capacities = C
    demands = D

    # Initialize allocations
    allocations = np.zeros((K, I), dtype=int)
    threshold = max(np.mean(demands), demands.max())

    for i in range(I):
        for k in range(K):
            # Check if current node can accommodate the current type of request
            if np.all(np.dot(s[1:, i], demands[i]) <= capacities[k] - np.sum(allocations[k])):
                allocations[k, i] += 1
                # Decrease pending request count after allocation
                if np.any(s[0]):
                    s[0, i] -= 1

    # Calculate overloaded and underloaded nodes based on the threshold
    overloaded = [idx for idx, load in enumerate(np.sum(allocations, axis=1)) if load > threshold]
    underloaded = [idx for idx, load in enumerate(np.sum(allocations, axis=1)) if load < threshold]

    # Choose the first underloaded node or an available node
    if underloaded:
        return underloaded[0] + 1  # Node indexing starts from 1 as per original logic
    elif not overloaded:
        for k in range(K):
            if np.all(np.dot(s[1:, i], demands[i]) <= capacities[k] - np.sum(allocations[k])):
                return k + 1
    return 0  # Return 0 if no suitable node is found 

def reward(s, a):
    _, p = action(s)
    if a:
        return R[p-1] * MU[p-1]
    else:
        return 0

def sum_transition_rates(s, a):
    action(s)
    sum = a * MU[p-1] if p!=0 else 0
    for i in range(I):
        sum += LAMBDA[i]
        for k in range(K):
            sum+= s[k+1,i] * MU[i]
    return sum

def transition_probability(s, a, s_):
    probability = 0
    p__ = s_[0]
    i_ = np.where(p__ != 0)[0]
    if i_.size > 0:
        p_ = i_[0]+1
    else:
        p_ = 0
    action(s)
    if a * p == 0:
        if p_!=0 and np.array_equal(s[1:,:],s_[1:,:]):
            probability = LAMBDA[p_-1] / sum_transition_rates(s, a)
        elif p_==0:
            indices = np.argwhere(s[1:,:] - 1 == s_[1:,:])
            if indices.size>0:
                probability = s[indices[0,0],indices[0,1]] * MU[indices[0,1]] / sum_transition_rates(s, a)
    elif a:
        k = action_placement(s)
        s_temp = np.copy(s)
        s_temp[k-1,p-1] += 1
        if p_!=0 and np.array_equal(s_[1:,:],s_temp[1:,:]):
            probability = LAMBDA[p_-1] / sum_transition_rates(s, a)
        elif p_==0:
            indices = np.argwhere(s[1:,:] - 1 == s_temp[1:,:])
            if indices.size>0:
                probability = s[indices[0,0],indices[0,1]] * MU[indices[0,1]] / sum_transition_rates(s, a)
    return probability

def gamma(s, a):
    return sum_transition_rates(s, a) / ( sum_transition_rates(s, a) - np.log(GAMMA) )