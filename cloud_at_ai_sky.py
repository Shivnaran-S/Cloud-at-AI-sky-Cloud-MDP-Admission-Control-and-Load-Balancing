import numpy as np
from itertools import product
from scipy.optimize import linprog

num_iterations = 2

# Parameters
gama = 0.9  # Discount factor

K = 2  # The data center is comprised of K physical machines called nodes
J = 3   # Number of resource types
I = 2  # Number of deployment requests type

p = 0
s = np.zeros((K+1, I)) # States or stocks in the data center. K+1 rows beacause the first row is p the pending deployment request.
                       # s[k,i] denotes the number of requests of type i hosted on machine/node k
                       # The pending deployment request type taking values in 1 ≤ i ≤ I that indicates the type of the request, or the value 0 in case there is no pending request
#c = np.zeros((K,J)) # Each node k has a specific capacity c[k,j]
#d = np.zeros((I,J)) # d[i,j] denotes the resource requirement of deployment type i from resource j
c = np.array([
    [10, 8, 6],
    [12, 10, 9]
])

# Demand matrix: Each row is the demand for one request type, each column is for a resource type
d = np.array([
    [1, 2, 1],
    [2, 1, 2]
])
#lamda = [12]*I # Arrival rate
#mu = [12]*I  # Life time rate
#r = [12]*I  # Profit rate ( profit per unit time )
lamda = np.array([5, 4])
mu = np.array([7, 6])
r = np.array([10, 15])
def max_request():
    num_unknowns = K * I
    objective = -np.ones(num_unknowns)  # Negated for maximization with linprog

    # Prepare constraints for s * d <= c
    A_ub = []
    b_ub = []

    for row in range(K):
        for j in range(J):
            constraint_row = np.zeros(num_unknowns)
            for i in range(I):
                constraint_row[row * I + i] = d[i, j]
            A_ub.append(constraint_row)
            b_ub.append(c[row, j])

    # Convert lists to numpy arrays for linprog
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds for each entry in s to be non-negative
    bounds = [(0, None) for _ in range(num_unknowns)]

    # Run linear programming optimization
    result = linprog(c=objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        return int(result.x.max())
    else:
        # Fallback if linprog fails
        denominator = np.dot(np.ones((K, I)), d)

        # Avoid division by zero by replacing zeros in denominator with a large value
        denominator[denominator == 0] = np.inf  # Avoid division by zero
        fallback_value = c / denominator
        return int(np.nanmax(fallback_value))  # Use nanmax to ignore NaN values



def check_capacity(node_requests):
    return np.all(np.dot(node_requests, d) <= c)

def enumerate_states():
    states = []

    # Generate all possible configurations for the pending request (0 means no pending request)
    for p_ in range(I + 1):
        s = np.zeros((K + 1, I), dtype=int)  # Initialize the state matrix

        # Set the pending request type
        if p_ != 0:
            s[0, p_ - 1] = 1  # Set pending request type to p_

        # Generate all combinations for s[1:, :] where each entry can range from 0 to a reasonable maximum
        max_requests = max_request()
        #max_requests = int(np.max( c / np.dot( np.full((K,I), 1) , d )  ))
        all_combinations = product(range(max_requests + 1), repeat=K * I)

        for comb in all_combinations:
            node_requests = np.array(comb).reshape(K, I)

            # Check if this configuration satisfies the capacity constraint
            if check_capacity(node_requests):
                s[1:, :] = node_requests
                states.append(np.copy(s))

    return states
# Generate all possible states that meet the capacity constraints
states = enumerate_states()

def action(s):
    k_= []
    p__ = s[0]
    i_ = np.where(p__ != 0)[0]
    if i_.size>0:
        p = i_[0] + 1 # It is the pending request
        for k in range(1, K+1):  # Check each node
            ru = np.zeros(J)  # Resource Utilised
            for i in range(I):
                ru += s[k,i]*d[i]
            ru += d[p-1]
            if np.all(ru<=c[k-1]):
                k_.append(k)
    return k_
def action_admission(s):
    k_ = action(s)
    if len(k_)>0:
        return 1
    return 0
def action_placement(s):
    # Extract the current capacities and demands
    capacities = c
    demands = d

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
    action(s)
    if a:
        return r[p-1] * mu[p-1]
    #Deployment of type i has a profit rate r[i]
    #A successful deployment is awarded with a profit r[i] times the time units it is deployed on the cloud
    #Each deployment has a life time assumed to follow exponential distribution with type-dependent mean µ[i]
    else:
        return 0

def sum_transition_rates(s, a):
    #a = action_admission(s)
    action(s)
    sum = a * mu[p-1] if p!=0 else 0
    for i in range(I):
        sum += lamda[i]
        for k in range(K):
            sum+= s[k+1,i] * mu[i]
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
            probability = lamda[p_-1] / sum_transition_rates(s, a)
        elif p_==0:
            indices = np.argwhere(s[1:,:] - 1 == s_[1:,:])
            if indices.size>0:
                probability = s[indices[0,0],indices[0,1]] * mu[indices[0,1]] / sum_transition_rates(s, a)
    elif a:
        k = action_placement(s)
        s_temp = np.copy(s)
        s_temp[k-1,p-1] += 1
        if p_!=0 and np.array_equal(s_[1:,:],s_temp[1:,:]):
            probability = lamda[p_-1] / sum_transition_rates(s, a)
        elif p_==0:
            indices = np.argwhere(s[1:,:] - 1 == s_temp[1:,:])
            if indices.size>0:
                probability = s[indices[0,0],indices[0,1]] * mu[indices[0,1]] / sum_transition_rates(s, a)
    return probability

def gamma(s, a):
    return sum_transition_rates(s, a) / ( sum_transition_rates(s, a) - np.log(gama) )

value_function_table = {tuple(map(tuple, state)): 0 for state in states}

def compute_value_function():
    for _ in range(num_iterations):
        new_value_function = {}
        for state in states:
            max_value = float('-inf')
            optimal_action = None

            for action in [0, 1]:  # 0: reject, 1: admit
                if action == 0:
                    expected_value = reward(state, action) + gamma(state, action) * np.sum([
                        transition_probability(state, action, s_next) * value_function_table[tuple(map(tuple, s_next))]
                        for s_next in states
                    ])
                else:
                    if action_admission(state):
                        '''
                        s_new = np.copy(state)
                        k = action_placement(state)
                        s_new[k, state[0].argmax()] += 1
                        s_new[0] = 0
                        '''
                        expected_value = reward(state, action) + gamma(state, action) * np.sum([
                            transition_probability(state, action, s_next) * value_function_table[tuple(map(tuple, s_next))]
                            for s_next in states
                        ])
                    else:
                        continue

                if expected_value > max_value:
                    max_value = expected_value
                    optimal_action = action

            new_value_function[tuple(map(tuple, state))] = max_value

        value_function_table.update(new_value_function)

def derive_optimal_policy():
    optimal_policy_table = {}
    for state in states:
        max_value = float('-inf')
        best_action = None

        for action in [0, 1]:
            if action == 0:
                expected_value = reward(state, action) + gamma(state, action) * np.sum([
                    transition_probability(state, action, s_next) * value_function_table[tuple(map(tuple, s_next))]
                    for s_next in states
                ])
            else:
                if action_admission(state):
                    '''
                    s_new = np.copy(state)
                    k = action_placement(state)
                    s_new[k, state[0].argmax()] += 1
                    s_new[0] = 0
                    '''
                    expected_value = reward(state, action) + gamma(state, action) * np.sum([
                        transition_probability(state, action, s_next) * value_function_table[tuple(map(tuple, s_next))]
                        for s_next in states
                    ])
                else:
                    continue

            if expected_value > max_value:
                max_value = expected_value
                best_action = action

        optimal_policy_table[tuple(map(tuple, state))] = best_action

    return optimal_policy_table

compute_value_function()

optimal_policy = derive_optimal_policy()

print("Value Function Table:")
for state, value in value_function_table.items():
    print(f"State: {state}, Value: {value}")

print("\nOptimal Policy Table:")
for state, action in optimal_policy.items():
    print(f"State: {state}, Optimal Action: {action}")
