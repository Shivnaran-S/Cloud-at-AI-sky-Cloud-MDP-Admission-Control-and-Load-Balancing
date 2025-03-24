# services/admission_control.py
import numpy as np
from models.mdp import enumerate_states, action_admission, action_placement, reward, sum_transition_rates, transition_probability, gamma

class AdmissionControl:
    def __init__(self):
        self.states = enumerate_states()
        self.value_function_table = {tuple(map(tuple, state)): 0 for state in self.states}

    def compute_value_function(self, num_iterations=2):
        for _ in range(num_iterations):
            new_value_function = {}
            for state in self.states:
                max_value = float('-inf')
                optimal_action = None

                for action in [0, 1]:  # 0: reject, 1: admit
                    if action == 0:
                        expected_value = reward(state, action) + gamma(state, action) * np.sum([
                            transition_probability(state, action, s_next) * self.value_function_table[tuple(map(tuple, s_next))]
                            for s_next in self.states
                        ])
                    else:
                        if action_admission(state):
                            expected_value = reward(state, action) + gamma(state, action) * np.sum([
                                transition_probability(state, action, s_next) * self.value_function_table[tuple(map(tuple, s_next))]
                                for s_next in self.states
                            ])
                        else:
                            continue

                    if expected_value > max_value:
                        max_value = expected_value
                        optimal_action = action

                new_value_function[tuple(map(tuple, state))] = max_value

            self.value_function_table.update(new_value_function)

    def derive_optimal_policy(self):
        optimal_policy_table = {}
        for state in self.states:
            max_value = float('-inf')
            best_action = None

            for action in [0, 1]:
                if action == 0:
                    expected_value = reward(state, action) + gamma(state, action) * np.sum([
                        transition_probability(state, action, s_next) * self.value_function_table[tuple(map(tuple, s_next))]
                        for s_next in self.states
                    ])
                else:
                    if action_admission(state):
                        expected_value = reward(state, action) + gamma(state, action) * np.sum([
                            transition_probability(state, action, s_next) * self.value_function_table[tuple(map(tuple, s_next))]
                            for s_next in self.states
                        ])
                    else:
                        continue

                if expected_value > max_value:
                    max_value = expected_value
                    best_action = action

            optimal_policy_table[tuple(map(tuple, state))] = best_action

        return optimal_policy_table