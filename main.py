from services.admission_control import AdmissionControl

def main():
    admission_control = AdmissionControl()
    admission_control.compute_value_function()
    optimal_policy = admission_control.derive_optimal_policy()

    print("Value Function Table:")
    for state, value in admission_control.value_function_table.items():
        print(f"State: {state}, Value: {value}")

    print("\nOptimal Policy Table:")
    for state, action in optimal_policy.items():
        print(f"State: {state}, Optimal Action: {action}")

if __name__ == "__main__":
    main()

# The handling of an incoming deployment request includes two decisions:
# first, it has to be decided whether to admit the request or reject it.
# second, in case it is admitted, it has to be placed on a specific node that can satisfy its resource requirements.

# There are no backlogs, so when a request is rejected it is lost.
# Obviously, a request is automatically rejected when there is no node with sufficient resources.

# Our objective is to maximize the long-run revenue coming from admitted deployments.