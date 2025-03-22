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