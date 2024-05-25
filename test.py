from src.optuna_tuning import OptunaFinetuning

# Example of a user-defined objective function
def user_objective(params):
    # Access instance variables or methods if needed
    print(
        '\nfrozen_number', params['frozen_number'], '\n',
        'int_number_grid', params['int_number_grid'], '\n',
        'int_number', params['int_number'], '\n',
        'float_number', params['float_number'], '\n',
          )
    # Example: optimizing a simple quadratic function
    return (params['int_number'] - 2) ** 2

# Example usage
optuna_finetuning = OptunaFinetuning(
    objective=user_objective,
    optuna_config_path='conf\optuna_config.yml',
    metrics_to_optimize=['x_quadratic_func'],
    directions=['maximize'],
    n_trials=10
)

# Run the tuning by calling the instance
study_trials, best_trials = optuna_finetuning()

