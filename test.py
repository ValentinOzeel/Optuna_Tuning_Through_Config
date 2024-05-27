from optuna_tuning_through_config.optuna_through_config import OptunaFinetuning

import time

# Example of a user-defined objective function
def user_objective(params):
    # Access instance variables or methods if needed
    print(
        '\nfrozen_number', params['frozen_number'], '\n',
        'int_number_grid', params['int_number_grid'], '\n',
        'int_number', params['int_number'], '\n',
        'float_number', params['float_number'], '\n',
          )
    #time.sleep(5)
    # Example: optimizing a simple quadratic function
    return ((params['frozen_number'] + params['int_number'] + params['int_number_grid']) / params['float_number']) ** 2

    

# Example usage
optuna_finetuning = OptunaFinetuning(
    objective=user_objective,
    optuna_config_path='conf\optuna_config.yml',
    metrics_to_optimize=['x_quadratic_func'],
    directions=['maximize'],
    n_trials=1000,
    top_percent_trials=30
)

# Run the tuning by calling the instance
study_trials, best_trials, best_trials_param_range_df, best_trials_param_distrib_plot, param_min_max_df = optuna_finetuning()
