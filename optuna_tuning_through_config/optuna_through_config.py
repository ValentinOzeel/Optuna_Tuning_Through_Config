from typing import List, Dict, Callable
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from pydantic import ValidationError

from .decorators import timer_and_counter, skip_trial_on_keypress_n
from .optuna_pydantic import OptunaPydantic
from .secondary_module import colorize, get_config


class OptunaFinetuning:
    def __init__(self, objective:Callable, optuna_config_path:str, metrics_to_optimize:List[str], directions:List[str], n_trials:int, top_percent_trials:int=20):
        
        self.objective = objective
        self.optuna_config_path = optuna_config_path
        self.metrics_to_optimize = metrics_to_optimize
        self.directions = directions
        self.n_trials = n_trials
        self.top_percent_trials = top_percent_trials
        self.trials_counter = 0
        self.config = None
                
        self._validate_input({'objective':objective, 
                              'optuna_config_path':optuna_config_path, 
                              'metrics_to_optimize':metrics_to_optimize, 
                              'directions':directions, 
                              'n_trials':n_trials,
                              'top_percent_trials':top_percent_trials})



    
    def _validate_input(self, kwargs):
        try:
            OptunaPydantic(**kwargs)
        except ValidationError as e:
            print(e)
        
        self.config = get_config(path=self.optuna_config_path)
            
        if len(self.metrics_to_optimize) != len(self.directions): 
            raise ValueError(f"The length of metrics_to_optimize ({len(self.metrics_to_optimize)}) and directions ({len(self.directions)}) args should be equal.")
        
        if self.config.get('OPTUNA_PARAMS'):
            self.optuna_hyperparameters = self.config['OPTUNA_PARAMS']
        else:
            raise ValueError("Could not find the dictionnary entry named 'OPTUNA_PARAMS' in the optuna config file.")
        
        
        if self.config.get('OPTUNA_FROZEN_PARAMS'):
            self.frozen_hyperparameters = self.config['OPTUNA_FROZEN_PARAMS']
        else:
            raise Warning("Could not find the dictionnary entry named 'OPTUNA_FROZEN_PARAMS' in the optuna config file.")
        

    @timer_and_counter
    @skip_trial_on_keypress_n
    def _objective(self, trial):
        params = self._set_trial_params(trial)
        return self.objective(params)
                

    def __call__(self):
        print('\n\nOptuna optimization begins...\n\n')
        
        # Create a study and optimize the objective function
        study = optuna.create_study(directions=self.directions)
        study.optimize(self._objective, n_trials=self.n_trials)
        
        # Reset counter
        self.trials_counter = 0


        self._print_trials(study.trials, 'Results all trials: ', 'LIGHTYELLOW_EX')
        self._print_trials(study.best_trials, 'Results best trials: ', 'LIGHTRED_EX')
  
        
        best_trials_param_range_df, param_min_max_df = self._get_top_trials_param_ranges(study.trials)
        best_trials_param_distrib_plot = self._plot_param_distributions(best_trials_param_range_df)
        
        return [study.trials, study.best_trials, best_trials_param_range_df, best_trials_param_distrib_plot, param_min_max_df]
            
            
            
    def _set_trial_params(self, trial) -> Dict:
        optuna_parameters = {}
        
        for param_name, suggest_type, param_range, param_options in self.optuna_hyperparameters:
            
            if 'suggest_grid' == suggest_type:
                param_range = [str(param) for param in param_range]
                trial_value = getattr(trial, 'suggest_categorical')(param_name, param_range)
                trial_value = float(trial_value)
                               
            elif 'categorical' in suggest_type:
                trial_value = getattr(trial, suggest_type)(param_name, param_range)
            
            else:
                trial_value = getattr(trial, suggest_type)(param_name, *param_range, **param_options)
            optuna_parameters[param_name] = trial_value
            
        if self.frozen_hyperparameters:
            for key, value in self.frozen_hyperparameters.items():
                optuna_parameters[key] = value
                
        return optuna_parameters
      
    def _print_trials(self, trials, first_print:str, delimiter_color:str):
        
        print(colorize(f'\n\n{first_print}', 'RED'))

        for trial in trials:
            print(colorize('\n\n----------------------------------------------\n', delimiter_color.upper()))
            print(colorize('TRIAL NUMBER: ', 'LIGHTGREEN_EX'), trial.number)
            print(colorize('HYPERPARAMETERS: ', 'LIGHTBLUE_EX'), trial.params)
            print(colorize(f'VALUES ({self.metrics_to_optimize}): ', 'LIGHTMAGENTA_EX'), trial.values)
        print('\n\n')
          



    


    def _get_top_trials_param_ranges(self, trials):
        # Sort trials by the average of their objective values
        trials_except_skipped = [trial for trial in trials if trial.values]
        
        if not trials_except_skipped:
            raise ValueError(f'All trials have skipped.\n\n')
        
        sorted_trials = sorted(trials_except_skipped, key=lambda x: x.values, reverse=True)
        top_n = int(len(sorted_trials) * self.top_percent_trials / 100)
            

        top_trials = sorted_trials[:top_n] if top_n >= len(trials_except_skipped) else trials_except_skipped
        
        
        param_ranges = {'trial_number': []}
        for trial in top_trials:
            param_ranges['trial_number'].append(trial.number)
            for param, value in trial.params.items():
                if param not in param_ranges:
                    param_ranges[param] = []
                param_ranges[param].append(value)

        # Collecting the parameter ranges into a list of dictionaries
        data = []
        for param, values in param_ranges.items():
            data.append({
                'Parameter': param,
                'Min': min(values),
                'Max': max(values)
            })
        
        param_min_max_df = pd.DataFrame(data)

        print(colorize(f'\n\nParameter range of top {str(self.top_percent_trials)}% trials\n', 'LIGHTYELLOW_EX'), param_min_max_df)
    
        return pd.DataFrame(param_ranges), param_min_max_df


    def _plot_param_distributions(self, param_df: pd.DataFrame):
        
        # Plot using seaborn pair plot with hue set to trial_number
        pair_plot = sns.pairplot(param_df, hue='trial_number', palette='viridis', diag_kind='kde')
       
       # pair_plot.figure.suptitle("Parameter Distributions of Top 20% Trials", y=1.02)

        fig = pair_plot._figure
        fig.savefig("best_trials_parameter_distributions.png") 

        plt.ion()
        plt.draw() 
        plt.pause(30)
        plt.ioff()
        plt.close()
        
        return fig


        