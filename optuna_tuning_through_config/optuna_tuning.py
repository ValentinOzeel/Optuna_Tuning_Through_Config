from typing import List, Dict, Callable

import optuna
from pydantic import ValidationError

from .decorators import trials_counter, timer
from .optuna_pydantic import OptunaPydantic
from .secondary_module import colorize, get_config


class OptunaFinetuning:
    def __init__(self, objective:Callable, optuna_config_path:str, metrics_to_optimize:List[str], directions:List[str], n_trials:int):
        
        self.objective = objective
        self.optuna_config_path = optuna_config_path
        self.metrics_to_optimize = metrics_to_optimize
        self.directions = directions
        self.n_trials = n_trials
        self.trials_counter = 0
        self.config = None
                
        self._validate_input({'objective':objective, 
                              'optuna_config_path':optuna_config_path, 
                              'metrics_to_optimize':metrics_to_optimize, 
                              'directions':directions, 
                              'n_trials':n_trials})



    
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
        
        
    @trials_counter
    @timer
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
  
        
        return study.trials, study.best_trials
            
            
            
    def _set_trial_params(self, trial) -> Dict:
        optuna_parameters = {}
        
        for param_name, suggest_type, param_range, param_options in self.optuna_hyperparameters:
            
            if 'suggest_grid' == suggest_type:
                param_range = [str(param) for param in param_range]
                trial_value = getattr(trial, 'suggest_categorical')(param_name, param_range)
                trial_value = float(trial_value)
                               
            elif 'categorical' in suggest_type:
                print('BBBB', suggest_type)
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
          



    

        
        
        
        
        
        
        
        
