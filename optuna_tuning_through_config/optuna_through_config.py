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
    """
    Class to perform hyperparameter tuning using Optuna based on configuration settings.

    Attributes:
        objective (Callable): The objective function to be optimized.
        optuna_config_path (str): Path to the Optuna configuration file.
        metrics_to_optimize (List[str]): List of metric names to optimize.
        directions (List[str]): List of optimization directions (e.g., "minimize" or "maximize").
        n_trials (int): Number of trials for the Optuna study.
        top_percent_trials (int): Top percentage of trials to consider for further analysis.
        verbose (int): Manage the print statements.
    """
    
    def __init__(self, objective:Callable, optuna_config_path:str, metrics_to_optimize:List[str], directions:List[str], n_trials:int, 
                 top_percent_trials:int=20, verbose=2
                 ):
     
        self.objective = objective
        self.optuna_config_path = optuna_config_path
        self.metrics_to_optimize = metrics_to_optimize
        self.directions = directions
        self.n_trials = n_trials
        self.top_percent_trials = top_percent_trials
        self.verbose = verbose
        self.trials_counter = 0
        self.config = None
                
        self._validate_input({'objective':objective, 
                              'optuna_config_path':optuna_config_path, 
                              'metrics_to_optimize':metrics_to_optimize, 
                              'directions':directions, 
                              'n_trials':n_trials,
                              'top_percent_trials':top_percent_trials})

    
    def _validate_input(self, kwargs):
        """
        Validates the input parameters using the OptunaPydantic model and loads configuration.
        Args:
            kwargs (dict): A dictionary of the input parameters to validate.
        """
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
        """
        This method wraps the custom objective function to be optimized by Optuna that has been passed by the user.
        This method is wrapped with decorators enabling timing, counting and skip trial with crtl+c keypress.
        The trial parameters selected for the trial (based on configuration) are passed to the custom objective function.
        Args:
            trial (optuna.trial.Trial): The trial object provided by Optuna.
        Returns:
            Any: The result of the objective function.
        """
        params = self._set_trial_params(trial)
        return self.objective(params)
                

    def __call__(self):
        """
        Runs the Optuna optimization process by calling the class instance.
        Returns:
            List: Contains the study trials, best trials, DataFrame of top x% best trials, parameter distribution plot and DataFrame of best trial parameter ranges.
        """
        
        print('\n\nOptuna optimization begins...\n\n')
        
        # Create a study and optimize the objective function
        study = optuna.create_study(directions=self.directions)
        study.optimize(self._objective, n_trials=self.n_trials)
        
        # Reset counter
        self.trials_counter = 0

        if self.verbose >= 2:
            self._print_trials(study.trials, 'Results all trials: ', 'LIGHTYELLOW_EX')
        if self.verbose >= 1:
            self._print_trials(study.best_trials, 'Results best trials: ', 'LIGHTRED_EX')
  
        
        best_trials_param_range_df, param_min_max_df = self._get_top_trials_param_ranges(study.trials)
        best_trials_param_distrib_plot = self._plot_param_distributions(best_trials_param_range_df)
        
        return [study.trials, study.best_trials, best_trials_param_range_df, best_trials_param_distrib_plot, param_min_max_df]
            
            
            
    def _set_trial_params(self, trial) -> Dict:
        """
        Sets the parameters for a trial based on the configuration.
        Args:
            trial (optuna.trial.Trial): The trial object provided by Optuna.
        Returns:
            Dict: A dictionary of trial parameters.
        """
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
        """
        Prints the details of trials.
        Args:
            trials (List[optuna.trial.Trial]): The list of trials to print.
            first_print (str): The initial message to print before trial details.
            delimiter_color (str): The color to use for the delimiter.
        """
        print(colorize(f'\n\n{first_print}', 'RED'))

        for trial in trials:
            print(colorize('\n\n----------------------------------------------\n', delimiter_color.upper()))
            print(colorize('TRIAL NUMBER: ', 'LIGHTGREEN_EX'), trial.number)
            print(colorize('HYPERPARAMETERS: ', 'LIGHTBLUE_EX'), trial.params)
            print(colorize(f'VALUES ({self.metrics_to_optimize}): ', 'LIGHTMAGENTA_EX'), trial.values)
        print('\n\n')
          

    def _get_top_trials_param_ranges(self, trials):
        """
        Gets the parameter ranges of the top trials.
        Args:
            trials (List[optuna.trial.Trial]): The list of trials to analyze.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of the top trials' parameter ranges and min-max parameter ranges.
        """
        # Sort trials by the average of their objective values
        trials_except_skipped = [trial for trial in trials if trial.values]
        
        if not trials_except_skipped:
            raise ValueError(f'All trials have skipped.\n\n')
        
        sorted_trials = sorted(trials_except_skipped, key=lambda x: x.values, reverse=True)
        top_n = int(len(sorted_trials) * self.top_percent_trials / 100)
        top_trials = sorted_trials[:top_n] if top_n <= len(sorted_trials) else sorted_trials
            
        param_ranges = {'trial_number':[], 'best_trials': []}
        for i, trial in enumerate(top_trials):
            param_ranges['trial_number'].append(trial.number)
            param_ranges['best_trials'].append(i+1)
            for param, value in trial.params.items():
                if param not in param_ranges:
                    param_ranges[param] = []
                param_ranges[param].append(value)

        best_trial_df = pd.DataFrame(param_ranges)
        
        
        # Collecting the parameter ranges into a list of dictionaries
        data = []
        for param, values in param_ranges.items():
            if param != 'trial_number':
                data.append({
                    'Parameter': param,
                    'Min': min(values),
                    'Max': max(values)
                })
        
        param_min_max_df = pd.DataFrame(data)

        print(colorize(f'\n\nParameter range of top {str(self.top_percent_trials)}% trials\n', 'LIGHTYELLOW_EX'), param_min_max_df)

        return best_trial_df, param_min_max_df

        

    def _plot_param_distributions(self, df, hue='best_trials'):
        df = df.drop(columns=['trial_number'])
        # Get list of numerical and categorical columns excluding trial_number
        num_cols = df.select_dtypes(include=['number']).columns.drop('best_trials')
        cat_cols = df.select_dtypes(include=['object']).columns

        # Create a figure with subplots
        num_plots = len(num_cols)
        cat_plots = len(cat_cols)
        total_plots = num_plots + cat_plots
        ncols = min((total_plots+1)//2, 12)  # Maximum of 12 columns
        nrows = (total_plots + ncols - 1) // ncols  # Calculate number of rows dynamically
        
        x_size = ncols*3 if ncols*3 >= 12 else 12
        y_size = nrows*4 if ncols*3 >= 10 else 10
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(x_size, y_size))

        # Add a general title
        fig.suptitle(f'Parameter Distributions Of Top {self.top_percent_trials}% Best Trials', fontsize=16)

        # Flatten axes array for easier indexing
        axes = axes.flatten()

        # Plot numerical columns
        for i, col in enumerate(num_cols):
            sns.scatterplot(x='best_trials', y=col, data=df, hue=hue, ax=axes[i], legend=False)
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Best trials')
            axes[i].set_ylabel(col)

        # Plot categorical columns
        for i, col in enumerate(cat_cols):
            sns.stripplot(x='best_trials', y=col, data=df, hue=hue, ax=axes[num_plots+i], dodge=True)
            axes[num_plots+i].set_title(f'{col}')
            axes[num_plots+i].set_xlabel('Best trials')
            axes[num_plots+i].set_ylabel(col)

        # Obtain handles and labels from the first subplot that has a legend
        handles, labels = None, None
        for ax in axes:
            if ax.get_legend_handles_labels()[0]:  # Check if legend is not empty
                if not handles:
                    handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

        # Add a single hue legend for the whole plot
        if handles and labels:
            fig.legend(handles, labels, loc='center right', title='Best trials', fontsize='small', bbox_to_anchor=(1.0, 0.5))

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 0.90, 0.95])

        fig.savefig("best_trials_parameter_distributions.png") 
       # plt.tight_layout()
        # Display the figure
        plt.ion()
        plt.draw()
        plt.pause(30)
        plt.ioff()
        plt.close()