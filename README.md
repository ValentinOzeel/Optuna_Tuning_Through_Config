# Optuna Hyperparameter Tuning with Configuration

## Overview

This repository provides a modular framework for hyperparameter tuning using Optuna, configured through a YAML file. The primary goal is to simplify the hyperparameter tuning process by using configuration files to define trial parameters and leveraging implemented features such as parameter grids, trial skipping, and best trials parameter analysis.

## Key Features:

    Configuration-Based Tuning: Define hyperparameter search spaces and settings in a configuration file.
    suggest_grid Feature: Specify a grid of values for certain parameters to be chosen from, enabling more controlled and specific tuning.
    Trial Skipping: Use Ctrl+C to skip trials interactively during the optimization process.
    Parameter Analysis and plots: Automatically generate information (dataframe) regarding parameter ranges for the best trials and plot their distributions.

## Problem Statement

Hyperparameter tuning is crucial for optimizing machine learning models, but it can be cumbersome. This repository aims to:

    Simplify the setup and execution of hyperparameter tuning.
    Provide an easy-to-use interface for defining search spaces and configurations.
    Enhance the tuning process with useful features like trial skipping and grid-based suggestions.
    Offer insights into the best hyperparameter settings through best trials parameter ranges analysis and visualization.



## Installation and Setup

### Steps to Install and Run the Package

Install poetry:
https://python-poetry.org/docs/

Clone the Repository:
'git clone https://github.com/ValentinOzeel/optuna_tuning_through_config.git'
'cd optuna_tuning_through_config'
Activate your virtual environment with your favorite environment manager such as venv or conda (or poetry will create one)
'poetry install'



## Configuration

### Create a YAML configuration file with the following structure (Do not change the OPTUNA_PARAMS and OPTUNA_FROZEN_PARAMS key names):

'  OPTUNA_PARAMS:  
    - ['categorical_entries', 'suggest_categorical', ['aa', 'bb', 'cc'], {}]
    - ['int_number', 'suggest_int', [500, 800], {}]
    - ['float_number', 'suggest_float', [0.001, 0.003], {'log': True}]
    # suggest grid feature
    - ['int_number_grid', 'suggest_grid', [65, 78, 87, 99, 103], {}]


  OPTUNA_FROZEN_PARAMS:
    frozen_number: 10
'


## Example Objective Function

### Define your objective function to be optimized by Optuna:


'def user_objective(params):
    # Example: optimizing a simple quadratic function
    return ((params['frozen_number'] + params['int_number'] + params['int_number_grid']) * params['float_number']) ** 2'

## Running the Tuning

### Create a Python script to run the tuning process:
### Return all study's trials, study's best_trials, dataframe with top X% (top_percent_trials kwarg) trials and their parameters, parameter distribution plots of top X% trials, dataframe with top X% trial parameter ranges.

'
from optuna_tuning_through_config.optuna_through_config import OptunaFinetuning
 
 #Example usage
 optuna_finetuning = OptunaFinetuning(
     objective=user_objective,
     optuna_config_path='Path/To/Your/optuna_config.yml',
     metrics_to_optimize=['quadratic_func_value'],
     directions=['maximize'],
     n_trials=100,
     top_percent_trials=20
 )
 
 #Run the tuning by calling the instance
 study_trials, best_trials, best_trials_param_range_df, best_trials_param_distrib_plot, param_min_max_df = optuna_finetuning()
'


## Implemented Features
- Parameter from config
Fill your configuration file and simply use your parameters in your objective function.

- suggest_grid
The suggest_grid feature allows you to define a fixed set of values for a parameter. Optuna will choose from these values instead of sampling from a continuous range. This is useful for parameters where only specific values are meaningful.

- Trial Skipping
During the execution of trials, you can skip the current trial by pressing Ctrl+C. This is particularly useful if you realize that the current trial is not worth pursuing further.
Parameter Analysis

- Analysis
After the tuning process, the framework provides information about the parameter ranges of the top-performing trials. It also generates plots to visualize the distribution of these parameters, helping you understand the impact of different hyperparameters on the performance.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions or find any bugs.

