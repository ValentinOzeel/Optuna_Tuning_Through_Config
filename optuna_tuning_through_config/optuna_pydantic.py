import os
from pydantic import BaseModel, field_validator, ValidationInfo
from typing import List, Callable

class OptunaPydantic(BaseModel):
    """
    A Pydantic model for validating the inputs to the Optuna hyperparameter tuning.

    Attributes:
        objective (Callable): The objective function to be optimized.
        optuna_config_path (str): Path to the Optuna configuration file.
        metrics_to_optimize (List[str]): List of metric names to optimize.
        directions (List[str]): List of optimization directions (e.g., "minimize" or "maximize").
        n_trials (int): Number of trials for the Optuna study.
        top_percent_trials (int): Top percentage of trials to consider for further analysis.
    """
    
    objective: Callable
    optuna_config_path: str
    metrics_to_optimize: List[str]
    directions: List[str]
    n_trials: int
    top_percent_trials: int

    @field_validator('objective')
    @classmethod
    def validate_callable(cls, value, info: ValidationInfo):
        """
        Validates that the 'objective' field is a callable function.
        Args:
            value (Callable): The value to be validated.
            info (ValidationInfo): Additional information about the field being validated.
        Returns:
            Callable: The validated callable.
        """
        if not isinstance(value, Callable):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {Callable} instance.")
        return value

    @field_validator('optuna_config_path')
    @classmethod
    def validate_str(cls, value, info: ValidationInfo):
        """
        Validates that the 'optuna_config_path' field is a string.
        Args:
            value (str): The value to be validated.
            info (ValidationInfo): Additional information about the field being validated.
        Returns:
            str: The validated string.
        """
        if not isinstance(value, str):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {str} instance.")
        return value

    @field_validator('optuna_config_path')
    @classmethod
    def validate_path(cls, value, info: ValidationInfo):
        """
        Validates that the 'optuna_config_path' field points to an existing file.
        Args:
            value (str): The path to be validated.
            info (ValidationInfo): Additional information about the field being validated.
        Returns:
            str: The validated path.
        """
        if not os.path.exists(value):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not a valid path.")
        return value


    @field_validator('metrics_to_optimize', 'directions')
    @classmethod
    def validate_list_of_str(cls, value, info: ValidationInfo):
        """
        Validates that the input is a list of strings and is not empty.
        Args:
            value (List[str]): The list to be validated.
            info (ValidationInfo): Additional information about the field being validated.
        Returns:
            List[str]: The validated list of strings.
        """
        if not isinstance(value, List):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not a {List} instance.")
        
        if not value:
            raise ValueError(f"Input {info.field_name} -- {value} -- is empty.")
        
        for v in value:
            if not isinstance(v, str):
                raise ValueError(f"Input {info.field_name} -- {value} -- should contain {str} instances only.")
        return value
    
    @field_validator('n_trials', 'top_percent_trials')
    @classmethod
    def validate_int(cls, value, info: ValidationInfo):
        """
        Validates that the input is an integer.
        Args:
            value (int): The value to be validated.
            info (ValidationInfo): Additional information about the field being validated.
        Returns:
            int: The validated integer.
        """
        if not isinstance(value, int):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {int} instance.")
        return value  
    
    
    
    @field_validator('top_percent_trials')
    @classmethod
    def validate_between_0_100(cls, value, info: ValidationInfo):
        """
        Validates that the 'top_percent_trials' is between 0 and 100.
        Args:
            value (int): The value to be validated.
            info (ValidationInfo): Additional information about the field being validated.
        Returns:
            int: The validated integer between 0 and 100.
        """
        if not (value >= 0 and value <= 100):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {int} instance.")
        return value  