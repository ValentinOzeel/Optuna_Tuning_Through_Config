import os
from pydantic import BaseModel, field_validator, ValidationInfo
from typing import List, Callable

class OptunaPydantic(BaseModel):
    objective: Callable
    optuna_config_path: str
    metrics_to_optimize: List[str]
    directions: List[str]
    n_trials: int
    top_percent_trials: int

    @field_validator('objective')
    @classmethod
    def validate_callable(cls, value, info: ValidationInfo):
        if not isinstance(value, Callable):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {Callable} instance.")
        return value

    @field_validator('optuna_config_path')
    @classmethod
    def validate_str(cls, value, info: ValidationInfo):
        if not isinstance(value, str):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {str} instance.")
        return value

    @field_validator('optuna_config_path')
    @classmethod
    def validate_path(cls, value, info: ValidationInfo):
        if not os.path.exists(value):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not a valid path.")
        return value


    @field_validator('metrics_to_optimize', 'directions')
    @classmethod
    def validate_list_of_str(cls, value, info: ValidationInfo):
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
        if not isinstance(value, int):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {int} instance.")
        return value  
    
    
    
    @field_validator('top_percent_trials')
    @classmethod
    def validate_between_0_100(cls, value, info: ValidationInfo):
        if not (value >= 0 and value <= 100):
            raise ValueError(f"Input {info.field_name} -- {value} -- is not an {int} instance.")
        return value  