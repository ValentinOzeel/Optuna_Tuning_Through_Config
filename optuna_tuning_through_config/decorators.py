
import time
from functools import wraps

from .secondary_module import colorize


def trials_counter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Ensure 'trials_counter' attribute exists
        if not hasattr(self, 'trials_counter'):
            setattr(self, 'trials_counter', 0)

        # Increment the counter
        self.trials_counter += 1
        print('\nOptuna trial number: ', colorize(str(self.trials_counter), 'RED'))
        
        return func(self, *args, **kwargs)

    return wrapper


def timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time = f'{elapsed_time:.4f}'
        
        print('Trial ', colorize(str(self.trials_counter), 'RED'), ' took -- ', colorize(str(elapsed_time), 'LIGHTRED_EX'), ' -- seconds to execute.')
        return result

    return wrapper

