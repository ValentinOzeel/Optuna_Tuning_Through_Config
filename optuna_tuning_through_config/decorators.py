
import time
from functools import wraps
import threading
import keyboard
from .secondary_module import colorize



def timer_and_counter(func):
    """
    Decorator to measure the execution time of a function and increment a trial counter.
    Args:
        func (Callable): The function to be decorated.
    Returns:
        Callable: The wrapped function with added timer and counter functionality.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        
        print('\n\nOptuna trial number: ', colorize(str(self.trials_counter), 'RED'))
        
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time = f'{elapsed_time:.4f}'
        
        print('Trial ', colorize(str(self.trials_counter), 'RED'), ' took -- ', colorize(str(elapsed_time), 'LIGHTRED_EX'), ' -- seconds to execute.')
        
        # Increment the counter
        self.trials_counter += 1
        
        return result

    return wrapper




def skip_trial_on_keypress_n(func):
    """
    Decorator to allow skipping the trial on a ctrl+c keypress event.
    Args:
        func (Callable): The function to be decorated.
    Returns:
        Callable: The wrapped function with keypress event handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        # Define the key press handler
        def on_keypress(event):
            if event.name == 'SIGINT':  
                raise KeyboardInterrupt()
            
        # Register the key press handler
        keyboard.on_press(on_keypress)

        try:
            # Call the original objective function
            result = func(*args, **kwargs)
            
        except KeyboardInterrupt as e:
            print(colorize('--- Skip trial requested ---', 'RED'))
            result = None  # You can decide how to handle the skipped trial
        finally:
            # Unregister the key press handler
            keyboard.unhook_all()

        return result

    return wrapper



