from colorama import init, Fore, Back, Style
init() # Initialize Colorama to work on Windows
import yaml


class ConfigLoad():
    def __init__(self, path:str):
        self.path = path
        with open(self.path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def get_config(self):
        return self.config
    
def get_config(path:str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def colorize(to_print, color):
    return f"{getattr(Fore, color) + to_print + Style.RESET_ALL}"




