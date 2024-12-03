from pathlib import Path
import SP
import os
import configparser

# constants
reffolder = Path(SP.__file__).parent.parent
CONFIG_FILE  = os.path.join(reffolder, "config.cfg")

def load_config(file_path):
    """
    Loads the configuration file.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    else:
        print(f"Loaded configuration file {file_path}!")
    config.read(file_path)
    return config

# Load the configuration
try:
    cfg = load_config(SP.CONFIG_FILE)
except FileNotFoundError as e:
    print(e)
    exit()
