from pathlib import Path
import diffuse_boost
import os
import configparser

# constants
print("file: ", diffuse_boost.__file__)
reffolder = Path(diffuse_boost.__file__).parent.parent
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
    config.read(file_path, encoding="utf8")
    return config

# Load the configuration
try:
    print("here")
    cfg = load_config(CONFIG_FILE)
except FileNotFoundError as e:
    print(e)
    exit()