from pathlib import Path
import os
import configparser

# Constants
reffolder = Path(__file__).parent  # Root folder of the spheres_in_cube module
CONFIG_FILE = os.path.join(reffolder, "config_Rd.cfg")  # Path to config_Rd.cfg in spheres_in_Rd

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
    cfg2 = load_config(CONFIG_FILE)  # Load config_Rd.cfg from spheres_in_Rd folder
except FileNotFoundError as e:
    print(e)
    exit()

# Expose `cfg` for other modules to use
__all__ = ["cfg2"]