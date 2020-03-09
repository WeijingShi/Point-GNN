"""This file implements configuration functions. """

import json

def load_config(filename):
    """Load a configuration file."""
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def save_config(filename, config):
    """Save a configuration file. """
    with open(filename, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

def load_train_config(filename):
    """Load a configuration file."""
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def save_train_config(filename, train_config):
    """Save a configuration file. """
    with open(filename, 'w') as f:
        json.dump(train_config, f, sort_keys=True, indent=4)
