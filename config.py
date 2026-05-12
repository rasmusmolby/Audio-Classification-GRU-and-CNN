import yaml
import copy

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)