import os
from box.exceptions import BoxValueError
import yaml
from src.cnnClassifier.logger import logging
from box import ConfigBox
from pathlib import Path
import torch



def read_yaml(path_to_yaml: Path) -> ConfigBox:
   
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

def create_directories(path_to_directories: list, verbose=True):
   
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")





def save_model(path: Path, model):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)