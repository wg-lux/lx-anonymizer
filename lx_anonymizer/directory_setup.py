from pathlib import Path, PosixPath
import logging
from .custom_logger import get_logger
import os

logger = get_logger(__name__)

'''
Main Directory Setup

The functions in this script define where the storage for anonymized data
and intermediate results will occur.

Functions:

- create_main_directory:
  - The main directory stores the final anonymized results as well as structured study and training data ready for export.
  
- create_results_directory:
    - The results directory stores the final anonymized images. This directory is cleaned up manually.
    
- create_temp_directory:
  - The temp directory stores intermediate results during the anonymization process. It is cleaned up regularly.
  
- create_blur_directory:
  - The blur directory stores blurred images. This directory is also cleaned up regularly.

To change the default installation paths, update these variables:

- main_directory
- temp_directory
'''


# Default directory paths for main and temp directories
# 
# CHANGE THIS IF YOU WANT TO USE A DIFFERENT DIRECTORY

default_main_directory = Path("./data/")
default_temp_directory = Path("./temp/")

# Check if environment variables are set and override if available

if os.getenv("AGL_ANONYMIZER_DEFAULT_MAIN_DIR"):
    MAIN_DIR = Path(os.getenv("AGL_ANONYMIZER_DEFAULT_MAIN_DIR"))
else:
    MAIN_DIR = default_main_directory

if os.getenv("AGL_ANONYMIZER_DEFAULT_TEMP_DIR"):
    TEMP_DIR_ROOT = Path(os.getenv("AGL_ANONYMIZER_DEFAULT_TEMP_DIR"))
else:
    TEMP_DIR_ROOT = default_temp_directory

from typing import List

def _str_to_path(path:str):
    if isinstance(path, str):
        path = Path(path)
        
    return path

def create_directories(directories:List[Path]=None)->List[Path]:
    """
    Helper function.
    Creates a list of directories if they do not exist.
    
    Args:
        directories (list): A list of directory paths to create.
    """

    if not directories:
        directories = [
            MAIN_DIR,
            TEMP_DIR_ROOT
        ]
        
    else: 
        directories = [_str_to_path(directory) for directory in directories]

    for dir_path in directories:
        if dir_path.exists():
            logger.info(f"Directory already exists: {dir_path}")

        else:
            dir_path.mkdir(parents = True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
           
    return directories
            
def create_main_directory(default_main_directory:Path = None):
    """
    Creates the main directory in a writable location (outside the Nix store).
    
    Args:
        directory (str): The path where the main directory will be created.
                         Defaults to `default_main_directory`.
    
    Returns:
        str: The path to the main directory.
    """
    # check if string path, make to path object if necessary
    
    
    if not default_main_directory:
        default_main_directory = MAIN_DIR
    else:
        default_main_directory = _str_to_path(default_main_directory)
    
    if default_main_directory.exists():
        logger.debug(f"Using default main directory: {default_main_directory.as_posix()}")

    else:
        logger.debug(f"Creating main directory, directory at {default_main_directory} not found")
        create_directories([default_main_directory])
        logger.info(f"Main directory created at {default_main_directory}")

    return default_main_directory

def create_results_directory(default_main_directory:Path=None) -> Path:
    if not default_main_directory:
        default_main_directory = MAIN_DIR
    else:
        default_main_directory = _str_to_path(default_main_directory)
    
    results_dir = Path(default_main_directory) / 'results'
    if results_dir.exists():
        logger.debug("Using default blur directory settings")
        return results_dir  # Add return here
    else:
        logger.info(f"Creating blur directory, directory at {results_dir} not found")
        results_dir = Path(default_main_directory) / 'results'
        create_directories([results_dir])
        logger.debug(f"Anonymization results directory created at {results_dir}")
        return results_dir

def create_model_directory(default_main_directory:Path=None) -> Path:
    """_summary_

    Args:
        default_main_directory (Path, optional): _description_. Defaults to None.

    Returns:
        Path: _description_
    """
    if not default_main_directory:
        default_main_directory = MAIN_DIR
    else:
        default_main_directory = _str_to_path(default_main_directory)
        
    models_dir = Path(default_main_directory) / 'models'
    
    if models_dir.exists():
        logger.debug(f"found models directory at:{models_dir}")
        return models_dir

    else:
        logger.debug(f"Creating models directory, directory at {models_dir} not found")
        create_directories([models_dir])
        logger.info(f"Models directory created at {models_dir}")
        return models_dir

def create_temp_directory(default_temp_directory:Path=None, default_main_directory:Path=None):
    """
    Creates 'temp' and 'csv' directories in the given temp and main directories.
    
    Args:
        temp_directory (str): The path where the temp directory will be created.
                              Defaults to `default_temp_directory`.
        main_directory (str): The main directory path, where the csv directory will be created.
                              If not provided, it will use the result of create_main_directory.
    
    Returns:
        tuple: Paths to temp_dir, base_dir, and csv_dir.
    """
    
    if not default_temp_directory:
        default_temp_directory = TEMP_DIR_ROOT
    else:
        default_temp_directory = _str_to_path(default_temp_directory)
    
    if not default_main_directory:
        default_main_directory = MAIN_DIR
    else:
        default_main_directory = _str_to_path(default_main_directory)    
    
    
    temp_dir = Path(default_temp_directory) / 'temp'
    csv_dir = Path(default_main_directory) / 'csv_training_data'
    # print("Using default temp and main directory settings")   
    
    if temp_dir.exists() and csv_dir.exists():
        return temp_dir, default_main_directory, csv_dir 
    
    else:
        logger.debug(f"Creating temp and csv directories, directories at {temp_dir} and {csv_dir} not found")
        create_directories([temp_dir, csv_dir])
        logger.info(f"Temp and csv directories created at {temp_dir} and {csv_dir}")
        return temp_dir, default_main_directory, csv_dir

def create_blur_directory(default_main_directory:Path=None) -> Path:
    if not default_main_directory:
        default_main_directory = MAIN_DIR
    else:
        default_main_directory = _str_to_path(default_main_directory)
    
    blur_dir = Path(default_main_directory) / 'blurred_results'
    if blur_dir.exists():
        logger.info(f"Using default blur directory settings at {blur_dir}")
        return blur_dir 
    else:
        logger.debug(f"Creating blur directory, directory at {blur_dir} not found")
        blur_dir = default_main_directory / 'blurred_results'
        blur_dir = Path(blur_dir)

        create_directories([blur_dir])
        logger.info(f"Blur directory created at {blur_dir}")
        return blur_dir

        
        
