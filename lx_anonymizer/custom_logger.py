import logging
import sys

def get_logger(name):
    """
    Create a logger with the given name.
    
    Args:
        name (str): The name of the logger.
    
    Returns:
        logging.Logger: A logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

# Create a logger for the module
logger = get_logger(__name__)
