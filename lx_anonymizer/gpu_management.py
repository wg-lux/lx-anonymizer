import torch

from .custom_logger import get_logger

logger = get_logger(__name__)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared GPU memory.")
        

