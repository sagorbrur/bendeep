import logging
logger = logging.getLogger(__name__)

try:
    import torch
    _torch_available = True  
    logger.info("PyTorch version {} available.".format(torch.__version__))
except ImportError:
    _torch_available = False  

def is_torch_available():
	return _torch_available
