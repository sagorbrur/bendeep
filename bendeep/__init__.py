

__version__="1.0"

import os
import logging
logger = logging.getLogger(__name__)


from .utils import is_torch_available

if is_torch_available():
    #import
    pass
if not is_torch_available():
    logger.warning("Please install pytorch")

