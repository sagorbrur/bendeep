

__version__="1.2"

import os
import logging
logger = logging.getLogger(__name__)


from .utils import is_torch_available

if is_torch_available():
    #import
    from bendeep import sentiment
    from bendeep import translation
    
if not is_torch_available():
    logger.warning("Please install pytorch")

