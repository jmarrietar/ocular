# -*- coding: utf-8 -*-

from functools import wraps
import logging
from time import time
import sys

def log_entry_exit(fn):
    """Log entry and exit into function """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.debug("Entering {:s}".format(fn.__name__))
        result = fn(*args, **kwargs)
        logger.debug("Finished {:s}".format(fn.__name__))
        return result
    return wrapper

def timed(func):
    """Logs the execution time for the decorated function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        start = time()
        result = func(*args, **kwargs)
        end = time()        
        logger.info("Execution Time for {}: {}s".format(func.__qualname__, round(end - start, 2)))
        return result
    return wrapper