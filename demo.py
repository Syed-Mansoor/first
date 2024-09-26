from diamond.exception.exception import DiamondException
from diamond.logger import logging
import sys
try:
    a = 1/0
    logging.info("Error occured here")
except Exception as e:
    raise DiamondException(e,sys)