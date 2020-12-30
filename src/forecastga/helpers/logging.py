#! /usr/bin/env python
# coding: utf-8
#

"""Logging Utility"""

import logging
from logging import DEBUG, INFO, ERROR, Formatter, getLogger  # noqa: unused-import


# file output
FILE_HANDLER = logging.FileHandler(filename="atspy.error.log")

FILE_HANDLER.setFormatter(
    Formatter("%(asctime)s [%(levelname)s]" "  %(name)s,%(lineno)s  %(message)s")
)
FILE_HANDLER.setLevel(DEBUG)

# console output
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(ERROR)
CONSOLE_HANDLER.setFormatter(Formatter("%(message)s"))

SDCT_LOGGER = getLogger('atspy')

# add handlers
SDCT_LOGGER.addHandler(CONSOLE_HANDLER)
SDCT_LOGGER.addHandler(FILE_HANDLER)
SDCT_LOGGER.setLevel(DEBUG)

logging.captureWarnings(True)


def get_logger(log_name, level=DEBUG):
    """
    :param level:   CRITICAL = 50
                    ERROR = 40
                    WARNING = 30
                    INFO = 20
                    DEBUG = 10
                    NOTSET = 0
    :type log_name: str
    :type level: int
    """
    module_logger = SDCT_LOGGER.getChild(log_name)
    if level:
        module_logger.setLevel(level)
    return module_logger
