# encoding: utf-8


import re
import pkg_resources

from . import (
    auth,
    utils,
    account,
    blueprint,
    columns,
    errors,
    query,
    segments,
)
from .auth import authenticate, authorize, revoke
from .blueprint import Blueprint

__version__ = pkg_resources.get_distribution("googleanalytics").version
