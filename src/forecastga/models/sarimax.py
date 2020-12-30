#! /usr/bin/env python
# coding: utf-8
#

"""ARIMA Model"""

from base import BaseModel

class ARIMA_Model():
    """ARIMA Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def forecast(self):
            raise NotImplementedError
