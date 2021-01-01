#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: SARIMA Model"""

from base import BaseModel


class SARIMA_Model:
    """SARIMA Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def forecast(self):
        raise NotImplementedError
