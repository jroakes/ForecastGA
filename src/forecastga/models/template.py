#! /usr/bin/env python
# coding: utf-8
#

"""ARIMA Model"""

from forecastga.models.base import BaseModel


class ARIMA_Model:
    """ARIMA Model Class"""

    def __init__(self, config):
        super().__init__(config)

    def train(self):
        raise NotImplementedError

    def forecast(self):
        raise NotImplementedError
