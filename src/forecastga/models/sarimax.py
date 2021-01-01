#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: SARIMAX Model"""

from forecastga.models.base  import BaseModel


class SARIMAX_Model:
    """ARIMA Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def forecast(self):
        raise NotImplementedError
