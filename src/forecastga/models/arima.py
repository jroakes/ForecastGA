#! /usr/bin/env python
# coding: utf-8
#

"""ARIMA Model: Automated ARIMA Modeling"""

import pmdarima as pm

from base import BaseModel


class ARIMA_Model(BaseModel):
    """ARIMA Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        self.model = pm.auto_arima(train, seasonal=True, m=seasons)

    def forecast(self):
        self.prediction = self.model.predict(forecast_len)
