#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: ARIMA Model"""

import pmdarima as pm

from base import BaseModel


class ARIMA_Model(BaseModel):
    """ARIMA Model Class"""

    def train(self, **kwargs):
        self.model = pm.auto_arima(self.train_df, seasonal=True, m=self.seasons)

    def forecast(self):
        self.prediction = self.model.predict(self.forecast_len)
