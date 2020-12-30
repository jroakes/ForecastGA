#! /usr/bin/env python
# coding: utf-8
#

"""HWAAS Model"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from base import BaseModel


class HWAAS_Model(BaseModel):
    """HWAAS Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        for i in range(2):
            use_boxcox = [True, False][i]
            try:
                self.model = ExponentialSmoothing(
                    train,
                    seasonal_periods=seasons,
                    trend="add",
                    seasonal="add",
                    damped=True,
                ).fit(use_boxcox=use_boxcox)
                break
            except:
                continue

    def forecast(self):
        self.prediction = model.forecast(forecast_len)
