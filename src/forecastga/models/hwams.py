#! /usr/bin/env python
# coding: utf-8
#

"""HWAMS Model"""

from base import BaseModel

class HWAMS_Model(BaseModel):
    """HWAMS Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):

        for i in range(3):

            params = [  {'trend': 'add', 'seasonal': 'mul', 'use_boxcox': True},
                        {'trend': 'add', 'seasonal': 'mul', 'use_boxcox': False},
                        {'trend': None, 'seasonal': 'add', 'use_boxcox': False}]
            try:
                self.model = ExponentialSmoothing(
                    train,
                    seasonal_periods=seasons,
                    trend=params[i]['trend'],
                    seasonal=params[i]['seasonal'],
                    damped=True,
                ).fit(use_boxcox=params[i]['use_boxcox'])
                break

            except:
                continue


    def forecast(self):
        self.forecast = self.model.forecast(forecast_len)
