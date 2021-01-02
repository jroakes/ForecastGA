#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: HWAMS Model"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecastga.models.base import BaseModel


class HWAMS_Model(BaseModel):
    """HWAMS Model Class"""

    def __init__(self, config):
        super().__init__(config)

    def train(self, **kwargs):

        for i in range(3):

            params = [
                {"trend": "add", "seasonal": "mul", "use_boxcox": True},
                {"trend": "add", "seasonal": "mul", "use_boxcox": False},
                {"trend": None, "seasonal": "add", "use_boxcox": False},
            ]
            try:
                self.model = ExponentialSmoothing(
                    self.train_df,
                    seasonal_periods=self.seasons,
                    trend=params[i]["trend"],
                    seasonal=params[i]["seasonal"],
                    damped_trend=True,
                ).fit(use_boxcox=params[i]["use_boxcox"])
                break

            except:
                continue

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
