#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: HWAMS Model"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from base import BaseModel


class HWAMS_Model(BaseModel):
    """HWAMS Model Class"""

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
                    damped=True,
                ).fit(use_boxcox=params[i]["use_boxcox"])
                break

            except:
                continue

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
