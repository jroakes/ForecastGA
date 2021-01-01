#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: HWAAS Model"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from base import BaseModel


class HWAAS_Model(BaseModel):
    """HWAAS Model Class"""

    def train(self, **kwargs):
        for i in range(2):
            use_boxcox = [True, False][i]
            try:
                self.model = ExponentialSmoothing(
                    self.train_df,
                    seasonal_periods=self.seasons,
                    trend="add",
                    seasonal="add",
                    damped_trend=True
                ).fit(use_boxcox=use_boxcox)
                break
            except:
                continue

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
