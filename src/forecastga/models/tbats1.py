#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: TBATS1 Model (Trend, Seasonal-one, and Box Cox)"""

from tbats import TBATS

from base import BaseModel


class TBATS1_Model(BaseModel):
    """TBATS1 Model Class"""

    def train(self, **kwargs):
        bat = TBATS(
            seasonal_periods=[self.seasons],
            use_arma_errors=False,
            use_box_cox=True,
            use_trend=True,
        )
        self.model = bat.fit(self.train_df)

    def forecast(self):
        self.prediction = model.forecast(self.forecast_len)
