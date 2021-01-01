#! /usr/bin/env python
# coding: utf-8
#

"""TForecastGA: BATS2 Model (TBATS1 With Two Seasonal Periods)"""

from tbats import TBATS

from forecastga.helpers.data import get_unique_N, season_list

from base import BaseModel


class TBATS2_Model(BaseModel):
    """TBATS2 Model Class"""

    def train(self, **kwargs):
        bat = TBATS(
            seasonal_periods=list(get_unique_N(season_list(self.train_df), 2)),
            use_arma_errors=False,
            use_box_cox=True,
            use_trend=True,
        )
        self.model = bat.fit(self.train_df)

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
