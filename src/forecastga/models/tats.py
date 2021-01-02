#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: TATS Model"""

from tbats import TBATS

from forecastga.helpers.data import get_unique_N, season_list

from forecastga.models.base import BaseModel


class TATS_Model(BaseModel):
    """TATS Model Class"""

    def __init__(self, config):
        super().__init__(config)

    def train(self, **kwargs):
        bat = TBATS(
            seasonal_periods=list(get_unique_N(season_list(self.train_df), 1)),
            use_arma_errors=False,
            use_trend=True,
        )
        self.model = bat.fit(self.train_df)

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
