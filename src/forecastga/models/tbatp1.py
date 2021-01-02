#! /usr/bin/env python
# coding: utf-8
#

"""ForecastGA: TBATP1 Model (TBATS1 but Seasonal Inference is Hardcoded by Periodicity)"""

from tbats import TBATS

from forecastga.models.base import BaseModel


class TBATP1_Model(BaseModel):
    """TBATP1 Model Class"""

    def __init__(self, config):
        super().__init__(config)

    def train(self, **kwargs):
        bat = TBATS(
            seasonal_periods=[self.periods],
            use_arma_errors=False,
            use_box_cox=True,
            use_trend=True,
        )
        self.model = bat.fit(self.train_df)

    def forecast(self):
        self.prediction = self.model.forecast(self.forecast_len)
