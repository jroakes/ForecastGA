#! /usr/bin/env python
# coding: utf-8
#

"""TBATP1 Model: TBATS1 but Seasonal Inference is Hardcoded by Periodicity"""

from tbats import TBATS

from base import BaseModel

class TBATP1_Model(BaseModel):
    """TBATP1 Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        bat = TBATS(
            seasonal_periods=[periods],
            use_arma_errors=False,
            use_box_cox=True,
            use_trend=True,
        )
        self.model = bat.fit(train)

    def forecast(self):
        self.forecast = model.forecast(forecast_len)
