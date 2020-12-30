#! /usr/bin/env python
# coding: utf-8
#

"""TATS Model"""

from tbats import TBATS

from atspy.etc.helpers import get_unique_N

from base import BaseModel

class TATS_Model(BaseModel):
    """TATS Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        bat = TBATS(
            seasonal_periods=list(get_unique_N(season_list(train), 1)),
            use_arma_errors=False,
            use_trend=True,
        )
        self.model = bat.fit(train)

    def forecast(self):
        self.forecast = model.forecast(forecast_len)
