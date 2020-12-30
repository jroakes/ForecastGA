#! /usr/bin/env python
# coding: utf-8
#

"""TBATS2 Model: TBATS1 With Two Seasonal Periods"""

from tbats import TBATS

from atspy.etc.helpers import get_unique_N, season_list

from base import BaseModel

class TBATS2_Model(BaseModel):
    """TBATS2 Model Class"""

    def __init__(self):
        raise NotImplementedError

    def dataframe(self):
        raise NotImplementedError

    def train(self):
        bat = TBATS(
            seasonal_periods=list(get_unique_N(season_list(train), 2)),
            use_arma_errors=False,
            use_box_cox=True,
            use_trend=True,
        )
        self.model = bat.fit(train)

    def forecast(self):
        self.forecast = model.forecast(forecast_len)
